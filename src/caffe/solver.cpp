#include <cstdio>

#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;         //设置solver动作
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {    
  if (action_request_function_) {    //如果动作请求置位，返回动作
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;    //否则返回none
}

//设计好需要优化的对象，以及用于学习的训练网络和用于评估的测试网络:
//构造函数：初始化net，调用init，有两种调用参数的方式 
//1.使用SolverParameter类型的param 
//2.使用string类型的param_file 
template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)      //构造函数，初始化net，调用init，使用SolverParameter类型的param
    : net_(), callbacks_(), requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)       //构造函数，初始化net，调用init，使用string类型的param_file 
    : net_(), callbacks_(), requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {    //初始化solver
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();     //检查是否有保存快照的权限
  if (param_.random_seed() >= 0) {     //设置随机数种子
    Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());
  }
  // Scaffolding code
  InitTrainNet();     //初始化trainnet
  InitTestNets();     //初始化testnet
  if (Caffe::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

// Load weights from the caffemodel(s) specified in "weights" solver parameter
// into the train and test nets.
template <typename Dtype>
void LoadNetWeights(shared_ptr<Net<Dtype> > net,      //载入模型权值到训练网络和测试网络
    const std::string& model_list) {      
  std::vector<std::string> model_names;   
  boost::split(model_names, model_list, boost::is_any_of(","));  //字符串分割获取模型名字
  for (int i = 0; i < model_names.size(); ++i) {
    boost::trim(model_names[i]);    //去除收尾空格
    LOG(INFO) << "Finetuning from " << model_names[i];
    net->CopyTrainedLayersFrom(model_names[i]);    //copy模型中的训好的层到net
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {       //初始化训练网络
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();        //网络数量
  const string field_names = "net, net_param, train_net, train_net_param"; 
  //区域名称，net、train_net对应要从文件中读取的参数；net_param、train_net_param对应相应类型的参数   
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;    
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;     
  NetParameter net_param;
  //得到参数，将对应的网络参数读/拷贝到net_param
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  //设置正确的训练网络状态
  NetState net_state;    
  net_state.set_phase(TRAIN);      //阶段设置为train
  net_state.MergeFrom(net_param.state());    
  net_state.MergeFrom(param_.train_state());    
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));     //调用net构造方法，重新构建网络
  for (int w_idx = 0; w_idx < param_.weights_size(); ++w_idx) {
    LoadNetWeights(net_, param_.weights(w_idx));      //循环载入网络权重
  }
}

//与train_net同理，但是test_net可以有多个
template <typename Dtype>
void Solver<Dtype>::InitTestNets() {            //初始化测试网络
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;    //同类网络数量
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;    //测试网络数量
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;   //同类网络用例数
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;   //测试网络用例数
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  //拷贝/读入测试网络的参数
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";       //标记网络参数
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);     //设置测试网络大小
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    //设置正确的测试网络状态
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));    //调用net构造方法，构造网络
    test_nets_[i]->set_debug_info(param_.debug_info());    //设置debug信息
    for (int w_idx = 0; w_idx < param_.weights_size(); ++w_idx) {
      LoadNetWeights(test_nets_[i], param_.weights(w_idx));          //权值载入测试网络
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;      //起始迭代次数
  const int stop_iter = iter_ + iters;    //结束时的迭代次数
  int average_loss = this->param_.average_loss();
  // 输出的loss为前average_loss次loss的平均值，在solver.prototxt里设置，默认为1，
  // losses存储之前的average_loss个loss，smoothed_loss为最后要输出的均值
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

  while (iter_ < stop_iter) {      //开始迭代
    // zero-init the params
    net_->ClearParamDiffs();     //清空上一次参数的梯度
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {    //判断是否需要测试
      if (Caffe::root_solver()) {
        TestAll();   
      }
      if (requested_early_exit_) {       //判断是否需要提前终止
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {    //回滚记录
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;  //判断当前迭代次数是否需要显示loss等信息
    net_->set_debug_info(display && param_.debug_info());     //设置debug信息
    // accumulate the loss and gradient
    Dtype loss = 0;  
    for (int i = 0; i < param_.iter_size(); ++i) {
    // iter_size也是在solver.prototxt里设置，实际上的 batch_size = iter_size * 网络定义里的batch_size ，
      loss += net_->ForwardBackward();    //完成前向后向传播计算
    }
    loss /= param_.iter_size();   //平均loss
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);   //更新平滑loss（定义在solver.cpp：504）
    if (display) {    //输出当前迭代信息
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_      //迭代次数、迭代速率、平滑loss等信息
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {       //网络的输出显示
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {  //回滚设置
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();     //执行梯度更新

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()         //判断是否保存快照
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {     //判断是否终止
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());       //检查当前是否是root_solver(多GPU模式下，只有root_solver才运行这一部分的代码)
  LOG(INFO) << "Solving " << net_->name();        //更新学习率策略
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;      //初始化提前结束为否

  if (resume_file) {          //判断`resume_file`这个指针是否NULL，如果不是则需要从resume_file存储的路径里读取之前训练的状态
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);          //执行实际的逐步迭代
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()              //迭代结束或者遇到系统信号提前结束后，判断是否需要在训练结束之后snapshot
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
     // 如果在`Step`函数的迭代过程中遇到了系统信号，且我们的处理方式设置为`STOP`，
     // 那么`requested_early_exit_`会被修改为true，迭代提前结束，输出相关信息
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  //优化结束后，运行附加的train和test来输出loss/outputs
  if (param_.display() && iter_ % param_.display() == 0) {    //判断是否需要输出最后的loss
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);      //对train net运行最后的前向传播，计算loss，显示

    UpdateSmoothedLoss(loss, start_iter, average_loss);    //更新并且平滑loss

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {    //判断是否需要最后的test
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {         //按网络循环，每个网络都调用test函数
    Test(test_net_id);
  }
}

//具体每个网络的测试
template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());      //检查当前是否是root_solver(多GPU模式下，只有root_solver才运行这一部分的代码)
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";      //测试网络的信息
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();    //获取动作信号
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {    //如果传入信号是保存快照，则调用Snapshot()函数保存快照
          Snapshot();
        } else if (SolverAction::STOP == request) {    //如果是stop则退出
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();    //继续获取动作信号
    }
    if (requested_early_exit_) {        //提前终止测试
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);         //执行前向传播测试
    if (param_.test_compute_loss()) {
      loss += iter_loss;            //累加loss用于后续统计
    }
    if (i == 0) {      
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);         //保存结果
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {     //测试提前终止
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  //测试结果打印
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {     //保存快照函数
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {    //保存方式
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();      //保存到二进制文件
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();       //保存到HDF5文件
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {    //检查保存快照文件的权限
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix()) 
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string& extension) {      //生成快照文件名称
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {     //以二进制形式保存快照文件
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());     
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {      //以HDF5格式保存文件
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {    //存储函数实现如何存储solver到快照模型中
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);   //调用具体的Solver的RestoreSolverStateFromHDF5来实现, 从HDF5文件来保存快照
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);   //调用具体的Solver的RestoreSolverStateFromBinaryProto来实现, 从二进制文件来保存快照
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {      //平滑loss
  if (losses_.size() < average_loss) {
    //计算要输出的smoothed_loss，如果losses里还没有存够average_loss个loss则将当前的loss插入，
    //如果已经存够了，则将之前的替换掉
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
