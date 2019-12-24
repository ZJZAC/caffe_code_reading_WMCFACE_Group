#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {        //可供请求的动作，在训练时可选择提前终止或保存快照而不退出
    enum Enum {     //动作的枚举
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;    //回滚，训练出现中断后可回到上次断点处继续训练
/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>       //在网络中执行优化
class Solver {
 public:
  explicit Solver(const SolverParameter& param);     //solver的主要功能
  explicit Solver(const string& param_file);        //内联函数
  void Init(const SolverParameter& param);    //solver参数初始化
  void InitTrainNet();    //每个Solver中包含一个训练网络对象和一个测试网络对象
  void InitTestNets();    //网络对象初始化

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);    //设定solver应该执行什么动作
  SolverAction::Enum GetRequestedAction();      //枚举得到的请求动作
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  //依次调用函数Restore、Step、Snapshot，然后运行net_的前向传播函数ForwardPrefilled，最后调用TestAll函数
  virtual void Solve(const char* resume_file = NULL);    //default为0，传入非0的iter到预训练网络
  inline void Solve(const string& resume_file) { Solve(resume_file.c_str()); }  //参数类型转换，string转换成char* 
  void Step(int iters);   //重复运行net前向传播反向传播计算,期间会调用函数TestAll、ApplyUpdate、Snapshot及类Callback两个成员函数 
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);      //存储函数实现如何存储solver到快照模型中
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();      //主要是基本的快照功能，存储学习的网络
  virtual ~Solver() {}     // 虚析构函数，当基类析构函数不是虚函数时，delete时函数只调用了基类的析构函数，这样如果派生类析构函数有需要对内存的释放(先子类后基类)时，不会释放子类内存
  inline const SolverParameter& param() const { return param_; }  //获得solver的参数
  inline shared_ptr<Net<Dtype> > net() { return net_; }     //获得train net
  //指向Net类型的智能指针（shared_ptr），Solver正是通过这个指针来和网络Net来交互并完成模型的优化
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {    //获得test net
    return test_nets_;
  }
  int iter() const { return iter_; }    //获得当前迭代次数

  // Invoked at specific points during an iteration
  class Callback {         //内部callback类，仅在多卡GPU模式下使用
   protected:              //迭代过程中调用特殊点
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }    //获得callback
  void add_callback(Callback* value) {      //增加一个callback
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();       //检查保存快照文件权限
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }   //返回solver类型

  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;    //生成并且应用当前迭代更新的权值

 protected:
  string SnapshotFilename(const string& extension);     //获取快照文件名
  string SnapshotToBinaryProto();     //写proto(层的参数定义在.proto文件中)到.caffemodel
  string SnapshotToHDF5();    //写proto到HDF5文件
  
  // The test routine
  void TestAll();      //测试环节，内部循环调用test（）
  void Test(const int test_net_id = 0);   //运行测试网络，net前向传播
  //快照，通常内部调用SnapshotToBinaryProto或SnapshotToHDF5、SnapshotSolverState函数
  virtual void SnapshotSolverState(const string& model_filename) = 0;     //存储快照
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;     //以HDF5文件保存快照
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;   //以二进制文件保存快照
  void DisplayOutputBlobs(const int net_id);        //虚拟函数，只有申明没有实现
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);    //更新平滑loss

// Caffe中类的成员变量名都带有后缀"_"。这样就易于区分暂时变量和类成员变量
  SolverParameter param_;     //solver参数
  int iter_;      //当前迭代次数
  int current_step_;    //当前的step（学习率变化时候的迭代次数）
  shared_ptr<Net<Dtype> > net_;    //训练网络
  vector<shared_ptr<Net<Dtype> > > test_nets_;     //测试网络
  vector<Callback*> callbacks_;    //callback
  vector<Dtype> losses_;       //loss
  Dtype smoothed_loss_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;      //通过该函数确定是保存快照还是提前退出

  // True iff a request to stop early was received.
  bool requested_early_exit_;     //提前终止请求，若收到则为true

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;     //定时信息，便于参数调优（如GPU数量等）
  float iterations_last_;     

  DISABLE_COPY_AND_ASSIGN(Solver);    //禁止使用solver类的拷贝和赋值操作（根solver）
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
