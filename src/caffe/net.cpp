#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_HDF5
#include "hdf5.h"
#endif  // USE_HDF5

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {            // 构造函数，使用NetParameter类型的变量初始化网络
  Init(param);
}

// param_file: Proto类型的文本文件名
// phase: 字段，网络的一种状态(TRAIN OR TEST)               // 这三种设置会在StateMeetsRule()函数中详细说明
// level: 级别，网络的一种状态
// stages: 同样为网络的一种状态，phase/level/stages三种均为网络的状态，定义在NetState消息中
template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages) {    // 创建网络，并使用从param_file中读取的网络参数和三种网络状态初始化
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);   // 从ProtoTxt文件中读取网络参数
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);              // 设置网络状态，字段（TRAIN or TEST）
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);   // 设置网络的stage
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);                                          // 初始化网络
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {   // 网络初始化函数
  // Set phase from the state.
  phase_ = in_param.state().phase();                    // 网络字段（TRAIN or TEST）
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);                 // 根据网络状态和layer的状态规则过滤部分layer，得到过滤后的网络参数
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param); // 将filtered_param复制到param，并拆分其中一些复用的blob数据，参见insert_splits.cpp
  // Basically, build all the layers and set up their connections.
  name_ = param.name();                                 // 设置网络的名称
  map<string, int> blob_name_to_idx;                    // <输出blob的名称, blob在blobs_中的索引>，只会保存输出blob的映射关系
    
  // AppendBottom()会从中删除已被用作输入的输出blob，而AppendTop()会将layer的输出blob都加入其中
  // 最终剩下的blob都认为是网络的输出，例如loss层的输出blob
  set<string> available_blobs;                          // 当前还未被当作输入的所有输出blob
  memory_used_ = 0;                                     // 内存使用大小，初始化为0
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());              // 初始化net中各项参数的大小，layer的个数
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) { // 对于网络中的各个layer
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);        // 若有layer未设置phase，则使用net本身的值设置
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id); // 当前layer的参数
    if (layer_param.propagate_down_size() > 0) {               // 设置了是否需要反向传播
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";                 // 检查设置的个数与layer的输入blob的个数是否相等
    }
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param)); // 根据layer参数创建layer
    layer_names_.push_back(layer_param.name());                        // 保存layer的名称
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false; // 该layer是否需要反向传播，下面会根据layer中输入blob和层的参数来决定该值，默认false

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();++bottom_id) { // 处理层的输入blob
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);     // 将layer的第bottom_id个输入blob添加到param中
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];            // 如果输入blob需要backward，则所在的layer也设置成需要
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {          // 处理层的输出blob
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx); // 将输出blob添加到net中
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {                      // 该layer为网络的数据输入层
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);             // 保存该输出blob的位置
        net_input_blobs_.push_back(blobs_[blob_id].get());      // 保存该输出blob的指针
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
      
    //允许创建匿名blob.允许输出blob的个数比ExactNumTopBlobs()或者MinTopBlobs()的值小
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs()); // 需要的输出blob的个数
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        // 匿名blob不可用做其他层的输入，但会存储在blobs_中
        AppendTop(param, layer_id, num_top, NULL, NULL);        // 添加到net中
      }
    }
    // After this layer is connected, set it up.
    // 调用layer的SetUp函数，检查输入输出blob的个数,调整大小等
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) { // 若输出blob对应的blob_id比权重的个数大
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0)); // 权重的个数加1，初始为0
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);  // 根据layer中的权重设置
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();                      // 累加输出blob中数据的个数
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();                             // layer参数中param参数的设置的个数
    const int num_param_blobs = layers_[layer_id]->blobs().size();               // layer中可学习的参数blob的个数
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();         // 设置的个数需小于等于参数blob的个数
    ParamSpec default_param_spec; // layer中blob参数的默认配置，例如layer参数的lr_mult_=1和decay_mult_=1等
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;                    // 使用设置的训练参数，或者使用默认配置
      const bool param_need_backward = param_spec->lr_mult() != 0;               // 学习率系数不为0，则需要反向传播
      need_backward |= param_need_backward; // layer的参数blob需要反向传播 or 输入blob需要反向传播
      layers_[layer_id]->set_param_propagate_down(param_id, param_need_backward);// 将参数blob是否反传保存到net的layers_中
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);                                    // 将layer的参数blob添加到net中
    }
    // Finally, set the backward flag
    // 设置layer是否需要反传（layer中输入blob和参数blob中任何一个需要，则该层都需要反传）
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;              // 将layer中的输出blob也设置为需要反传
      }
    }
  } // layer的第一个for循环结束
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire layer
  set<string> blobs_under_loss;                                                  // 参与loss计算的blob的名称的集合
  set<string> blobs_skip_backp;                                                  // 可跳过反向传播的blob的名称的集合
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {           // 倒序判断哪些layer对网络的loss计算有贡献
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];     // 第layer_id层的第top_id个输出blob的名称
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) { // 若该输出blob权重不为0，或者在参与计算loss的集合中能找到
        layer_contributes_loss = true;                                           // 输出blob有用，则该layer设置为对loss计算有用
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {          // 若该输出blob不在可跳过反传的集合中
        layer_skip_propagate_down = false;                                       // 则设为不可跳过
      }
      // layer中只要有任意一个输出blob参与loss计算，且不可跳过反向传播，则整个layer都是此状态，不必再判断其他输出blob了
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    // 如果之前layer设置了需要反传，但是此处是可跳过反传的（此处可跳过，说明该层的所有输出blob均可跳过反传，均不参与loss计算）
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;                      // 所有输入blob也设置为不需要反传
      }
    }
    // layer的输出blob对loss的计算无用，则layer设置为不需要反传（此处只设置了layer，后续才设置了输入参数不需要）
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {                                              // 若layer对loss计算有用
        /* 把layer的所有输入blob的名称保存到blobs_under_loss中，表示与该输入blob对应的上层layer的
           输出blob也同样参与了loss的计算，保存名称方便设置这些上层layer的输出blob */
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;                      // 否则，设置输入blob为不需要反传
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {                         // 若当前这个输入blob不需要反传
        // 将其名称保存在blobs_skip_backp中，表示与该输入blob对应的上层layer的输出blob可以跳过反传
        const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  } // layer第二个for循环结束
  // Handle force_backward if needed.
  // 如果net参数中设置了需要强制反传，则根据layer的是否设置允许强制反传再设置一遍
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true; // 默认设为需要
      for (int bottom_id = 0; bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        // 输入blob设置了需要反传，或者该输入blob设置了允许强制反传
        // 详见layer.hpp，允许强制反传则优先遵从layer的强制设置，不允许的话则只考虑自身的设置
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        // 找到该输入blob在blobs_中的索引，并在blob_need_backward_中设置对应的值
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      // layer还是要遵守net的设置，全部置为需要反传
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  // 最后，将所有当前还未被当作输入的输出blob（保存在available_blobs中）全部认为是网络的输出
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());            // 将这些输出blob在blobs_中的指针统一保存起来
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);                   // 保存在blobs_中的索引
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;                           // 关联blob名称和在blobs_中的索引，方便由名称得到位置
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;                       // 关联layer名称和在layers_的索引，方便由名称得到位置
  }
  ShareWeights(); // 设置共享参数的数据和梯度指针
  debug_info_ = param.debug_info(); // 是否计算和打印调试信息
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

// 根据网络设置的NetState类型状态和各层内部设置的NetStateRule，判断网络是否需要包含该层
// 过滤掉这些层之后得到新的网络参数param_filtered
template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());                    // 读取网络的状态
  param_filtered->CopyFrom(param);                      // 先从param中拷贝所有数据
  param_filtered->clear_layer();                        // 然后清空所有layer
  for (int i = 0; i < param.layer_size(); ++i) {        // 大循环，判断所有layer
    const LayerParameter& layer_param = param.layer(i); // 当前layer的参数（LayerParameter类型的消息）
    const string& layer_name = layer_param.name();      // layer的名称
    // 检查，include和exclude不能同时设置，不然可能会冲突，参见caffe.proto中的LayerParameter
    // include和exclude均为NetStateRule类型的消息，用于设置layer的状态
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0); // 先把未设置include的layer默认包含进来，以便检查它们的exclude
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) { // 检查它们的exclude
        layer_included = false; // 满足layer设置的任意一个exclude，则net运行时不能包含该层layer
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) { // 检查设置了include的layer
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;  // 满足layer设置的任意一个included，则net运行时需要包含该层layer
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param); // 将符合的layer的参数加入到网络参数中
    }
  }
}

// 检查state（Parameter中的参数）中的设置是否满足rule（LayerParameter中的参数）中的所有规则
// 网络的phase必须与layer的phase相同，网络的level必须在layer的min_level和max_level之间
// 网络的stage必须包含layer的所有stage字符串，不能包含not_stage中的任意一个字符串，这些都满足才说明网络的设置满足layer的要求
template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {                             // rule中设置的一堆stage字符串
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }            // 检查rule.stage(i)是否在state.stage(...)中
    }
    if (!has_stage) {                                                       // state中必须包含rule中的所有stage，否则返回false
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {                         // rule中设置了一堆not_stage字符串
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }        // 检查rule.not_stage(i)是否在state.stage(...)中
    }
    if (has_stage) {                                                        // state中不能包含rule中的任何一个not_stage，否则返回false
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// 根据第layer_id层的第top_id个输出blob，创建一个空的blob保存到网络参数param中的blobs_中，同址计算则不创建
// 并且无论是否为同址计算，将该blob数据的指针保存到top_vecs_中，将其在blobs_中的索引保存在top_id_vecs_中
// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));                       // 拷贝当前layer的参数
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";                         // 输出blob的名称，或者用"(automatic)"(匿名blob)替代
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {                       // 当前输出blob的名称与对应位置的输入blob的名称相同，为同址计算
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
    // (*blob_name_to_idx)[blob_name]为该名称对应的blob数据在blobs_中的索引
    // top_vecs_中会保存第layer_id层第top_id个输出blob的指针位置，top_id_vecs_保存索引
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // blob_name_to_idx只会存储输出blob的名称和位置，此处找到同名的输出blob，说明layer的输出blob的名称设置有问题
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    // 出现普通的输出blob，将其信息存储到Net中
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());           // 创建一个空的blob，返回其指针
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);                                     // blob_id为该输出blob在blobs_中的索引，将指针存入
    blob_names_.push_back(blob_name);                                   // 设置名称
    blob_need_backward_.push_back(false);                               // 初始设置不需要反向传播
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; } // 保存名称与位置的映射关系
    top_id_vecs_[layer_id].push_back(blob_id);                          // top_id_vecs_[layer_id][top_id]为其位置索引
    top_vecs_[layer_id].push_back(blob_pointer.get());                  // top_vecs_[layer_id][top_id]为其blob指针
  }
  // 该输出blob可被后续的layer当成输入，用于AppendBottom()中
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// 将第layer_id层的第bottom_id个输入blob加入到网络参数param中的对应layer的参数中
// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);            // 当前层的参数
  const string& blob_name = layer_param.bottom(bottom_id);              // 输入blob的名称
  // available_blobs存放着第0至layer_id-1层的layer中所有还未被当作输入的输出blob的名称，
  // 在这里面找不到该层的输入blob的名称，说明该输入blob在之前层中找不到对应的输出blob，报错
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];                   // 名称得到其在blobs_中的索引id
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  // 每个输入blob都会调用一次AppendBottom()，所以bottom_vecs_[layer_id][bottom_id]存放着当前输入blob的指针
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());              // 将其在blobs_中的数据指针保存在bottom_vecs_中
  bottom_id_vecs_[layer_id].push_back(blob_id);         // bottom_id_vecs_[layer_id][bottom_id]保存该输入blob在blobs_中的索引
  available_blobs->erase(blob_name);                    // blob_name对应的输出blob已被当作输入blob，要在available_blobs中删除
  bool need_backward = blob_need_backward_[blob_id];                    // blobs_[blob_id]是否需要反向传播
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);              // 读取设置中第bottom_id个的值
  }
  // bottom_need_backward_[layer_id][bottom_id]保存该输入blob是否需要反向传播
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;                                                       // 返回该输入blob在blobs_中的位置
}

// 将第layer_id层的第param_id个参数数据保存到网络中，并保存一些用于索引的值
// 对于非共享的参数，还会将数据指针保存在learnable_params_中
// learnable_param_ids_保存第第layer_id层的第param_id个参数在learnable_params_的索引，共享参数保存其来源参数的索引
template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param(); // layer的参数
  const int param_size = layer_param.param_size();                      // layer参数中param参数的设置的个数
  // 按照caffe.proto文件中ParamSpec消息的说明，当需要在layer之间共享参数时，可以param_name会是一个非空的名称
  string param_name =
  (param_size > param_id) ? layer_param.param(param_id).name() : "";    // 参数blob的名称或者默认的""
  if (param_name.size()) {
    param_display_names_.push_back(param_name);                         // 非空，保存其名称，方便其他layer查询
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());           // 使用param_id作为其名称
  }
  const int net_param_id = params_.size();                              // net_param_id为当前参数在net的params_中的索引
  params_.push_back(layers_[layer_id]->blobs()[param_id]);              // 将第layer_id层的第param_id个参数数据保存在params_中
  param_id_vecs_[layer_id].push_back(net_param_id); // param_id_vecs_[layer_id][param_id]为在params_中的索引net_param_id
  //param_layer_indices_[net_param_id]为参数在网络中的位置(layer_id, param_id)
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;                     // layer中blob参数的默认训练参数，lr_mult_=1和decay_mult_=1等
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
  &layer_param.param(param_id) : &default_param_spec;                   // 使用设置的或者默认配置
  if (!param_size || !param_name.size() || (param_name.size() &&
    param_names_index_.find(param_name) == param_names_index_.end())) {
  // layer中没有配置param参数，或者param参数中没有设置名称，或者设置了名称但是还不在param_names_index_中
  // （设置了名称但还不在param_names_index_中，说明该参数首次出现，是源参数，但是需要共享给其他的layer）
  // This layer "owns" this parameter blob -- it is either anonymous
  // (i.e., not given a param_name) or explicitly given a name that we
  // haven't already seen.
    param_owners_.push_back(-1);    // 参数来自自身的layer，为源参数，设置为-1
    if (param_name.size()) {        // param中设置了名称，说明需要共享给其他layer的某个参数
        param_names_index_[param_name] = net_param_id;          // 将参数名称param_name与在params_的位置net_param_id关联起来
    }
    const int learnable_param_id = learnable_params_.size();    // 当前参数在learnable_params_中的索引
    learnable_params_.push_back(params_[net_param_id].get());   // 保存参数指针
    learnable_param_ids_.push_back(learnable_param_id);         // 保存索引
    has_params_lr_.push_back(param_spec->has_lr_mult());        // 保存参数是否设置了对应的学习率系数
    has_params_decay_.push_back(param_spec->has_decay_mult());  // 保存参数是否设置了对应的衰减率系数
    params_lr_.push_back(param_spec->lr_mult());                // 保存参数的学习率系数
    params_weight_decay_.push_back(param_spec->decay_mult());   // 保存参数的衰减率系数
    } else {
        // 说明该参数来源于其他的layer参数，为共享参数
        // 共享参数配置的param中的name必须与源参数的param中的name相同
        // Named param blob with name we've seen before: share params
        const int owner_net_param_id = param_names_index_[param_name];  // 先找到源参数在net的params_的索引
        param_owners_.push_back(owner_net_param_id);            // 保存源参数的索引
        const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];               // 得到源参数在网络中的位置(layer_id, param_id)
        const int owner_layer_id = owner_index.first;           // 源参数所在的层
        const int owner_param_id = owner_index.second;          // 源参数所在的位置
        LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
            << "' owned by "
            << "layer '" << layer_names_[owner_layer_id] << "', param "
            << "index " << owner_param_id;
        Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();  // 当前参数的blob指针，第layer_id层第param_id个
        Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get(); // 源参数在net中的blob指针，第owner_layer_id层第owner_param_id个
        const int param_size = layer_param.param_size();
        if (param_size > param_id && (layer_param.param(param_id).share_mode() == ParamSpec_DimCheckMode_PERMISSIVE)) {
            // 检查参数共享的模式，PERMISSIVE模式下只要求两个blob的总数据个数相同
            // Permissive dimension checking -- only check counts are the same.
            CHECK_EQ(this_blob->count(), owner_blob->count())
                << "Cannot share param '" << param_name << "' owned by layer '"
                << layer_names_[owner_layer_id] << "' with layer '"
                << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
                << "shape is " << owner_blob->shape_string() << "; sharing layer "
                << "shape is " << this_blob->shape_string();    // 检查数据的总个数是否相等
        } else {
            // Strict模式下要求两个参数blob的数据的各个维度值都相等
            // Strict dimension checking -- all dims must be the same.
            CHECK(this_blob->shape() == owner_blob->shape())
                << "Cannot share param '" << param_name << "' owned by layer '"
                << layer_names_[owner_layer_id] << "' with layer '"
                << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
                << "shape is " << owner_blob->shape_string() << "; sharing layer "
                << "expects shape " << this_blob->shape_string();// 检查形状是否相同
        }
        // owner_net_param_id虽然是源参数在params_的索引，但是每次调用AppendParam()时，params_与learnable_param_ids_
        // 都会存入一个值，他们的大小一致，是逐个对应的（注意learnable_params_与params_不是），所以他们之间的索引可以通用
        const int learnable_param_id = learnable_param_ids_[owner_net_param_id];  // 得到参数在learnable_params_中的索引
        learnable_param_ids_.push_back(learnable_param_id);      // 将源参数的索引当成当前的共享参数的索引，保存起来
        if (param_spec->has_lr_mult()) {                         // 当前的共享参数配置的param中有设置学习率系数
            if (has_params_lr_[learnable_param_id]) {            // 源参数也配置了学习率系数
                CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
                    << "Shared param '" << param_name << "' has mismatched lr_mult."; // 要求两者相等，不然参数共享时更新的步长不一致
            } else {
                has_params_lr_[learnable_param_id] = true;       // 源参数未设置，则将共享参数的设置保存到源参数的设置中
                params_lr_[learnable_param_id] = param_spec->lr_mult();
            }
        }
        if (param_spec->has_decay_mult()) {                      // 权重衰减系数同样处理
            if (has_params_decay_[learnable_param_id]) {
                CHECK_EQ(param_spec->decay_mult(),
                         params_weight_decay_[learnable_param_id])
                    << "Shared param '" << param_name << "' has mismatched decay_mult.";  // 要求相等
            } else {
                has_params_decay_[learnable_param_id] = true;
                params_weight_decay_[learnable_param_id] = param_spec->decay_mult();  // 配置之前未设置的源参数
            }
        }
    }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    for (int c = 0; c < before_backward_.size(); ++c) {
      before_backward_[c]->run(i);
    }
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
    for (int c = 0; c < after_backward_.size(); ++c) {
      after_backward_[c]->run(i);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string& trained_filename) {
  if (H5Fis_hdf5(trained_filename.c_str())) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string& trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string& trained_filename) {
#ifdef USE_HDF5
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
#else
  LOG(FATAL) << "CopyTrainedLayersFromHDF5 requires hdf5;"
             << " compile with USE_HDF5.";
#endif  // USE_HDF5
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
// This code is taken from https://github.com/sh1r0/caffe-android-lib
#ifdef USE_HDF5
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
// This code is taken from https://github.com/sh1r0/caffe-android-lib
#else
  LOG(FATAL) << "ToHDF5 requires hdf5; compile with USE_HDF5.";
#endif  // USE_HDF5
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
