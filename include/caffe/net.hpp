#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string& trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string& trained_filename);
  void CopyTrainedLayersFromHDF5(const string& trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void run(int layer) = 0;

    template <typename T>
    friend class Net;
  };
  const vector<Callback*>& before_forward() const { return before_forward_; }
  void add_before_forward(Callback* value) {
    before_forward_.push_back(value);
  }
  const vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) {
    after_forward_.push_back(value);
  }
  const vector<Callback*>& before_backward() const { return before_backward_; }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const vector<Callback*>& after_backward() const { return after_backward_; }
  void add_after_backward(Callback* value) {
    after_backward_.push_back(value);
  }

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);

  /// @brief The network name
  string name_;                                 // 网络的名称
  /// @brief The phase: TRAIN or TEST
  Phase phase_;                                 // 网络的字段状态(TRAIN or TEST)
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;   // 网络的各个层,layers_[layer_id]
  vector<string> layer_names_;                  // 网络的各个层的名称
  map<string, int> layer_names_index_;          // <layer的名称layer_name, layer在layers_中的索引layer_id>
  vector<bool> layer_need_backward_;            // 网络的各个层是否需要反向传播
    
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;     // 存储网络的所有输出blob,blobs_[blob_id]
  vector<string> blob_names_;                   // blob_names_[blob_id]为blobs_[blob_id]存储的blob数据的名称
  map<string, int> blob_names_index_;           // <blob数据的名称blob_name, blob数据在blobs_中的索引blob_id>
  vector<bool> blob_need_backward_;             // blobs_中的各个数据是否需要反向传播
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;   // bottom_vecs_[i][j]为第i层第j个输入blob数据的指针,数据在blobs_中
  vector<vector<int> > bottom_id_vecs_;         // bottom_id_vecs_[i][j]为第i层第j个输入blob在blobs_中的索引blob_id
  vector<vector<bool> > bottom_need_backward_;  // bottom_need_backward_[i][j]为第i层第j个输入blob是否需要反传
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;      // top_vecs_[i][j]为第i层第j个输出blob数据的指针,数据在blobs_中
  vector<vector<int> > top_id_vecs_;            // top_id_vecs_[i][j]为第i层第j个输出blob在blobs_中的索引blob_id
  /// Vector of weight in the loss (or objective) function of each net blob, indexed by blob_id.
  vector<Dtype> blob_loss_weights_;             // blobs_[blob_id]对应的loss权重
    
  vector<vector<int> > param_id_vecs_;          // param_id_vecs_[i][j]为第i层第j个参数blob在params_中的位置
  // param_owners_[i]为参数params_[i]的拥有者的id,源参数(参数来源于所在的layer)的值为-1,而共享参数的值为源参数在params_的索引
  vector<int> param_owners_;
  // param_display_names_[i]为参数params_[i]显示的名称,需共享的参数会设置非一个非空的值,未设置则用参数的id替代
  //由于AppendParam()中参数名称未设置时会用param_id替代,所以该变量中可能会存在重复的名称.因此该变量不可用于寻找或匹配与共享参数同名的源参数的位置
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_; // param_layer_indices_[i]为参数params_[i]在网络中的位置(layer_id, param_id)
  // <参数名称, 参数在params_的索引>,非空名称的参数只有在首次出现时才会保存在该变量中,后续出现都作为共享参数,所以该变量可用于寻找和匹配共享参数的源参数的位置
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;          // net_input_blob_indices_[i]为网络的数据输入blob在blobs_中的索引
  vector<int> net_output_blob_indices_;         // net_output_blob_indices_[i]为网络的数据输出blob在blobs_中的索引
  vector<Blob<Dtype>*> net_input_blobs_;        // 网络的数据输入blob的指针
  vector<Blob<Dtype>*> net_output_blobs_;       // 网络的数据输出blob的指针(所有未被当作其他层输入的输出blob)
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;    // 网络中的所有参数blob(包括源参数和共享参数)
  vector<Blob<Dtype>*> learnable_params_;       // 网络中的所有学习参数,只会保存源参数
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;             // learnable_param_ids_[i]为参数params_[i]在learnable_params_中的索引
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;                     // params_lr_[i]为参数learnable_params_[i]的学习率系数
  vector<bool> has_params_lr_;                  // has_params_lr_[i]为参数learnable_params_[i]是否设置了学习率系数
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;           // params_weight_decay_[i]为参数learnable_params_[i]的权重衰减系数
  vector<bool> has_params_decay_;               // has_params_decay_[i]为参数learnable_params_[i]是否设置了权重衰减系数
  /// The bytes of memory used by this net
  size_t memory_used_;                          // 网络中所有输出blob的大小之和
  /// Whether to compute and display debug info for the net.
  bool debug_info_;   // 网络是否允许计算并打印调试信息,允许的话则会打印参与前向计算的layer的输出blob和参数blob的数据的绝对值之和的均值
  // Callbacks
  vector<Callback*> before_forward_;            // 网络执行前向计算之前的回调函数
  vector<Callback*> after_forward_;             // 网络执行前向计算之后的回调函数
  vector<Callback*> before_backward_;           // 网络执行反向传播之前的回调函数
  vector<Callback*> after_backward_;            // 网络执行反向传播之后的回调函数

DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
