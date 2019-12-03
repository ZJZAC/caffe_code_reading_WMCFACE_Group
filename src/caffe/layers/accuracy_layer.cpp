#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);//outer_num_=100
  inner_num_ = bottom[0]->count(label_axis_ + 1);//inner_num_为每个图像所对应的类别数，所以=1
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //假设 batch size 100 label 10
  Dtype accuracy = 0;//准确率
  const Dtype* bottom_data = bottom[0]->cpu_data();//100*10
  const Dtype* bottom_label = bottom[1]->cpu_data();//100*1
  const int dim = bottom[0]->count() / outer_num_;//10
  const int num_labels = bottom[0]->shape(label_axis_);//全连接后的blob是2维的 所以label_axis = 1 
  //bottom[0] -> shape(1) = 10 type数量
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {//100 图像的个数
    for (int j = 0; j < inner_num_; ++j) {//每个图像对应的类
      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);//每个图像对应相对的类
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);

 //接下来把测试评分与类别ID挂勾
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      const Dtype prob_of_true_class = bottom_data[i * dim
                                                   + label_value * inner_num_
                                                   + j];

      int num_better_predictions = -1;  // true_class also counts as "better"
      // Top-k accuracy
      for (int k = 0; k < num_labels && num_better_predictions < top_k_; ++k) {
        num_better_predictions +=
          (bottom_data[i * dim + k * inner_num_ + j] >= prob_of_true_class);//把测试评分与类别ID挂勾
      }
      // check if there are less than top_k_ predictions
      if (num_better_predictions < top_k_) {
        ++accuracy;//就是最后概率向量最大的前k名中，只要出现了正确概率即为预测正确
        if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
      }
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = (count == 0) ? 0 : (accuracy / count);
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

#ifdef CPU_ONLY
STUB_GPU(AccuracyLayer);
#endif

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
