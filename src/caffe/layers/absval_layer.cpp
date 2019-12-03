#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {//输入和输出向量 保存着Blob指针的序列
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();//看向量中有几个元素
  Dtype* top_data = top[0]->mutable_cpu_data();//获取输出top的地址 便于下一步赋值
  caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();//前面一层关于本层top输出的偏导
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();    //计算偏导数计算关于本层bottom的偏导
    caffe_cpu_sign(count, bottom_data, bottom_diff);//将bottom里面的值的正负号赋给bottom_diff
    caffe_mul(count, bottom_diff, top_diff, bottom_diff);// 梯度相乘回传  
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsValLayer);
#endif

INSTANTIATE_CLASS(AbsValLayer);
REGISTER_LAYER_CLASS(AbsVal);

}  // namespace caffe
