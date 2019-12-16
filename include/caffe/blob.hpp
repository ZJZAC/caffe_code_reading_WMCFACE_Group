//从"blob->layer->net->solver->综合->其他功能"这个顺序去阅读caffe源码不失为一种高效的学习caffe的手段。
//Blob是一个四维（维度从高到低分别是:num_（一个batch中的样本数量），channels_，height_，width_）的数组，用于存储数据，包括输入数据、输出数据、权值；同时它还隐含提供了在CPU和GPU之间同步数据的能力。
//直观的可以把它Blob看成一个有4维的结构体（主要包含数据和梯度），而实际上，它们只是一维的指针而已.Blob在也不一定全是4维的，后期的版本已经deprecated，而是直接用vector<int> shape_。
//Layer层则是神经网络中具体的各层结构，主要用于计算，在根据配置文件初始化网络结构后，前向计算结果，反向更新参数，而它的输入和输出都是Blob数据；
//Net的层就是多个Layer组合而成的有向无环图结构，也就是具体的网络.
//实际上BLOL包含了三类数据：（1）data，前向传播所用到的数据（2）diff，反向传播所用到的数据（3）shape，解释data和diff的shape数据。围绕这三类数据有对应的方法。
#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_ //这样才能保证头文件被多个其他文件引用(include)时，内部的数据不会被多次定义而造成错误。

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"//里面声明了Blobproto、Blobshape等遵循caffe.proto协议的数据结构
#include "caffe/syncedmem.hpp"//CPU/GPU共享内存类，用于数据同步，很多实际的动作都在这里面执行

const int kMaxBlobAxes = 32;//blob的最大维数目
//头文件还可以定义：在编译的时候就已知道其值的cosnt对象和inline 函数。在头文件中定义上述实体，是因为编译器需要它们的定义来产生代码。
namespace caffe {//命名空间为caffe，有利于避免命名冲突，因此创建caffe作用域。

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Blob {
 public:
  //blob的构造函数：1、默认参数。2、传入N、C、H、W构造，最终调用Reshape函数
  //Blob 类在初始化时并没有分配内存，也是通过调用 Reshape 来分配内存的。
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {} //默认构造函数.data_(), diff_()是用于存放数据的指针. 这种方式回头还要显式的调用reshape.

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  //explicit构造函数只能被显式调用。
  explicit Blob(const int num, const int channels, const int height,//通过四个数（数量，通道数，高度，宽度）初始化Blob
      const int width);
  explicit Blob(const vector<int>& shape);//通过shape矢量初始化Blob

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  //几个Reshape函数，对blob的维度进行更改
  void Reshape(const int num, const int channels, const int height,
      const int width);// 用户的reshape接口（常用）,通过四个数（数量，通道数，高度，宽度）改变Blob的形状
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);// 通过重载调用真正的reshape函数，推荐使用这个
  void Reshape(const BlobShape& shape);// 用户的reshape接口,通过BlobShape类型的shape改变Blob的形状
  void ReshapeLike(const Blob& other);//ReshapeLike的作用是为data_和diff_ 重新分配一块空间，大小和另一个blob的一样
  //获取Blob的形状字符串，用于打印log，比如： 10 3 256 512 （3932160），总元素个数
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " "; //打印每一个维度信息
    }
    stream << "(" << count_ << ")"; //打印总的元素的个数
    return stream.str();
  }
  inline const vector<int>& shape() const { return shape_; }// 成员函数，返回blob的形状信息（常用）
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {  // 返回blob特定维度的大小(常用),支持负索引，例如：shape的数据顺序是(N,C,H,W)，那么，shape(0)返回N，shape(-1)返回W，shape(-2)返回H
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int num_axes() const { return shape_.size(); }// 返回blob维度
  inline int count() const { return count_; } // 返回元素的个数，按照shape的结构返回N*C*H*W

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int count(int start_axis, int end_axis) const {// 返回特定维度区间的元素的个数，返回的乘积为shape从start_axis自身开始到end_axis之前为止的shape中各个元素的乘积，如count(0,2)返回N*C
    // 判断维度的索引是否在范围内
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  //一个自定义的count函数，返回的乘积为shape从atart_axis自身开始到shape中最后一个元素的乘积
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {// 检查输入的维度的合法性,同时对负数（index可能是负数）规范化。
    CHECK_GE(axis_index, -num_axes()) //GE即为great equation，意为大于等于，即判断axis_index是否大于等于-num_axes()，如果不满足则打印。
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes()) //LT即为lower to ，意为小于
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();//返回索引为负值的时候的正确值
    }
    return axis_index;
  }
  //下面是四个弃用的函数，作用是返回Blob的shape中的四个数值，使用shape(0)，shape(1),shape(2),shape(3)代替
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }// 返回样本的个数
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }// 返回通道的个数
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }// 返回样本维度一，对于图像而言是高度
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }// 返回样本维度二，对于图像而言是宽度
  //返回特定维度的大小，包含对输入维度的合法性检查，被上面函数调用
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)//检查blob的维度个数是不是小于4，也许以前的blob只有四维，但是现在的blob应该为了通用而采用了大于四维的方法
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);// 检查维度索引是不是小于4
    CHECK_GE(index, -4);// 检查维度索引是不是大于-4
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }
  /*num_, channel_, height_, width_主要用来做定位offset和reshape处理。        
  *对于输入(n, c, h, w)位置的数据位置为((n*channels_+c)*height_+h)*width_+w，        
  *可以依据位置取data_()或diff_()中的数据。        
  */

  // 计算当前的样本的偏移量，供后面序列化寻址使用，因为数据在内存是以一维数组形式的，所以需要计算偏移量来访问
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }
  //同时也可以通过一个索引矢量返回Blob中的偏移量
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  // 从source blob来拷贝到当前的blob中，默认是只拷贝数据，不拷贝梯度的（反之，只拷贝梯度，不拷贝数据），如果形状不一致需要使能reshape，不然无法拷贝
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);
  // 返回特定位置的元素值（前向传输时使用）
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];//这个是序列化的值
  }
  // 返回特定位置的梯度值（反向传输时使用）
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }
  // 根据index重载返回特定元素的值，作用与上面函数相同
  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }
  // 根据index重载返回特定梯度的值，作用与上面函数相同
  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }
  // 返回当前的训练样本的数据（指针）（常用）（cpu与gpu上存储的所有数据）
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }
  // 返回当前训练样本的梯度（指针）（常用）（cpu与gpu上存储的所有梯度）
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;// 只读获取cpu的data_的指针
  void set_cpu_data(Dtype* data);// 设置cpu的data_指针
  const int* gpu_shape() const;// 只读获取gpu上数据的形状信息
  const Dtype* gpu_data() const;// 只读获取gpu上的data_的指针
  void set_gpu_data(Dtype* data);
  const Dtype* cpu_diff() const;// 只读获取cpu的diff_的指针
  const Dtype* gpu_diff() const;// 只读获取gpu的diff_的指针
  Dtype* mutable_cpu_data();// 读写访问cpu data，一般在改变数据之前调用
  Dtype* mutable_gpu_data();// 读写访问gpu data，一般在改变数据之前调用
  Dtype* mutable_cpu_diff();// 读写访问cpu diff，一般在改变数据之前调用
  Dtype* mutable_gpu_diff();// 读写访问cpu diff，一般在改变数据之前调用
  void Update();// 数据更新，即减去当前计算出来的梯度
  void FromProto(const BlobProto& proto, bool reshape = true);// 将数据进行反序列化，从磁盘导入之前存储的blob//从protobuf序列化文件读取blob对象
  void ToProto(BlobProto* proto, bool write_diff = false) const;// 将数据进行序列化为protobuf文件，便于存储

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;// 计算data的L1范数
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;// 计算diff的L1范数
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;// 计算data的L2范数
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;// 计算diff的L2范数

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);// 按照一个标量进行伸缩data_
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);// 按照一个标量进行伸缩diff_

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);//从另一个Blob共享数据
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);// 从另一个Blob共享梯度

  bool ShapeEquals(const BlobProto& other);// 判断两个blob的形状是否一致

 protected:
    /*

    *主要数据有两个data和diff，用num、channels、height和width

    *这四个维度来确定数据的具体位置，做一些数据查询和Blobreshape的操作

    */
  //Blob同时保存了data_和diff_,其类型为SyncedMemory的指针，注意是指针。
  shared_ptr<SyncedMemory> data_;// 类的属性---数据
  shared_ptr<SyncedMemory> diff_;// 类的属性---梯度
  shared_ptr<SyncedMemory> shape_data_;//已经弃用，建议使用下一行的shape_替代
  vector<int> shape_;// 类的属性---形状信息(N,C,H,W)
  int count_;// 有效元素总的个数N*C*H*W
  int capacity_;// 存放bolb容器的容量信息，大于等于count_，因为Blob的形状会发生变化 

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
