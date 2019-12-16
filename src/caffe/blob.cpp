#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
  //定义caffe命名空间，有利于避免命名冲突，因此创建caffe作用域。
  //各种Blob<Dtype>::Reshape都是为了进行blob类的初始化构造。
  //这里泛型编程，便于适应各种不同的数据类型的输入，Dtype为泛型类型，无具体类型
  //根据输入类型决定具体类型，增强函数复用性。


template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,//reshape函数，使用4个实数初始化Blob的shape
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);//在这里调用下文紧接的Reshape
}
// 完成blob形状shape_的记录，大小count_的计算，合适大小capacity_存储的申请
template <typename Dtype>
  /*Reshape函数的作用是改变一个blob的大小        
  *1.读入num_，channels_，height_，width_的大小         
  *2.计算count_：count_ = num_ * channels_ * height_ * width_;         
  *3.如果count_不为0，则重新为data_和diff_分配一块空间         
  *如果count为0，则都初始化为NULL        
  */
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);//是否小于规定的最大BLOB的维度(35维)
  count_ = 1;
  shape_.resize(shape.size());//首先将大小设置为vector<int> shape_; 即新的形状数据的大小
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {// shape_data_ 未初始化或者内存太小
    //重新为shared_ptr智能指针赋值并为其分配shape大小的内存空间
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  //强制类型转换，转换成指向int数据类型指针
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    // 检查形状数据是否合法
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    // 计算数据个数
    count_ *= shape[i];
    shape_[i] = shape[i];//在这里初始化shape的数据，最终shape的数据会写到shape_和shape_data中
    shape_data[i] = shape[i];//形状数据指针
  }
  if (count_ > capacity_) { // 判断是否大于存储的容量
    capacity_ = count_;//因为Blob中间存储的数据量，因此当数据量减少时，Blob的容量上限也会发生变化。
    // 重新分配内存
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());//在这个reshape函数中，先将Blob的shape参数转化为vector<int>类型，然后再初始化
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());//这个函数实现了用其他Blob的shape来初始化
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {//技巧，先初始化容量为0，然后用reshape来分配内存了
  Reshape(num, channels, height, width);//在这个函数中，先初始化了Blob的capacity_，然后用4个实数初始化了Blob的形状
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);//在这个函数中，先初始化了Blob的capacity_，然后用vector<int>& shape初始化了Blob的形状
}
//获取blob在gpu上存储数据形状的指针，const限定无法通过这一指针修改该数据
template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  /*在这里执行的gpu_data的操作不是本cpp中的gpu_data，而是SyncedMemory类的gpu_data()方法,具体含义在解析SyncedMemory类说明，因为返回值类型是const,在这里明白
是用只读方式得到gpu上面的形状指针*/
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();//SyncedMemory类的cpu_data()方法，目的是只读方式得到cpu上面的数据指针
}

template <typename Dtype>


void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);//SyncedMemory类的set_cpu_data()方法，目的是将访问cpu数据的指针指向data，意味着访问cpu数据可以从data开始
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {// 只读方式得到gpu上存储的数据指针
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {//只读方式得到cpu上存储的梯度指针
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {//只读方式得到gpu上存储的梯度指针
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {//得到cpu上存储的数据指针，一般在改变cpu上面的数据指针之前调用，还是使用了SyncedMemory类的mutable_cpu_data()
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {//得到gpu上存储的数据指针，一般在改变gpu上面的数据之前调用，还是使用了SyncedMemory类的mutable_gpu_data()
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {//得到cpu上存储的梯度指针，一般在改变cpu上面存储的梯度之前调用，还是使用了SyncedMemory类的mutable_cpu_data()，不过是diff_指针调用的
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {//得到gpu上存储的梯度指针，一般在改变gpu上面存储的梯度之前调用，还是使用了SyncedMemory类的mutable_gpu_data()，不过是diff_指针调用的
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {//与另外一个Blob共享数据
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {//与另外一个blob共享梯度
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
//update是为net中的参数blob准备的，只有float或double，不支持int和unsigned int
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
/*更新data_的数据，就是减去diff_的数据。计算data=-1 * diff + data       
*1.判断blob的位置        
*2.调用caffe_axpy：在math_functions.cpp可以找到该函数的实现，其实这函数也是封装了mkl的函数。这里调用是为了实现了两个向量的减法。         
*3.调用caffe_gpu_axpy：在math_functions.cpp可以找到该函数的实现，其实这函数也是封装了cublas的函数。这里调用是为了实现了两个向量的减法。        
*/

void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    //  axpby即alpha * x plus beta *y 这个含义,blas的函数命名真是见名知意    
    // template <> void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }    
    // caffe_axpy计算的是Y=alpha * X + Y ，其中alpha=-1了这里    
    // 存储的时候用到了mutable_cpu_data，防止其他线程访问
 
    // Y=alpha * X + Y ，其中alpha=-1了这里
    caffe_axpy<Dtype>(count_, Dtype(-1),//caffe_axpy函数存在math_function.cpp中，核心功能就是向量相加
        static_cast<const Dtype*>(diff_->cpu_data()),//在这先获得梯度值
        static_cast<Dtype*>(data_->mutable_cpu_data()));//然后进行数据的更新
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}
//同样不支持uint 和int类型
template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}
// 计算data的L1范数
template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());//在这里应用caffe_cpu_asum函数进行数据的L1范数求取，caffe_cpu_asum函数存在math_function.cpp中
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}
// 计算diff的L1范数
template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());//梯度同之前的数据一样
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}
// 计算sum of square of data(L2范数)
template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);//在这里应用caffe_cpu_dot函数进行数据的L2范数求取，caffe_cpu_dot函数存在math_function.cpp中
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}
// sum of square of diff
template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);//梯度同之前的数据一样
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);//在这里使用caffe_scal进行数据的放缩
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);//在这里使用caffe_scal进行梯度的放缩
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {//在这里判断两个Blob的形状是否相等，依次验证W,H,C,N.
  // 判断是否是旧的blob
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  // 如果不是旧的blob则直接判断
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
/***从source拷贝数据。copy_diff作为标志来区分是拷贝data还是拷贝diff        
*1.如果是GPU： 如果是拷贝diff：调用cudaMemcpy函数将source的diff拷贝过来，否则拷贝data         
*2.如果是CPU： 如果是拷贝diff：调用memcpy函数将source的diff拷贝过来 否则拷贝data        
*/
/*该函数从source blob复制数据，bool类型的reshape控制需不需要用source blob的shape来变更目前的blob，

而copy_diff则判断是复制偏差还是复制数据，若copy_diff为真，则复制偏差，为假则复制数据*/


void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);// 复制shape数据
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),// GPU复制diff
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),// CPU复制diff
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
 /**功能：从proto读数据进来，其实就是反序列化 （反序列化是指把字节序列恢复为对象的过程）        
*1.先把blob的大小改变一下         
*2.得到cpu中数据的地址         
*3.用proto中的data覆盖blob中的data         
*4.用proto中的diff覆盖blob中的diff        
*/
//resize成protobuf中对应形状，并从proto读取data和diff
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {//FromProto函数从Proto复制了shape,数据和偏差到Blob中
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      // 如果是旧的blob直接转换为新的blob中的shape数据
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);// 复制shape数据到当前blob
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();// 获取当前的blob在内存上的数据指针，该指针是互斥的
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  // copy diff
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();// 获取当前的diff在内存上的数据指针，该指针是互斥的
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}
// BlobProto和BlobShape是protobuf定义的，其中一些函数是自动生成的
// 包括mutable_shape、add_dim、clear_double_data、clear_double_diff、add_double_data、add_double_diff等
// 见src/caffe/proto/caffe.proto
//下面两个分别写double和float的blob到protobuf里
template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {//ToProto函数将信息从Blob中写入Proto中
  proto->clear_shape();
  // 存shape
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  // 存data
  const double* data_vec = cpu_data();//调用的是上文代码中的cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  // 存diff
  if (write_diff) {
    const double* diff_vec = cpu_diff();//调用的是上文代码中的cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe

