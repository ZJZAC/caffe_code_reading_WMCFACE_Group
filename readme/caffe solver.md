# caffe solver

## 概要

solver算是caffe的核心的核心，它协调着整个模型的运作。caffe程序运行必带的一个参数就是solver配置文件。运行代码一般为

```
# caffe train --solver=*_slover.prototxt
```

在Deep Learning中，loss  function往往是非凸的，没有解析解，因此需要通过优化方法来求解。**solver的主要作用就是交替调用前向（forward)算法和后向（backward)算法来更新参数，从而最小化loss，实际上就是一种迭代的优化算法。**

## Solver的流程：

**主流程**

1. 设计好需要优化的对象，以及用于学习的训练网络和用于评估的测试网络。
2. 通过forward和backward迭代的进行优化来更新参数
3. 定期的评价测试网络
4. 在优化过程中显示模型和solver的状态

**每一步迭代的过程（功能体现在solvers文件夹中）**--------\caffe-master\caffe-master\src\caffe\solvers

1. 通过forward计算网络的输出和loss
2. 通过backward计算网络的梯度
3. 根据solver方法，利用梯度来对参数进行更新
4. 根据learning rate，history和method来更新solver的状态

## solver配置参数（42个）

#### 模型网络定义prototxt相关

```
net: "train_test.prototxt"
net_param {
  name: "LeNet"
  layers {
    name: "mnist"
    type: DATA
    top: "data"
    top: "label"
    data_param {
      source: "examples/mnist/mnist_train_lmdb"
      backend: LMDB
      batch_size: 64
    }
    transform_param {
      scale: 0.00390625
    }
    include: { phase: TRAIN }
  }

 ...

  layers {
    name: "loss"
    type: SOFTMAX_LOSS
    bottom: "ip2"
    bottom: "label"
    top: "loss"
  }
}
train_net: "train.prototxt"
test_net: "test.prototxt"
train_net_param： {...}
test_net_param： {...}
```

net：训练网络用的prototxt文件，该文件可能包含不止一个的测试网络，通常不与train_net和test_net同时定义；

net_param：内联的训练网络prototxt定义，可能定义有不止一个的测试网络，通常忽略；

train_net_param：内联的训练网络prototxt定义，通常忽略；

test_net_param：内联的测试网络prototxt定义，通常忽略；

train_net：训练网络用的prototxt文件，通常不与net同时定义；

test_net：测试网络用的prototxt文件，通常不与net同时定义；

####模型运行状态

```
train_state: { 
phase: TRAIN
}
test_state: { 
phase: TEST
stage: 'test-on-test' 
}
```

train_state：训练状态定义，默认为TRAIN，否则按照模型网络prototxt定义的来运行；

test_state：测试状态定义，默认为TEST并在测试集上进行测试，否则按照模型网络prototxt定义的来运行；

####测试网络参数配置

```
test_iter: 50             
test_interval: 200
test_compute_loss: false    
test_initialization: true
```

test_iter：测试网络前向推理的迭代次数，注意每测试迭代一次是一个测试网络定义的batch size大小，test_iter与test_batch_size的乘积应为整个测试集的大小；
test_interval：训练时每隔多少次迭代则进行一次测试，默认为0即每次训练完后都会进行一次测试，应该要配置该参数，否则训练速度超级慢；
test_compute_loss：测试时是否计算损失值，默认为假，通常用于debug分析用；
test_initialization：在第一次训练迭代之前先运行一次测试，用于确保内存够用和打印初始的loss值，默认为真；

####学习率相关的参数配置

```
base_lr: 0.1
lr_policy: "multistep"
max_iter: 100000
stepvalue: 10000
stepsize: 5000
gamma: 0.1
power: 0.75
```

base_lr ：初始的学习率；
lr_policy：学习率调整策略；
maxiter：训练迭代的最大次数；
stepsize：lr_policy为“step”时学习率多少次训练迭代会进行调整；
stepvalue：lr_policy为“multistep”时学习率多少次训练迭代会进行调整，该参数可设置多个以用于多次学习率调整；
gamma：用于计算学习率的参数，lr_policy为step、exp、inv、sigmoid时会使用到；
power：用于计算学习率的参数，lr_policy为inv、poly时会使用到；

lr_policy学习率调整策略：

- - fixed：保持base_lr不变.
- - step：如果设置为step，则还需要设置一个stepsize，返回base_lr * gamma ^ (floor(iter / stepsize))，其中iter表示当前的迭代次数
- - exp：返回base_lr * gamma ^ iter， iter为当前迭代次数
- - inv：如果设置为inv，还需要设置一个power，返回base_lr * (1 + gamma * iter) ^ (- power)
- - multistep：如果设置为multistep，则还需要设置一个stepvalue。这个参数和step很相似，step是均匀等间隔变化，而multstep则是根据stepvalue值变化
- - poly：学习率进行多项式误差，返回 base_lr * (1 - iter/max_iter) ^ (power)
- - sigmoid：学习率进行sigmod衰减，返回 base_lr * ( 1/(1 + exp(-gamma * (iter - stepsize))))

####模型优化相关参数

```
type: "Adam"
solver_type: "Adam"(已弃用)
momentum: 0.9
momentum2: 0.999
rms_decay: 0.98
delta: 1e-8
weight_decay: 0.0005
regularization_type: "L2"
clip_gradients: 0.9
```

type：优化器类型；
solver_type：已弃用的优化器类型；
momentum：用到动量来进行权重优化的优化器动量；
momentum2：“Adam”优化器的动量参数；
rms_decay：“RMSProp”优化器的衰减参数，其计算方式为MeanSquare(t) = rms_decay*MeanSquare(t-1) + (1-rms_decay)*SquareGradient(t)
delta：RMSProp、AdaGrad、AdaDelta及Adam等优化器计算值为0时的最小限定值，用于防止分母为0等溢出错误；
weight_decay：权重衰减参数，用于防止模型过拟合；
regularization_type：正则化方式，默认为L2正则化，可选的有L0、L1及L2，用于防止模型过拟合；
clip_gradients：限定梯度的最大值，用于防止梯度过大导致梯度爆炸；

可选的caffe优化器类型：

到目前的为止，caffe提供了六种优化算法来求解最优参数，在solver配置文件中，通过设置type类型来选择：

- Stochastic Gradient Descent (type: "SGD"或“0”)
- Nesterov’s Accelerated Gradient (type: "Nesterov"或“1”)
- Adaptive Gradient (type: "AdaGrad"或“2”)
- RMSprop (type: "RMSProp"或“3”)
- AdaDelta (type: "AdaDelta"或“4”)
- Adam (type: "Adam"或“5”)

####模型保存快照相关参数

```
snapshot: 1000
snapshot_prefix: "examples/finetune_pascal_detection/pascal_det_finetune"
snapshot_diff: false
snapshot_format: BINARYPROTO
snapshot_after_train: true
```

snapshot：保存模型的间隔，即每隔多少次训练迭代保存一次模型快照，默认为0；
snapshot_prefix：模型保存的路径及路径名，但无后缀扩展类型，如果不设定，则使用无扩展的prototxt路径和文件名来作为模型保存文件的路径和文件名；
snapshot_diff：是否保存推理结果中的差异，默认不保存，如果保存可帮助调试但会增大保存文件的大小；
snapshot_format：模型保存的类型，有“HDF5”和“BINARYPROTO”两种，默认为后者BINARYPROTO；
snapshot_after_train：默认为真，即训练后按照模型保存设定的参数来进行快照，否则直到训练结束都不会保存模型；

####其他的solver参数

```
display: 1000
average_loss: 50
iter_size: 1
solver_mode: GPU
device_id: 0
random_seed: 600
debug_info: false
layer_wise_reduce: true
weights: "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
```

display：训练迭代多少次后显示相关信息到终端，如果置0则不会有任何有效信息打印；
average_loss：显示上一次迭代平均损失值的间隔，默认为1，通常不设定；
iter_size：用于多少个batch_size后再更新梯度，通常在GPU内存不足时用于扩展batch_size，真时的batch_size为iter_size*batch_size大小；
solver_mode：训练时使用CPU还是GPU，默认为GPU；
device_id：使用GPU时的设备id号，默认为0；
random_seed：随机种子起始数字，默认为-1参考系统时钟；
debug_info：默认为假，如果置真，则会打印模型网络学习过程中的状态信息，可用于分析调试；
layer_wise_reduce：数据并行训练的重叠计算和通信，默认为真；
weights：预训练模型路径，可用于加载预训练模型，如果命令行训练时也有定义“--weights”则其优先级更高将会覆盖掉solver文件中该参数的配置，如果命令行训练时有定义“--snapshot”时则其具有最高优先级将会覆盖掉“--weights”，如果存在多个权重模型用于加载，可使用逗号进行分离表示；