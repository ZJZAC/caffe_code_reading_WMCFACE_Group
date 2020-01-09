# Net前半部分预备概念
- - - -
::初始化部分的一些函数以及作用::
**1.Init()**
初始化函数，用于创建blobs和layers，用于调用layers的setup函数来初始化layers。
**2.FilterNet()**
将protobuf描述的网络结构，根据网络状态等要求，转换成网络在某种状态下运行的结构，给定当前的phase_level_stage，移除指定的层。
**3.StateMeetsRule**
用来判断NetState是否符合NetStateRule的规则，符合的条件如下：
——NetState的phase与NetStateRule的phase一致。
——NetState的level在NetStateRule的[min_level, max_level]区间里。
——NetState的stage包含NetStateRule所列出的所有stage并且不包含任何一个not_stage。
**非常适合于搭建级联网络，或者多网络协同工作等情况。通过设置level，对应不同级联阶段，到了哪个阶段包含哪些层就一目了然了，不用来回删加，也不用写多个文件。stage类似，又增加了一维灵活性。**
**4.AppendTop()**
在网络中附加新的top的blob，blob分配内存空间，将其指针压入到top_vecs_中。
**5.AppendBottom()**
在网络中附加新的bottom的blob，由于当前层的输入blob是前一层的输出blob，所以此函数并没没有真正的创建blob，只是在将前一层的指针压入到了bottom_vecs_中。
**6.AppendParam()**
在网络中附加新的参数blob。
修改和参数有关的变量，实际的层参数的blob在上面提到的setup()函数中已经创建。如：将层参数blob的指针压入到params_。
::关于NetState和NetStateRule::
NetState描述网络的State，在caffe.proto里的定义如下:
```
message NetState {
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}
```
NetStateRule描述的是一种规则，在层的定义里设置，用来决定Layer是否被加进网络，在caffe.proto里的定义如下:
```
message NetStateRule {
  optional Phase phase = 1;
  optional int32 min_level = 2;
  optional int32 max_level = 3;
  repeated string stage = 4;
  repeated string not_stage = 5;
}
```
网络在初始化的时候会调用函数net.cpp里的FilterNet函数，根据网络的NetState以及层的NetStateRule搭建符合规则的网络。
使用NetStateRule的好处就是可以灵活的搭建网络，可以只写一个网络定义文件，用不同的NetState产生所需要的网络，比如常用的那个train和test的网络就可以写在一起。
**例如，如下定义的网络经过初始化以后’innerprod’层就被踢出去了：**
```
state: { level: 2 } 
name: 'example' 
layer { 
  name: 'data' 
  type: 'Data' 
  top: 'data' 
  top: 'label' 
} 
layer { 
  name: 'innerprod' 
  type: 'InnerProduct' 
  bottom: 'data' 
  top: 'innerprod' 
  include: { min_level: 3 } 
} 
layer { 
  name: 'loss' 
  type: 'SoftmaxWithLoss' 
  bottom: 'innerprod' 
  bottom: 'label' 
}
```
::关于include和exclude::
NetStateRule则需要在层的定义（LayerParameter）中设置，LayerParameter提供include和exclude两种规则，include的优先级高于exclude，有include的时候只看include，符合inlude才被加入；没有include的时候看exclude，符合exclude的层会被踢出网络，未设置规则的层则默认加进网络。