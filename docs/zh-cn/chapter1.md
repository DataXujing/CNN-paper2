## 残差网络
------

ResNet(Residual Neural Network)由微软研究院的Kaiming He等4名华人提出，通过Residual Unit(见图1)成功训练152层深的神经网络，在ILSVRC 2015比赛中获得了冠军，取得3.57%的top-5错误率，同时参数比VGGNet低，效果非常突出。

<div align=center>
<img src="zh-cn/img/ResNet/0.png" /> 
</div>
**图1：ResNet的残差学习模块**

理论上来说网络的深度越深，那么就可以很好的拟合任意的数据分布，实际上通过实验得知，当网络深度增加到一定的程度的时候，学习过程反而变的很差，训练的错误率不降反升。并且收敛的速度也会变的很慢，如下图所示：

<div align=center>
<img src="zh-cn/img/ResNet/1.png" />
</div>
这就是所谓的Degradation的问题，即准确率先上升再达到饱和，再持续增加深度则会导致准确率的下降，这并不是过拟合的问题，因为不光在测试集上误差增加，训练集误差也会增加。假设有一个比较浅的网络达到饱和的准确率，那么后再加上几个y=x的映射层，起码误差不会增加，映射将前一层的输出传给后面的思想就是ResNet的灵感来源。

残差网络的出现意在解决，网络训练的深度问题和训练的复杂度，其两个主要的特点是：

+ 收敛快，相比普通网络收敛快，容易训练；

+ 可以通过简单的堆叠这种结构以获得更高的正确率；

### 1.什么是残差块

------

<div align=center>
<img src="zh-cn/img/ResNet/2.png" />
</div>

+ 直观认为学习一个非线性的结构要比学习一个恒等映射要难的多

+ 残差块之所以叫残差，相当于应层中学习的是残差，而非非线性的函数函数映射

+ 这种办法使得网络额复杂度并没有增加多少，但是他可以变的更深，残差的收敛速度更快，并且使得网络的层数更深


### 2.残差网络的结构

------

<div align=center>
<img src="zh-cn/img/ResNet/3.png" />
</div>

残差网络在各种图像识别和目标检测的任务重都获得了不错的成绩，详细的可以参考：Deep Residual Learning for Image Recognition


### 3. ResNet V2

分析了残差块背后的计算传播方式，表明了当shortcut以及附加激活项都使用恒等映射(identity mappings)时，前向和后向的信号能够直接的从一个block 传递到其他任意一个block。一系列的“消融”实验(ablation experiments)也验证了这些恒等映射的重要性。作者提出了一个新的残差单元，它使得训练变得更简单，同时也提高了网络的泛化能力。

#### Introduction

深度残差网络(ResNets)由很多个“残差单元”组成。每一个单元(Fig.1 (a))可以表示为： 

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_1.png" />
</div>

其中 $x_l$ 和 $x_{l+1}$ 是第l个单元的输入和输出， F 表示一个残差函数。在He2016中， $h(x_l)=x_l$  代表一个恒等映射，f 代表 ReLU, 这像极了Gradient Boosting.

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_2.png" />
</div>
Fig.1 (a) 原始残差单元；(b) 本文提出的残差单元；右：1001层ResNets 在CIFAR-10上的训练曲线。实线对应测试误差(右侧的y轴)，虚线对应训练损失(左侧的y轴)。本文提出的单元使得ResNet-1001的训练更简单。

实验表明： 还是恒等映射好，网络的深度很重要 （为什么恒等映射好，理论上没有说明）

####  Analysis of Deep Residual Networks

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_3.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_4.png" />
</div>

在mini-batch中，梯度(Eq.5)不可能出现梯度消失，因为在一个mini-batch上Eq.5后边的式子基本不会总为-1，也就是说即使很小的权重，也不会存在梯度消失，我想这也是ResNet之所以训练这多深的原因（实际上我们在训练深层网络结构时，是不害怕梯度爆炸的，但是非常害怕梯度消失）

#### On the Importance of Identity Skip Connections

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_5.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_6.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_7.png" />
</div>

##### Experiments on Skip Connections 

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_8.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_9.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_10.png" />
</div>

##### Discussions 

如Fig.2中灰色箭头所示，捷径连接是信息传递最直接的路径。 捷径连接中的操作 (缩放、门控、1×1 的卷积以及 dropout) 会阻碍信息的传递，以致于对优化造成困难。

值得注意的是1×1的卷积捷径连接引入了更多的参数，本应该比恒等捷径连接具有更加强大的表达能力。事实上，shortcut-only gating 和1×1的卷积涵盖了恒等捷径连接的解空间(即，他们能够以恒等捷径连接的形式进行优化)。然而，它们的训练误差比恒等捷径连接的训练误差要高得多，这表明了这些模型退化问题的原因是优化问题，而不是表达能力的问题。

#### On the Usage of Activation Functions

以上的实验内容验证了Eq.5和Eq.8中的分析，两个公式都是基于连接后的激活函数 f 为恒等连接的假设。但是在上述实验中f是以He2016中的ReLU设计的，因此，Eq.5和8只是以上实验的估计。接下来我们研究f的影响。

我们希望通过重新安排激活函数(ReLU和/或BN)来使得 f 为一个恒等映射。He2016中的原始残差连接的形状如Fig.4(a) — BN在每一个权重层之后使用，之后再接一个ReLU，在最后的元素相加之后还有最后一个ReLU(即f= ReLU)。 Fig.4(b-e)展示了我们研究的其他形式，解释如下。

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_11.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_12.png" />
</div>

##### Experiments on Activation 

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_13.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_14.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_15.png" />
</div>

<div align=center>
<img src="zh-cn/img/resnetv2/ch1_16.png" />
</div>


<div align=center>
<img src="zh-cn/img/resnetv2/ch1_17.png" />
</div>

https://blog.csdn.net/wspba/article/details/60750007



### 4.Tensorflow实现ResNet

------

以上主要是ResNet V1版本的理论知识，在ResNet作者的第二篇论文：Identity Mappings in Deep Residual Networks中， ResNet V2被提出，Residual V2和Residual V1的主要区别在于：

+ 作者通过研究ResNet残差学习单元的传播公式，发现前馈和反馈信号可以直接传输，因此skip connection的非线性激活函数（如ReLU）替代为Identity Mapping (y = x)。

+ ResNet V2在每一层都使用了Batch Normalization。这样处理之后，新的残差学习单元将比以前更容易训练且泛化性更强。

下面我们就用Tensorflow实现一个ResNet V2网络。我们依然使用方便的contrib.slim库来辅助创建ResNet。

```python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
# ResNet V2
# 载入模块、TensorFlow
import collections
import tensorflow as tf
 
slim = tf.contrib.slim
# 注意tf.contrib类,很多和卷积相关的模块都在slim中，例如AlexNet,VGG
 
# 定义Block,结构体,自动以的一种数据结构
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block'

# 说明： 见bottleneck这个函数
# scope:"block1","block2".....
# unit_fn: bottleneck 残差学习单元
# args: 参数 见resnet_v2_50

 
# 定义降采样subsample方法
def subsample(inputs, factor, scope=None):
    '''
    factor: 1不做修改直接返回input,其他时使用最大池化1X1的尺寸，步长是factor
    scope: 变量的命名空间
    '''
 
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
 
# 定义conv2d_same函数创建卷积层
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    '''
    如果factor=1直接“SAME"填充
    如果factor!=1,input做填充
    '''
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1  #pad的总数
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        # input,张量维度：在8个维度上的填充，填充模式，填充值默认为0
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)
 
 
@slim.add_arg_scope
# 装饰目标函数，arg_scope为目标函数设置默认参数
# 定义堆叠Blocks函数
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
 
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            #name_or_scope,default_name=None,values=None
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, # 残差学习的生成函数 bittleneck函数
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
            # 将输出net添加到collection
 
    return net
 
# 创建ResNet通用arg_scope，定义函数默认参数值
# arg_sope用来定义默写函数参数的默认值
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
 
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
 
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
 
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
 
 
@slim.add_arg_scope
# 定义核心bottleneck残差学习单元
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
 
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 获取输入的最后一个维度，即channel数，min_rank限定最少为4个维度
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # batch normalization,并进行Relu预激活
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')  # 降采样 使得输入通道数epth_in与depth一致
        else: #用1X1的卷积改变通道数
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
 
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')
 
        output = shortcut + residual  # identity的
        
        # output加入到集合并返回output
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)
 
# 定义生成ResNet V2的主函数
def resnet_v2(inputs, 
              blocks,  # 定义好的类的列表
              num_classes=None, #最后输出的类数
              global_pool=True, # 是否加入最后一层的全局平均池化
              include_root_block=True,  # 是否加入最前面通常使用的7X7的卷积池化
              reuse=None,  #是否重用
              scope=None):  #整个网络的名称（命名空间）
 
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
 
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
 
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True) # avg_pool
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            # 将collection转化为dict
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            # 返回net和end_points
            return net, end_points
 

#----------------------------------------------------------
# 设计层数为50的ResNet V2
def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        # Depth:输出通道数,depth_bottleneck,stride
        # (256,64,1): 256第三层输出的通道数，64:前两层输出的通道数，1:中间层的步长
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
# 设计101层的ResNet V2
def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
# 设计152层的ResNet V2
def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
# 设计200层的ResNet V2
def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)
 
 
from datetime import datetime
import math
import time
 
# 评测函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))
 
 
batch_size = 32
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(inputs, 1000) # 152层评测
 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "Forward")


```

### 5.Reference

[1].Andrew Ng deeplearning.ai中的教程

[2].Deep Residual Learning for Image Recognition

[3].Identity Mappings in Deep Residual Networks

[4]. Tensorflow实现卷积神经网络






