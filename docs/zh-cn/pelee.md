## Pelee: A Real-Time Object Detection System on Mobile Devices

!> 论文地址：https://arxiv.org/abs/1804.06882

### 0.摘要

近年来, 可在移动设备上运行的卷积神经网络的需求不断增长, 促进了相关高效模型的设计和研究. 目前已经有大量的研究成果发表, 如 MobileNet, ShuffleNet(V1,V2), MobileNet (V1,V2,V3),SqueezeNet,EffNet等. 但是, 所有的这些模型都严重依赖于深度可分卷积(depthwise separable convolution), 但是这在大多数深度学习框架中都缺乏有效的实现. 在本文中, 我们提出了一个高效的结构, 命名为 PeleeNet, 它是通过 传统的卷积 构建的. 在 ImageNet ILSVRC 2012 数据集上, 本文提出的 PeleeNet 相对于 MobileNet 和 MobileNetV2 来说, 不仅具有更高的准确率, 同时还具有更快的速度(1.8倍). 同时, PeleeNet 的模型大小只有 MobileNet 的 66%. 我们还将 PeleeNet 和 SSD 方法结合起来, 提出了一个实时的目标检测系统, 并将其命名为 Pelee, 在 VOC2007 数据集上达到了 76.4% 的 mAP, **在 COCO 数据集上达到了 22.4 的 mAP, 在 iPone8 手机上达到了 23.6 FPS, 在 NVIDIA TX2 上达到了 125 FPS.**


### 1.Introduction

越来越多的研究开始关注在限制内存和计算成本的条件下, 如何构建可以高效运行的神经网络模型. 目前已经有很多创新的模型被提出, 如 MobileNet系列, ShuffleNet系列, NASNet-A,SquezeNet,EffNet, 但是所有的这些模型都严重依赖于深度可分卷积(depthwise separable convolution), 但是这种卷积缺乏有效的实现. 同时, 有一些研究会将高效的模型和快速的目标检测算法结合起来(Speed/Accuracy trade-offs). 因此, 本文主要的研究就是设计一个用于图片分类和目标检测任何的高效 CNN 模型, 主要的贡献点有以下几点:

#### (1) PeleeNet

我们提出了 DenseNet(参考本教程[DenseNet](./zh-cn/densenet)) 的一种变体, 并将其命名为 PeleeNet, 它主要是为了在移动设备上运行而设计的. PeleeNet 延续了 DenseNet 的 connectivity pattern 和 一些关键的设计原则. 同时, 它的有些设计也是为了解决有限的内存和算力问题而存在的. 实现表明, PeleeNet 在 ImageNet ILSVRC 2012 上的 top-1 准确率为 72.1% (比 MobileNet 高 1.6%). 同时需要注意, PeleeNet 的模型大小只有 MobileNet 的 66%. PeleeNet 主要有以下几点关键特征:

+ **Two-Way Dense Layer**: 受到 GoogLeNet 的启发, 我们使用了 2-way dense layer 来获取不同尺寸的感受野. 其中一路使用了`3×3`大小的卷积核. 另一路使用了两个 `3×3 `大小的卷积核来学习更大物体的视觉特征. 具体的结构如下图所示.


<div align=center>
<img src="zh-cn/img/pelee/p1.png" /> 
</div>

+ **Stem Block**: 受到 Inception-v4 和 DSOD 的启发, 我们在第一层 dense layer 之前设计了一个高效低成本(cost efficient)的 stem block. 该 stem block 的结构如下图所示. 它可以有效的提升特征表达能力, 同时不需要增减太大的计算成本, 要其他方法(增加第一个卷积层的通道数或者增加通道数的增长速度)要好.

<div align=center>
<img src="zh-cn/img/pelee/p2.png" /> 
</div>

+ **Dynamic Number of Channels in Bottleneck Layer**: 另一个亮点是 bottleneck 层的通道数是根据输入形状变化的, 而不是原始 DenseNet 中固定的 4 倍增长速度. 在 DenseNet 中, 我们观察到, 对于前一个 dense layers, bottleneck层通道的数量远远大于其输入通道的数量, 这意味着对于这些层, 瓶颈层增加了计算成本, 而不是降低了成本. 为了保持体系结构的一致性, 我们仍然将 bottleneck 层添加到所有的 dense layers 当中, 但是数量是根据输入数据的 shape 动态调整的, 以 确保通道的数量不超过输入通道的数量. 实验显示, 和原始的 DenseNet 结构相比, 这个方法可以节省 28.5% 的算力耗费, 但是只会轻微降低最终的结果. 如下图所示

<div align=center>
<img src="zh-cn/img/pelee/p3.png" /> 
</div>

+ **Transition Layer without Compression**: 我们的实验表明, DenseNet 提出的压缩因子(compression factor)对于特征表达有一定的负面影响. 我们在 transition layers 当中总是保持输出通道的数量和输入通道的数量相同.
+ **Composite Function**: 为了提高实际速度, 我们使用传统的 “后激活(conv+bn+relu)” 作为我们的复合函数, 而不是 DenseNet 中使用的预激活(这样会降低准确率). 对于后激活方法来说, 所有的 BN 层都可以与卷积层合并, 从而大大加快了推理的速度. 为了弥补这种变化对精度的负面影响, 我们使用了一种浅而宽的网络结果. 在最后一个 dense block 之后, 我们还添加了一个`1×1 `的卷积层, 以获得更强的表达能力.

!> 我们将会参考：https://github.com/nnUyi/PeleeNet, 在后文中对这些结构的实现进一步的在代码层面说明！


#### (2)我们优化了 SSD 的结构, 使其速度更快, 然后将它与我们的 PeleeNet 相结合.

我们将结合后的模型称为 Pelee, 该模型达到了 76.4% mAP on VOC 2007, 22.4 mAP on COCO. 为了平衡速度和准确度而提出的改善措施主要如下:

+ **Feature Map Selection**: 我们以一种不同于原始SSD的方式构建了目标检测网络, 并精心选择了一组5 个尺度的特征图谱`(19, 10, 5, 3, 1)`. 为了降低计算的复杂度, 我们没有使用`38×38`大小的 feature map.

+ **Residual Prediction Block**: 我们令特征沿着网络进行传递. 对于每个用于检测的特征图, 我们构建一个残差块, 具体的结构如下图所示.

<div align=center>
<img src="zh-cn/img/pelee/p4.png" /> 
</div>

+ **Small Convolutional Kernel for Prediction**: 残差预测块使得我们可以应用 `1×1 `的卷积核来预测类别得分和框的偏移量. 实验表明, 使用 `1×1 `核的模型精度与使用 `3×3 `核的模型精度基本相同. 然而, `1×1` 核的计算成本减少了 21.5%.

#### (3).我们在 NVIDIA TX2 嵌入式平台上和 iPhone8 上为不同的高效分类模型和不同的单阶段目标检测方法提供了一个 benchmark test.


### 2.PeleeNet: An Efficient Feature Extraction Network

#### 2.1Architecture

我们提出的 PeleeNet 的架构如下表所示. 整个网络由一个 stem block 和四个阶段的特征提取器构成(four stages of feature extractor). 除了最后一个阶段外, 每个阶段的最后一层是步长为2的平均池化层. 四阶段(不算 stem)结构是大型模型设计中常用的结构形式. ShuffleNet 使用了一个三阶段的结构, 并在每个阶段的开始将 feature map 的大小缩小. 虽然这可以有效的降低计算成本, 但我们认为, 早期阶段的特征对于视觉任务非常重要, 过早的减小特征图的大小会损害表征能力. 因此, 我们仍然保持四阶段结构. 前两个阶段的层数会专门控制在一个可介绍的范围内.

<div align=center>
<img src="zh-cn/img/pelee/p5.png" /> 
</div>


```python
#参考:https://github.com/nnUyi/PeleeNet
#layer.py

import tensorflow as tf
import tensorflow.contrib.slim as slim

class Layer:        
    # stem_block       
    def _stem_block(self, input_x, num_init_channel=32, is_training=True, reuse=False):
        block_name = 'stem_block'
        with tf.variable_scope(block_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               normalizer_fn=slim.batch_norm,
                                               activation_fn=tf.nn.relu) as s:
                conv0 = slim.conv2d(input_x, num_init_channel, 3, 1, scope='stem_block_conv0')
                
                conv1_l0 = slim.conv2d(conv0, int(num_init_channel/2), 1, 1, scope='stem_block_conv1_l0')
                conv1_l1 = slim.conv2d(conv1_l0, num_init_channel, 3, 1, scope='stem_block_conv1_l1')
                
                maxpool1_r0 = slim.max_pool2d(conv0, 2, 1, padding='SAME', scope='stem_block_maxpool1_r0')
                
                filter_concat = tf.concat([conv1_l1, maxpool1_r0], axis=-1)
                
                output = slim.conv2d(filter_concat, num_init_channel, 1, 1, scope='stem_block_output')
                
            return output

    def _dense_block(self, input_x, stage, num_block, k, bottleneck_width, is_training=True, reuse=False):
        with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           normalizer_fn=slim.batch_norm,
                                           activation_fn=tf.nn.relu) as s:
            output = input_x
            
            for index in range(num_block):
                dense_block_name = 'stage_{}_dense_block_{}'.format(stage, index)
                with tf.variable_scope(dense_block_name) as scope:
                    if reuse:
                        scope.reuse_variables()
                    
                    inter_channel = k*bottleneck_width
                    # left channel
                    conv_left_0 = slim.conv2d(output, inter_channel, 1, 1, scope='conv_left_0')
                    conv_left_1 = slim.conv2d(conv_left_0, k, 3, 1, scope='conv_left_1')
                    # right channel
                    conv_right_0 = slim.conv2d(output, inter_channel, 1, 1, scope='conv_right_0')
                    conv_right_1 = slim.conv2d(conv_right_0, k, 3, 1, scope='conv_right_1')
                    conv_right_2 = slim.conv2d(conv_right_1, k, 3, 1, scope='conv_right_2')
                    
                    output = tf.concat([output, conv_left_1, conv_right_2], axis=3)
            return output
                    
    def _transition_layer(self, input_x, stage, output_channel, is_avgpool=True, is_training=True, reuse=False):
        transition_layer_name = 'stage_{}_transition_layer'.format(stage)
        
        with tf.variable_scope(transition_layer_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                               normalizer_fn=slim.batch_norm,
                                               activation_fn=tf.nn.relu) as s:
                conv0 = slim.conv2d(input_x, output_channel, 1, 1, scope='transition_layer_conv0')
                if is_avgpool:
                    output = slim.avg_pool2d(conv0, 2, 2, scope='transition_layer_avgpool')
                else:
                    output = conv0
            return output
    
    def _classification_layer(self, input_x, num_class, keep_prob=0.5, is_training=True, reuse=False):
        classification_layer_name = 'classification_layer'
        with tf.variable_scope(classification_layer_name) as scope:
            if reuse:
                scope.reuse_variables()
            with slim.arg_scope([slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                        normalizer_fn=None,
                                                        activation_fn=None), \
                 slim.arg_scope([slim.dropout], keep_prob=keep_prob) as s:
                
                shape = input_x.get_shape().as_list()
                filter_size = [shape[1], shape[2]]
                global_avgpool = slim.avg_pool2d(input_x, filter_size, scope='global_avgpool')
                
                # dropout
                # dropout = slim.dropout(global_avgpool)
                flatten = tf.reshape(global_avgpool, [shape[0], -1])
                logits = slim.fully_connected(flatten, num_class, scope='fc')
                
                return logits
    
if __name__=='__main__':
    input_x = tf.Variable(tf.random_normal([64,224,224,32]))
    layer = Layer()
    stem_block_output = layer._stem_block(input_x, 32)
    dense_block_output = layer._dense_block(input_x, 0, 3, 16, 2)
    transition_layer_output = layer._transition_layer(dense_block_output, 0, is_avgpool=False)
    print(stem_block_output.get_shape().as_list())
    print(dense_block_output.get_shape().as_list())
    print(transition_layer_output.get_shape().as_list())


```

```python
#参考:https://github.com/nnUyi/PeleeNet
#pelee.py

import tensorflow as tf
import numpy as np
import time
import os

from tqdm import tqdm
from layers import Layer
from utils import get_data, gen_batch_data

class PeleeNet:
    model_name = 'PeleeNet'
    '''
        PeleeNet Class
    '''
    def __init__(self, config=None, sess=None):
        self.sess = sess
        self.config = config
        
        self.num_class = self.config.num_class
        self.input_height = self.config.input_height
        self.input_width = self.config.input_width
        self.input_channel = self.config.input_channel
        
        self.batchsize = self.config.batchsize
        
        self.layer = Layer()
        
    def peleenet(self, input_x, k=32, num_init_channel=64, block_config=[3,4,8,6], bottleneck_width=[2,2,4,4], is_training=True, reuse=False):
        with tf.variable_scope(self.model_name) as scope:
            if reuse:
                scope.reuse_variables()
                
            '''
            --------------------------------------------------------------------
                                    feature extraction
            --------------------------------------------------------------------
            '''
            # _stem_block(self, input_x, num_init_channel=32, is_training=True, reuse=False):    
            from_layer =  self.layer._stem_block(input_x,
                                                 num_init_channel=num_init_channel,
                                                 is_training=is_training,
                                                 reuse=reuse)
            
            # _dense_block(self, input_x, stage, num_block, k, bottleneck_width, is_training=True, reuse=False):
            # _transition_layer(self, input_x, stage, is_avgpool=True, is_training=True, reuse=False):
            stage = 0
            for num_block, bottleneck_coeff in zip(block_config, bottleneck_width):
                stage = stage + 1
                # dense_block
                from_layer = self.layer._dense_block(from_layer,
                                                     stage,
                                                     num_block,
                                                     k,
                                                     bottleneck_coeff,
                                                     is_training=is_training,
                                                     reuse=reuse)

                is_avgpool = True if stage < 4 else False
                output_channel = from_layer.get_shape().as_list()[-1]
                # transition_layer
                from_layer = self.layer._transition_layer(from_layer,
                                                          stage,
                                                          output_channel=output_channel,
                                                          is_avgpool=is_avgpool,
                                                          is_training=is_training,
                                                          reuse=reuse)

            '''
            --------------------------------------------------------------------
                                    classification
            --------------------------------------------------------------------
            '''
            # _classification_layer(self, input_x, num_class, keep_prob=0.5, is_training=True, reuse=False):
            logits = self.layer._classification_layer(from_layer, self.num_class, is_training=is_training, reuse=reuse)
            return logits

    def build_model(self):
        self.input_train = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channel], name='input_train')
        self.input_test = tf.placeholder(tf.float32, [self.batchsize, self.input_height, self.input_width, self.input_channel], name='input_test')
        self.one_hot_labels = tf.placeholder(tf.float32, [self.batchsize, self.num_class], name='one_hot_labels')

        # logits data and one_hot_labels
        self.logits_train = self.peleenet(self.input_train, is_training=True, reuse=False)
        self.logits_test = self.peleenet(self.input_test, is_training=False, reuse=True)
        # self.one_hot_labels = tf.one_hot(self.input_label, self.num_class)
        
        # loss function
        def softmax_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.softmax_cross_entropy_with_logits(targets=x, labels=y)
        # weights regularization
        self.weights_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(softmax_cross_entropy_with_logits(self.logits_train, self.one_hot_labels)) + self.config.weight_decay*self.weights_reg
        
        # optimizer
        '''
        self.adam_optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                                 beta1=self.config.beta1,
                                                 beta2=self.config.beta2).minimize(self.loss)
        '''
        self.rmsprop_optim = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate,
                                                       momentum=self.config.momentum).minimize(self.loss)

        # accuracy
        self.predicetion = tf.nn.softmax(self.logits_test, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predicetion, 1), tf.argmax(self.one_hot_labels, 1)), tf.float32))
        
        # summary
        self.loss_summary = tf.summary.scalar('entrophy loss', self.loss)
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        
        # saver
        self.saver = tf.train.Saver()
        
    def train_model(self):
        # initialize variables
        tf.global_variables_initializer().run()
        
        # load model
        if self.load_model():
            print('load model successfully')
        else:
            print('fail to load model')
            
        # get datasource
        datasource = get_data(self.config.dataset, is_training=True)
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=True)
        ites_per_epoch = int(len(datasource.images)/self.batchsize)
        
        step = 0
        for epoch in range(self.config.epochs):
            for ite in tqdm(range(ites_per_epoch)):
                images, labels = next(gen_data)
                _, loss, accuracy, summaries = self.sess.run([self.rmsprop_optim, self.loss, self.accuracy, self.summaries], feed_dict={
                                                                                        self.input_train:images,
                                                                                        self.input_test:images,
                                                                                        self.one_hot_labels:labels
                                                                                        })
                
                step = step + 1
                self.summary_writer.add_summary(summaries, global_step=step)
            
            # test model
            if np.mod(epoch, 1) == 0:
                print('--epoch_{} -- training accuracy:{}'.format(epoch, accuracy))
                self.test_model()

            # save model
            if np.mod(epoch, 5) == 0:
                self.save_model()
        
    def test_model(self):
        if not self.config.is_training:
            # initialize variables
            tf.global_variables_initializer().run()
            # load model
            if self.load_model():
                print('load model successfully')
            else:
                print('fail to load model')
        
        datasource = get_data(self.config.dataset, is_training=False)      
        gen_data = gen_batch_data(datasource, self.batchsize, is_training=False)
        ites_per_epoch = int(len(datasource.images)/self.batchsize)
        
        accuracy = []
        for ite in range(ites_per_epoch):
            images, labels = next(gen_data)
            accuracy_per_epoch = self.sess.run([self.accuracy], feed_dict={
                                                                            self.input_test:images,
                                                                            self.one_hot_labels:labels
                                                                            })
            accuracy.append(accuracy_per_epoch[0])
    
        acc = np.mean(accuracy)
        print('--test epoch -- accuracy:{:.4f}'.format(acc))
        
    # load model
    def load_model(self):
        if not os.path.isfile(os.path.join(self.model_dir, 'checkpoint')):
            return False
        self.save.restore(self.sess, self.model_pos)
    
    # save model
    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.saver.save(self.sess, self.model_pos)
    
    @property
    def model_dir(self):
        return '{}/{}'.format(self.config.checkpoint_dir, self.config.dataset)
    
    @property
    def model_pos(self):
        return '{}/{}/{}'.format(self.config.checkpoint_dir, self.config.dataset, self.model_name)

if __name__=='__main__':
    input_x = tf.placeholder(tf.float32, [64, 224,224,3], name='input_train')
    peleenet = PeleeNet()
    start_time = time.time()
    output = peleenet.peleenet(input_x)
    end_time = time.time()
    print('total time:{}'.format(end_time-start_time))
    print(output.get_shape().as_list())


```

#### 2.2Ablation Study(消融实验)

**Dataset**

自定义了 Stanford Dogs 数据集用来进行消融实验(从 ILSVRC 2012 数据集的子集中创建)

+ 类别数: 120
+ 训练集图片数: 150466
+ 验证集图片数: 6000

**Effects of Various Design Choices on the Performance:**

我们构建了一个类似于 DenseNet 的网络, 并将其命名为 DenseNet-41, 作为我们的 baseline 模型. 该模型和原始的 DenseNet 模型有两点不同. 第一, 首层 conv layer 参数不同, 其通道数设定为 24 而不是 64, 核的大小从 `7×7`改变到 `3×3`. 第二点不同是, 调整了每个 dense block 中的层的数量以满足算力限制.

我们在这部分的模型都是有 batch size 为 256 的 `PyTorch`进行 120 epochs 的训练. 我们遵循了 ResNet 的大多数训练设置和超参数. 下表显示了各种设计选择对性能的影响. 可以看到, 在综合了所有这些设计选择以后, Peleenet 的准确率达到了 79.25%, 比 DenseNet-41 的准确率高 4.23%. 并且计算成本更低.

<div align=center>
<img src="zh-cn/img/pelee/p6.png" /> 
</div>

#### 2.3 Results on ImageNet 2012

<div align=center>
<img src="zh-cn/img/pelee/p7.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/pelee/p8.png" /> 
</div>

#### 2.4 Speed on Real Devices

<div align=center>
<img src="zh-cn/img/pelee/p9.png" /> 
</div>

使用 FP16 而不是 FP32 是一个常用的在 inference 阶段的加速方法. 但是基于 depthwise separable convolution 的网络却很难从 TX2 的 half-precision(FP16)中获益, 如下图所示.

<div align=center>
<img src="zh-cn/img/pelee/p10.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/pelee/p11.png" /> 
</div>


### 3.Pelee A Real-Time Object Detection System

#### 3.1 Overview

本小节介绍了我们的目标检测系统以及对SSD做出的一些优化. **我们的优化目的主要是在提升速度的同时还要保持一定的精确度. **除了我们上一节提到的特征提取网络以外, 我们还构建与原始SSD不同的目标检测网络, 并精心选择了一组`5` 个尺度的特征图. 同时, 对于每一个用于检测的特征图谱, 我们在进行预测之前建立了一个残差块(如图). 我们还使用小卷积核来预测对象类别和边界框位置, 以降低计算成本. 此外, 我们使用了非常不同的训练超参数. 尽管这些贡献单独看起来影响很小, 但是我们注意到最终的模型在 PASCAL VOC 2007 上达到了 70.9% 的 mAP, 在 MS COCO 上实现了 22.4 的 mAP.

在我们的模型中我们使用了 5 种尺寸的特征图谱: `19, 10, 5, 3, 1`. 我们没有使用 38 大小的特征图谱是为了平衡速度与精度. `19 大小的特征图谱使用了2种尺寸的 default boxes, 其他 4 个特征图谱使用了1种尺寸的 default box`. Speed/Accuracy Trade-offs 论文中在使用 SSD 与 MobileNet 结合时, 也没有使用 38 尺寸的特征图谱. 但是, 他们额外添加了一个 `2×2` 的特征图谱来保留6个尺寸的特征图谱进行预测, 这与我们的解决方案不同.

<div align=center>
<img src="zh-cn/img/pelee/p12.png" /> 
</div>

#### 3.2 Results on VOC 2007

我们的目标检测模型是基于 SSD 的源码实现的(Caffe). batch-size 为 32, 初始的 learning rate 为 0.005, 然后在 80k 和 100k 次迭代时降低 10 倍. 总的迭代数是 120K.

**Effects of Various Design Choices**

下表显示了不同设计选择对性能的影响. 我们可以看到残差预测模块可以有效的提升准确率. 有残差预测模块的模型比无残差预测模块的模型精度高 2.2%. 使用 1×1 卷积核进行预测的模型和使用 3×3 的模型的精度几乎相同. 但是 `1×1` 的内核减少了 21.5% 的计算成本和 33.9% 的模型大小.

<div align=center>
<img src="zh-cn/img/pelee/p13.png" /> 
</div>

**Comparison with Other Frameworks**

下表显示了我们的模型与其他不同模型的对比

<div align=center>
<img src="zh-cn/img/pelee/p14.png" /> 
</div>

**Speed on Real Devices**

<div align=center>
<img src="zh-cn/img/pelee/p15.png" /> 
</div>


#### 3.3 Results on COCO

<div align=center>
<img src="zh-cn/img/pelee/p16.png" /> 
</div>

### 4.Conclusion

深度可分卷机不是唯一构建高效模型的选择，PeleeNet和pelee在ILSVRC2012，VOC2007和COCO数据集上的表现得到验证.

PeleeNet和Pelee目标检测系统可以在移动端设备上做到实时检测，在Iphone8上测试，pelee目标检测系统的帧率为：23.6FPS, NVIDIA TX2上的帧率wei 125FPS，且有较高的准确度.


### Reference

[1].https://arxiv.org/abs/1804.06882

[2].https://hellozhaozheng.github.io/z_post/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89-Pelee-NIPS2018/

[3].https://github.com/nnUyi/PeleeNet

[4].[caffe版pelee detection](https://github.com/tangtangchx/Pelee)

[5].[keras版pelee detection](https://github.com/markshih91/pelee_keras)

