## 从MobileNet到ShuffleNet

### 1.Xception

------

Xception是google继Inception后提出的对Inception v3的另一种改进，主要是采用depthwise separable convolution来替换原来Inception v3中的卷积操作。要学习Xception首先要对Inception有详细的了解，论文中说Xception比Inception V3还要牛B一些。

<div align=center>
<img src="zh-cn/img/xception/p1.png" />
</div>

Inception背后的基本假设是使用交叉通道相关性和空间相关性充分解耦，最好不要将他们联合起来。上图是一个标准的Inception V3结构。其简化版本如下图所示：

<div align=center>
<img src="zh-cn/img/xception/p2.png" />
</div>


<div align=center>
<img src="zh-cn/img/xception/p3.png" />
</div>

上图是等价的Inception对于一个输入，先用一个统一的1x1卷积核卷积，然后连接3个3x3的卷积，这3个卷积操作只将前面1x1卷积结果中的一部分作为自己的输入（这里是将1/3channel作为每个3x3卷积的输入）。

<div align=center>
<img src="zh-cn/img/xception/p4.png" />
</div>

extreme version,3x3卷积的个数和1x1卷积的输出channel个数一样，每个3*3卷积都是和1个输入chuannel做卷积。


在Xception中主要采用depthwise separable convolution，什么是depthwise separable convolution,我们会在MobileNet V1中给出详细的讲解，或详细的参考`N种卷积`这一章的内容。
但是Iception结构的极端情况和depthwise separable convolution有如下细微的差别：

+ 操作顺序，depthwise separable convolution 就是TensorFlow中的通常实现，先执行通道空间的卷积在执行逐点卷积1x1卷积，而Inception的极端情况，首先执行1x1的卷积。

+ 第一次操作后是否存在非线性激活操作，Inception中两个操作后都跟着ReLu激活，而DSC的通常实现时没有非线性操作。

而最终的Xception是一个带有残差连接的depthwise separable convolution层的线性堆叠

<div align=center>
<img src="zh-cn/img/xception/p5.png" />
</div>

**实验结果**

参数配置

+ On ImageNet

	- Optimizer:SGD
	- Momentum:0.9
	- Initial learning rate: 0.045
	- Learning rate decay: decay of rate 0.94 every 2 epochs

+ On JFT(JFT是Google内部标注的数据集 Hinton)

	- Optimizer:RMSprop
	- Momentum:0.9
	- Initial learning rate: 0.001
	- Learning rate decay: decay of rate 0.94 every 3,000,000 samples

下面两个表是ImageNet和JFT数据集的几个网络的对比结果:

<div align=center>
<img src="zh-cn/img/xception/p6.png" />
</div>
<div align=center>
<img src="zh-cn/img/xception/p7.png" />
</div>

网络结构中并没有在可分卷积和逐点卷积之间加入Relu或Elu等非线性层，因为这样实验效果并不是很好。
Xception作为Inception v3的改进，主要是在Inception v3的基础上引入了depthwise separable convolution，在基本不增加网络复杂度的前提下提高了模型的效果。有些人会好奇为什么引入depthwise separable convolution没有大大降低网络的复杂度，因为depthwise separable convolution在mobileNet中主要就是为了降低网络的复杂度而设计的。原因是作者加宽了网络，使得参数数量和Inception v3差不多，然后在这前提下比较性能。因此Xception目的不在于模型压缩，而是提高性能。



### 2.MobileNet(V1,V2)

#### MobileNet:Efficient Convolutional Neural NetWorks for Mobile Vision Applications

##### 0.摘要
MobileNet V1 为移动和嵌入式视觉应用提供了一类高效模型。 基于流线型架构，使用深度可分离卷积来构建轻量级深度神经网络。 模型引入两个简单的全局超参数，可以在时间和准确性之间进行有效折中。 这些超参数允许模型构建者根据问题的约束为其应用程序选择合适大小的模型。 MobileNet V1 paper中在资源和精度折中方面提出了大量实验，并且与ImageNet分类上的其他流行模型相比显示出强大的性能。 并且演示MobileNets在广泛的应用和用例（包括对象检测，精细分类，人脸属性提取和大规模地理定位）中的有效性。

##### 1.引言

自从AlexNet赢得ImageNet挑战赛（ILSVRC2012）推广深度卷积神经网络以后，卷积神经网络在计算机视觉领域变得普遍存在。总得趋势是使用更深层更复杂的网络来实现更高的准确度。然而，这些提高准确率的进步并不一定会使网络在尺寸和速度方面更有效率。在许多现实世界的应用中，例如机器人，自动驾驶汽车和AR技术，识别任务需要及时地在有限计算资源的平台上进行。本文介绍了一种高效的网络结构和两个超参数，以便构建非常小的，快速度的模型，可以轻松匹配移动和嵌入式视觉应用的设计要求。第二节回顾了之前在构建小模型的工作。第三节描述了MobileNets架构和两个超参数（widthmultiplier和resolutionmultiplier），以便定义更小更高效的MobileNets。第四节介绍了ImageNet上的实验以及各种不同的应用和用例。第五节以汇总和结论来结束论文。

##### 2.Prior Work(前人的工作)

在最近的文献中，关于建立小型高效的神经网络的兴趣日益增加。一般来说，这些不同的方法可以归为两类，压缩训练好的模型和直接训练小网络模型。本文提出了一类网络结构，允许模型开发人员选择与其应用程序的资源限制（延迟，大小）相匹配的小型网络。MobileNets主要侧重于优化延迟，但也能够产生小型网络。很多关于小型网络的论文只关注大小，却不考虑速度。 

MobileNets主要由深度可分离
卷积构建。扁平化的网络通过完全分解的卷积建立，并显示出极大的因式分解网络的潜力。独立于本文的FactorizedNetworks论文，引入了类似于因式分解卷积以及拓扑连接的使用。随后，Xception网络展示了如何将深度可分离的过滤器扩展到比InceptionV3网络还优秀。另一个小型网络是SqueezeNet，它采用瓶颈方法来设计一个非常小的网络。其他减少计算量的网络包括structured transform networks和deep fried convnets。 
获取小型网络的另一个办法是缩小，分解或者压缩训练好的模型。文献中基于压缩的方法有product quantization，哈希法与pruning，vector quantization和霍夫曼编码压缩。此外，各种各样的因式分解方法被提出来用于加速训练好的网络。训练小型网络的另一种方法是distillation（蒸馏法），使用更大的网络来学习小一点的网络。另一种新兴的方法是low bit networks。

<!-- Figure1 -->
<div align=center>
<img src="zh-cn/img/mobilenet/v1/p1.png" />
</div>


##### 3.MobileNet Architecture 结构

在本MobileNet构建的核心层:深度可分卷积。模型机构中融入两个模型缩小参数（widthmultiplier和resolutionmultiplier）

**3.1 深度可分卷积**

MobileNet模型是基于深度可分离卷积，它是factorized convolutions的一种，而factorized convolutions将标准化卷积分解为深度卷积和1x1卷积（pointwise convolution）。对于MobileNets，深度卷积将单个滤波器应用到每一个输入通道。然后，pointwise convolution用1x1卷积来组合不同深度卷积的输出。深度可分离卷积将其分成两层，一层用于滤波，一层用于组合。这种分解过程能极大减少计算量和模型大小。Figure2展示了如何将一个标准卷积分解为深度卷积和1x1的pointwise convolution。 

<!-- Figure2 -->
<div align=center>
<img src="zh-cn/img/mobilenet/v1/p2.png" />
</div>


<div align=center>
<img src="zh-cn/img/mobilenet/v1/p3.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v1/p4.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v1/p5.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v1/p6.png" />
</div>


**3.2 网络结构和训练**

如前面部分所述的那样，MobileNet结构是由深度可分离卷积建立的，其中第一层是全卷积。通过以简单的方式定义网络，可以轻松地探索网络拓扑，以找到一个良好的网络。MobileNet的结构定义在Table1中。所有层之后都是BatchNormalization和ReLU非线性激活函数，但是最后的全连接层例外，它没有非线性激活函数，直接到softmax层进行分类。Figure 3比较了常规卷积和深度可分离卷积（都跟着BN层和ReLU层）。在depthwise convolution和第一层卷积层中都能处理下采样问题。最后一个平均池化层在全连接层之前，将特征图的维度降维1x1。如果将depthwise convolution和pointwise convolution算为不同的层，MobileNet有28层。 

<!-- Table 1 -->
<div align=center>
<img src="zh-cn/img/mobilenet/v1/p7.png" />
</div>

*MobileNet V1的结构*

<!-- Figure 3 -->
<div align=center>
<img src="zh-cn/img/mobilenet/v1/p8.png" />
</div>

*左图是标准卷积，右图是深度可分离卷积*


仅仅通过少量的Mult-Adds简单地定义网络是不够的。确保这些操作能有效实现也很重要。例如，非结构化的稀疏矩阵操作通常不比密集矩阵运算快，除非是非常稀疏的矩阵。MobileNet V1模型结构将几乎全部的计算复杂度放到了1x1卷积中。这可以通过高度优化的通用矩阵乘法（GEMM）功能来实现。通常卷积由GEMM实现，但需要在称为im2col的内存中进行初始重新排序，以将其映射到GEMM。这个方法在流行的Caffe包中正在使用。1x1的卷积不需要在内存中重新排序而可以直接被GEMM（最优化的数值线性代数算法之一）实现。MobileNet在1x1卷积花费了95%计算复杂度，也拥有75%的参数（Table 2）。几乎所有的额外参数都在全连接层。

<!-- Table 2 -->
<div align=center>
<img src="zh-cn/img/mobilenet/v1/p9.png" />
</div>

*每种层的分布*


使用类似于InceptionV3的异步梯度下降的RMSprop，MobileNet模型在TensorFlow中进行训练。然而，与训练大模型相反，我们较少地使用正则化和数据增强技术，因为小模型不容易过拟合。当训练MobileNets时，通过限制croping的尺寸来减少图片扭曲。另外，我们发现重要的是在depthwise滤波器上放置很少或没有重量衰减（L2正则化），因为它们参数很少。

**3.3. Width Multiplier: Thinner Models(alpha参数：更小的模型)**

尽管基本MobileNet架构已经很小,延迟很低，但特定用例或应用程序需要更小更快的模型。为了构建这些较小且计算量较少的模型，我们引入了一个非常简单的参数α，称为width multiplier。这个参数widthmultiplier的作用是在每层均匀地减负网络。对于一个给定的层和widthmultiplier α，输入通道的数量从M变成αM，输出通道的数量从N变成αN。深度可分离卷积（以widthmultiplier参数α为计）的计算复杂度： 
α∈(0,1]，通常设为1，0.75，0.5和0.25。α=1表示基准MobileNet，而α<1则表示瘦身的MobileNets。Width multiplier有减少计算复杂度和参数数量（大概α二次方）的作用。Width multiplier可以应用于任何模型结构，以定义一个具有合理准确性，延迟和尺寸的新的较小的模型。它用于定义新的简化结构，但需要重新进行训练。


**3.4. Resolution Multiplier: Reduced Representation**

降低神经网络的第二个超参数是resolution multiplier ρ。我们将其应用于输入图像，并且每个层的内部特征随后被减去相同的乘数。实际上，我们通过设置输入分辨率隐式设置ρ。我们现在可以将网络核心层的计算复杂度表示为具有width multiplier α和resolution multiplier ρ的深度可分离卷积：ρ∈(0,1]，通常设为224,192,160或者128。ρ=1是基本MobileNets而ρ<1示瘦身的MobileNets。Resolutionmultiplier可以减少计算复杂度ρ的平方。作为一个例子，我们可以看一下MobileNet中的一个典型的卷积层，看看深度可分离卷积，width multiplier和resolution multiplier如何降低计算复杂度和参数。Table 3显示了不同的架构收缩方法应用于该层，而表示的计算复杂度和参数数量。第一行显示了Multi-Adds和参数，其中Multi-Adds具有14x14x512的输入特征图，内核K的大小为3x3x512x512。我们将在下一节中详细介绍模型资源和准确率的折中

<!-- Table 3 -->
<div align=center>
<img src="zh-cn/img/mobilenet/v1/p10.png" />
</div>

##### 4.实验

该部分详细的展示了MobileNet V1 设置的两个超参数的作用和MobileNet V1 在图像分类，目标检测和人脸识别的移动端的应用场景，具体的结果参考MobileNet paper。


##### 5.结论

MobileNets一种基于深度可分离卷积的新型模型架构。讨论了一些可以得到高效模型的重要设计决策。然后，演示了如何通过使用widthmultiplier和resolutionmultiplier来构建更小更快的MobileNets，以减少合理的精度来减少模型尺寸和延迟。将不同的MobileNets与流行的模型进行比较，展示了尺寸，速度和精度。最终通过展示MobileNet在应用于各种任务时的有效性来得出结论。作为帮助采用和探索MobileNets的下一步，作者们已在TensorFlow中发布模型。

附：MobileNet V1 code

```python

#用mobilenet实现对mnist数据集进行训练， 代码如下：

import tensorflow as tf
from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"

# create variable
def create_variable(name, shape, initializer,
    dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
            initializer=initializer, trainable=trainable)

"""
公式如下：
y=γ(x-μ)/σ+β
其中x是输入，y是输出，μ是均值，σ是方差，γ和β是缩放（scale）、偏移（offset）系数。
一般来讲，这些参数都是基于channel来做的，比如输入x是一个16*32*32*128(NWHC格式)的feature map，
那么上述参数都是128维的向量。其中γ和β是可有可无的，有的话，就是一个可以学习的参数（参与前向后向），
没有的话，就简化成y=(x-μ)/σ。而μ和σ，在训练的时候，使用的是batch内的统计值，测试/预测的时候，采用的是训练时计算出的滑动平均值。    
"""
# batchnorm layer

def bacthnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
    inputs_shape = inputs.get_shape().as_list()#inputs代表输入,scope代表命名
    params_shape = inputs_shape[-1:]#得到numpy数组的维度，例如[3,4]
    axis = list(range(len(inputs_shape) - 1))#得到最后一个维度[4]

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_variable("moving_mean", params_shape,
                            initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_variable("moving_variance", params_shape,
                            initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=momentum)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)#低级的操作函数，调用者需要自己处理张量的平均值和方差。


# depthwise conv2d layer
def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, channel_multiplier],
                       initializer=tf.truncated_normal_initializer(stddev=0.01))

    return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, strides, strides, 1],
                                padding="SAME", rate=[1, 1])

# conv2d layer
def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, num_filters],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.conv2d(inputs, filter, strides=[1, strides, strides, 1],
                        padding="SAME")

# avg pool layer
def avg_pool(inputs, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding="VALID")

# fully connected layer
def fc(inputs, n_out, scope, use_bias=True):
    inputs_shape = inputs.get_shape().as_list()
    n_in = inputs_shape[-1]
    with tf.variable_scope(scope):
        weight = create_variable("weight", shape=[n_in, n_out],
                    initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:
            bias = create_variable("bias", shape=[n_out,],
                                   initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)
        return tf.matmul(inputs, weight)


class MobileNet(object):
    def __init__(self, inputs, labels, num_classes=10, is_training=True,
                 width_multiplier=0.5, scope="MobileNet", ):
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.width_multiplier = width_multiplier


        # construct model
        with tf.variable_scope(scope):
            # conv1
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier), filter_size=1,
                         strides=2)  # ->[N, 14, 14, 1]
            net = tf.nn.relu(bacthnorm(net, "conv_1/bn", is_training=self.is_training))
            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                "ds_conv_2") # ->[N, 14, 14, 64]
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_3") # ->[N, 14, 14, 128]
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                "ds_conv_4") # ->[N, 14, 14, 128]
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_5") # ->[N, 14, 14, 256]
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                "ds_conv_6") # ->[N, 14, 14, 256]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_7") # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_8") # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_9")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_10")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_11")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                "ds_conv_12")  # ->[N, 14, 14, 512]
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_13", downsample=True) # ->[N, 7, 7, 1024]
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                "ds_conv_14") # ->[N, 7, 7, 1024]
            net = avg_pool(net, 7, "avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, "fc_16")
            self.predictions = tf.nn.softmax(self.logits)
            self.loss = -tf.reduce_mean(labels * tf.log(self.predictions))

    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
        """depthwise separable convolution 2D function"""
        num_filters = round(num_filters * width_multiplier)
        strides = 2 if downsample else 1

        with tf.variable_scope(scope):
            # depthwise conv2d
            dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
            # batchnorm
            bn = bacthnorm(dw_conv, "dw_bn", is_training=self.is_training)
            # relu
            relu = tf.nn.relu(bn)
            # pointwise conv2d (1x1)
            pw_conv = conv2d(relu, "pointwise_conv", num_filters)
            # bn
            bn = bacthnorm(pw_conv, "pw_bn", is_training=self.is_training)
            return tf.nn.relu(bn)

if __name__ == "__main__":
    # test data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    x_label = tf.reshape(y_, [-1, 10])
    mobileNet = MobileNet(x_image, x_label)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(mobileNet.loss)
    keep_prob = tf.placeholder("float")
    correct_prediction = tf.equal(tf.arg_max(mobileNet.predictions, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accurary %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


tf.nn.separable_conv2d()

```


#### MobileNet V2: Inverted Residuals and Linear Bottlenecks

##### 0.摘要

MobileNet V2提高了移动模型在多个任务和多个基准数据集以及在不同模型尺寸范围内的最佳性能。在SSDLite新框架中将移动模型应用于目标检测。通股票DeepLab V3的简化形式，称为Mobile DeepLab V3来构建移动端的予以分割模型。

MobileNet V2架构基于倒置的采茶结构，其中shortcut位于窄的瓶颈层之间，中间展开层使用轻量级的深度卷积作为非线性源来过滤特征，此外，作者发现为了保持表现能力，去除了窄层中的非线性是非常重要的，作者在ImageNet,COCO,VOC等数据集均作了模型性能的评估。

##### 1.引言

神经网络已经彻底改变了机器智能的许多领域，使具有挑战性的图像识别任务获得了超过常人的准确性。然而，提高准确性的驱动力往往需要付出代价；现代先进网络需要超出许多移动和嵌入式应用能力之外的高计算资源。

本文介绍了一种专为移动和资源首先环境量身定制的新型神经网络架构。我们的网络通过显著减少所需要操作和内存的数量，同时保持相同的精度推进了移动定制计算机视觉模型的最新水平。

MobileNet V2的主要贡献是一个新的层模块：**具有线性瓶颈的倒置残差**。该模块将输入的低维压缩表示首先扩展到高维并用轻量级深度卷积进行过滤。随后用线性卷积将特征投影到低维表示。官方实现以作为TensorFloe-Slim模型库的一部分。


这种卷积特别适用于移动设计，因为它可以通过从不完全实现大型中间张量来显著减少推断过程中所需要的内存占用，这减少了许多嵌入式硬件设计中对主存储器访问的需求。


##### 2.相关工作

调整深层神经网络架构以在精确性和性能之间达到最佳平衡已成为过去几年研究活跃的一个领域。由许多团队进行的手动架构搜索和训练算法的改进，已经比早期的设计（如AlexNet,VGGNet,GoogleNet和ResNet）有了显著的改进。最近在算法架构探索方面取得了很多进展，包括超参数优化[9,10,11],各种网络修剪方法[12,13,14,15,16,17]和连接学习[18,19]。也有大量的工作致力于改变内部卷积块的连接结构如ShufleNet或引入稀疏性[21]。

最近，[23,24,25,26]开辟了一个新的方向，将遗传算法和强化学习等优化方法带入架构搜索。一个缺点是最终所得到的网络非常复杂。我们的方法基于MobileNet V1,它保留了其简单性，并且不需要任何特殊的运算符，同时显著额提高了它的准确性，为移动应用实现了在多种图像分类和监测任务上的最新技术。


<div align=center>
<img src="zh-cn/img/mobilenet/v2/p1.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p2.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p3.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p4.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p5.png" />
</div>

*残差块和倒置残差之间的差异。对角阴影线层不使用非线性。我们用每个块的厚度来表明其相对数量的通道。注意经典残差是如何将通道数量较多的层链接起来的，而倒置残差则是链接瓶颈，最好通过颜色看*

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p6.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p7.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p8.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p9.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p10.png" />
</div>

<div align=center>
<img src="zh-cn/img/mobilenet/v2/p11.png" />
</div>

*表2：MobileNet V2:每行描述一个或多个相同（模步长）层的序列，重复n次。统一序列中的所有图层具有相同数量的c个输出通道。每个序列的第一层有一个步长s，所有其他的都使用步长1.所有空间卷积使用3X3的核，宽展系数t总是应用于输入尺寸，如表1所示*

+ t: 扩展因子
+ c: 输出的通道数
+ n: 重复的次数
+ s: 步长


<div align=center>
<img src="zh-cn/img/mobilenet/v2/p12.png" />
</div>


<div align=center>
<img src="zh-cn/img/mobilenet/v2/p13.png" />
</div>

*上图描述了不同网络结构的差异，我们主要看一下MobileNet V2，只有stride=1时，才会用elementwise的sum将输入输出链接（左图）；stride=2时无shortcut链接输入和输出特征*


##### 5-6.内存的高效利用和实验

第5,6两部分描述了MobileNet V2的内存使用和在图像分类，目标检测，语义分割上的实验结果，具体的细节可以参考MobileNet V2的paper中的这两部分的细节。


##### 7.结论

MobileNet V2 适用于移动应用，允许非常有效的内存推断。

对于ImageNet数据集，我们的架构改善了许多性能点的最新技术水平。对于目标检测任务，我们的网络在精度和模型的复杂度方面优于COCO数据集上的最新实时检测器。值得注意的是，我们的架构与SSDLite检测模块相比计算少了20倍，参数比YOLO V2少10倍。

##### Reference

[1] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. Int. J. Comput. Vision, 115(3):211–252, December 2015.

[2] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dolla?r, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[3] Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserma. The pascal visual object classes challenge a retrospective. IJCV, 2014.

[4] Mobilenetv2 source code. Available from https://github.com/tensorflow/ models/tree/master/research/slim/nets/mobilenet.

[5] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Bartlett et al. [48], pages 1106–1114.

[6] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014.

[7] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 1–9. IEEE Computer Society, 2015.

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.

[9] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13:281–305, 2012.

[10] Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical bayesian optimization of machine learning algorithms. In Bartlett et al. [48], pages 2960–2968.

[11] Jasper Snoek, Oren Rippel, Kevin Swersky, Ryan Kiros, Nadathur Satish, Narayanan Sundaram, Md. Mostofa Ali Patwary, Prabhat, and Ryan P. Adams. Scalable bayesian optimization using deep neural networks. In Francis R. Bach and David M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 2171–2180. JMLR.org, 2015.

[12] Babak Hassibi and David G. Stork. Second order derivatives for network pruning: Optimal brain surgeon. In Stephen Jose Hanson, Jack D. Cowan, and C. Lee Giles, editors, Advances in Neural Information Processing Systems 5, [NIPS Conference, Denver, Colorado, USA, November 30 - December 3, 1992], pages 164–171. Morgan Kaufmann, 1992.

[13] Yann LeCun, John S. Denker, and Sara A. Solla. Optimal brain damage. In David S. Touretzky, editor, Advances in Neural Information Processing Systems 2, [NIPS Conference, Denver, Colorado, USA, November 27-30, 1989], pages 598–605. Morgan Kaufmann, 1989.

[14] Song Han, Jeff Pool, John Tran, and William J. Dally. Learning both weights and connections for efficient neural network. In Corinna Cortes, Neil D. Lawrence, Daniel D. Lee, Masashi Sugiyama, and Roman Garnett, editors, Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pages 1135–1143, 2015.

[15] Song Han, Jeff Pool, Sharan Narang, Huizi Mao, Shijian Tang, Erich Elsen, Bryan Catanzaro, John Tran, and William J. Dally. DSD: regularizing deep neural networks with dense-sparse-dense training flow. CoRR, abs/1607.04381, 2016.

[16] Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pages 1379–1387, 2016.

[17] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. CoRR, abs/1608.08710, 2016.

[18] Karim Ahmed and Lorenzo Torresani. Connectivity learning in multi-branch networks. CoRR, abs/1709.09582, 2017.

[19] Tom Veniat and Ludovic Denoyer. Learning time-efficient deep architectures with budgeted super networks. CoRR, abs/1706.00046, 2017.

[20] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. CoRR, abs/1707.01083, 2017.

[21] Soravit Changpinyo, Mark Sandler, and Andrey Zhmoginov. The power of sparsity in convolutional neural networks. CoRR, abs/1702.06257, 2017.

[22] Min Wang, Baoyuan Liu, and Hassan Foroosh. Design of efficient convolutional layers using single intra-channel convolution, topological subdivisioning and spatial ”bottleneck” structure. CoRR, abs/1608.04337, 2016.

[23] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures for scalable image recognition. CoRR, abs/1707.07012, 2017.

[24] Lingxi Xie and Alan L. Yuille. Genetic CNN. CoRR, abs/1703.01513, 2017.

[25] Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc V. Le, and Alexey Kurakin. Large-scale evolution of image classifiers. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 2902–2911. PMLR, 2017.

[26] Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. CoRR, abs/1611.01578, 2016.

[27] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam.
Mobilenets: Efficient convolutional neural networks for mobile vision applications. CoRR, abs/1704.04861, 2017.

[28] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[29] Dongyoon Han, Jiwhan Kim, and Junmo Kim. Deep pyramidal residual networks. CoRR, abs/1610.02915, 2016.

[30] Saining Xie, Ross B. Girshick, Piotr Dolla ?r, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. CoRR, abs/1611.05431, 2016.

[31] Mart?n Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane?, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Vie?gas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[32] Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. Caffe: Convolutional architecture for fast feature embed- ding. arXiv preprint arXiv:1408.5093, 2014.

[33] Jonathan Huang, Vivek Rathod, Chen Sun, Men- glong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, et al. Speed/accuracy trade-offs for modern convolutional object detectors. In CVPR, 2017.

[34] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In ECCV, 2016.

[35] Joseph Redmon and Ali Farhadi. Yolo9000: Better, faster, stronger. arXiv preprint arXiv:1612.08242, 2016.

[36] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, pages 91–99, 2015.

[37] Jifeng Dai, Yi Li, Kaiming He, and Jian Sun. R-fcn: Object detection via region-based fully convolutional networks. In Advances in neural information processing systems, pages 379–387, 2016.

[38] Jonathan Huang, Vivek Rathod, Derek Chow, Chen Sun, and Menglong Zhu. Tensorflow object detection api, 2017.

[39] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. CoRR, abs/1706.05587, 2017.

[40] Matthias Holschneider, Richard Kronland-Martinet, Jean Morlet, and Ph Tchamitchian. A real-time algorithm for signal analysis with the help of the wavelet transform. In Wavelets: Time-Frequency Methods and Phase Space, pages 289–297. 1989.

[41] Pierre Sermanet, David Eigen, Xiang Zhang, Michae?l Mathieu, Rob Fergus, and Yann LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. arXiv:1312.6229, 2013.

[42] George Papandreou,Iasonas Kokkinos, and Pierre Andre Savalle. Modeling local and global deformations in deep learning: Epitomic convolution, multiple instance learning, and sliding window detection. In CVPR, 2015.

[43] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. TPAMI, 2017.

[44] Wei Liu, Andrew Rabinovich, and Alexander C. Berg. Parsenet: Looking wider to see better. CoRR, abs/1506.04579, 2015.

[45] Bharath Hariharan, Pablo Arbela?ez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In ICCV, 2011.

[46] Christian Szegedy, Sergey Ioffe, and Vincent Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. CoRR, abs/1602.07261, 2016.

[47] Guido Montu?far, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of linear regions of deep neural networks. In Proceedings of the 27th International Conference on Neural Information Processing Systems, NIPS’14, pages 2924–2932, Cambridge, MA, USA, 2014. MIT Press.

[48] Peter L. Bartlett, Fernando C. N. Pereira, Christopher J. C. Burges, Le?on Bottou, and Kilian Q. Weinberger, editors. Advances in Neural Information Processing Systems 25: 26th Annual Conference on Neural Information Processing Systems 2012. Proceedings of a meeting held December 3-6, 2012, Lake Tahoe, Nevada, United States, 2012.



------

### 3.ShuffleNet(V1,V2)

### 3.1 (ShuffleNet V1) ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile

#### 0.摘要

论文介绍了一个效率极高的CNN架构ShuffleNet，专门用于计算力受限的移动设备。新的架构利用两个操作，**逐点群卷积（pointwise group convolution)和通道混洗(channel shuffle)**,在ImageNet和COCO数据集上分类任务比MobileNet高，性能达到40MFLOPs，性能是AlexNet的13倍。

#### 1.介绍

现在许多CNNs模型的发展方向是更大更深，这让深度网络很难运行在移动设备上，针对这一问题，许多工作重点都放在了对现有预训练的模型的修剪，压缩和使用低精度数据表示。

论文提出的ShuffNet是探索一个可以满足受限的条件的高效基础框架。论文的Insight是现有的先进base架构如Xception,ResNeXt在小型网络中效率低，因为大量的1X1卷积耗费很多计算资源，论文提出了逐点群卷积来帮助降低计算复杂度，但是使用逐点群卷积会有副作用，故在此基础上，论文提出了通道混洗来帮助通道信息流通。基于这两个trick,构建了一个名为ShuffleNet的高效架构，相比于其他先进的网络，对于给定的计算复杂度预算，ShuffleNet允许使用更多的特征映射通道，在小型网络上有助于编码更多信息。

论文在ImageNet和COCO数据集上做了相关实验，展现出了ShuffleNet设计原理的有效性和结构优越性。同时在论文还讨论了在真实嵌入式设备上的运行效率。


#### 2.Related Work

+ **高效模型设计**: CNNs在CV任务中取得了极大的成功，在嵌入设备上运行高质量的深度神经网络需求越来越大，这也促进了对高效模型的探究。例如：与单纯的堆叠卷积层，GoogleNet增加了网络的宽度，复杂度降低很多；SqueezeNet在保持精度的同时，大大减小参数和计算量；ResNet利用高效的Bottleneck结构实现惊人的效果。Xception中提出深度可分卷积概括了Inception序列。MobileNet利用深度可分卷积构建的轻量级模型获得了先进的成果；ShuffleNet的工作是推广分组卷积和深度可分卷积。

+ **模型加速**: 该方向意在保持训练模型的精度的同时加速Inference过程。常见的工作有：通过修剪网络连接或减少通道数减少模型中链接冗余；量化和因式分解减少计算中的冗余；不修改参数的前提下，通过FFT和其他方法优化卷积计算的消耗；蒸馏将大模型的知识转化为小模型，使得小模型的训练更加容易；ShuffleNet的工作专注于设计更好的模型，直接提高性能，而不是加速或转化现有的模型。

#### 3.Approach

**针对群卷积的通道混洗(Channel Shuffle for Group Convolutions)**

现代卷积神经网络会包含多个重复模块。其中，最先进的网络例如Xception和ResNeXt将有效的深度可分离卷积或分组卷积引入构建block中，在表示能力和计算消耗之间取得很好的折中。但是，我们注意到这两个设计都没有充分采用1×1的逐点卷积，因为这需要很大的计算复杂度。例如，在ResNeXt中3×3卷积配有分组卷积，逐点卷积占了93.4%的multiplication-adds。

在小型网络中，昂贵的逐点卷积造成有限的通道之间充满约束，这会显著的损失精度。为了解决这个问题，一个直接的方法是应用通道稀疏连接，例如组卷积(group convolutions)。通过确保每个卷积操作仅在对应的输入通道组上，组卷积可以显著的降低计算损失。然而，如果多个组卷积堆叠在一起，会有一个副作用： 某个通道输出仅从一小部分输入通道中导出，如下图(a)所示，这样的属性降低了通道组之间的信息流通，降低了信息表示能力。

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p1.png" />
</div>

如果我们允许组卷积能够得到不同组的输入数据，即上图(b)所示效果，那么输入和输出通道会是全关联的。具体来说，对于上一层输出的通道，我们可做一个混洗(Shuffle)操作，如上图（c）所示，再分成几个组，feed到下一层。

对于这个混洗操作，有一个有效高雅(efficiently and elegantly)的实现:

对于一个卷积层分为g gg组，

+ 1.有g×n个输出通道
+ 2.reshape为(g,n)
+ 3.再转置为(n,g)
+ 4.平坦化,再分回g组作为下一层的输入

示意图如下：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p2.png" />
</div>

这样操作有点在于是可微的，模型可以保持end-to-end训练.

**Shuffle Unit**

前面我们讲了通道混洗的好处了，在实际过程中我们构建了一个ShuffleNet unit，便于构建实际模型。

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p3.png" />
</div>

+ 图(a)是一个残差模块。对于主分支部分，我们可将其中标准卷积3×3拆分成深度分离卷积(MobileNet)。我们将第一个1×1卷积替换为逐点组卷积，再作通道混洗(即(b))。

+ 图(b)即ShuffleNet unit，主分支最后的1×1Conv改为1×1GConv，为了适配和恒等映射做通道融合。配合BN层和ReLU激活函数构成基本单元.

+ 图(c)即是做降采样的ShuffleNet unit，这主要做了两点修改：

    - 在辅分支加入步长为2的3×3平均池化
    - 原本做元素相加的操作转为了通道级联，这扩大了通道维度，增加的计算成本却很少
    - 归功于逐点群卷积和通道混洗，ShuffleNet unit可以高效的计算。相比于其他先进的单元，在相同设置下复杂度较低。

例如：给定输入大小h×w×c,通道数为c。bottleneck通道为m:

+ ResNet unit需要hw(2cm+9m^2)FLOPS计算量

+ ResNeXt需要hw(2cm+9m^2/g)FLOPS

+ 而ShuffleNet unit只需要hw(2cm/g+9m)FLOPS

其中g代表组卷积数目。即表示：给定一个计算限制，ShuffleNet可以使用更宽的特征映射。我们发现这对小型网络很重要，因为小型网络没有足够的通道传递信息。

**需要注意的是:**虽然深度卷积可以减少计算量和参数量，但在低功耗设备上，与密集的操作相比，计算/存储访问的效率更差。故在ShuffleNet上我们只在bottleneck上使用深度卷积，尽可能的减少开销.


**NetWork Architecture**

在上面的基本单元基础上，我们提出了ShuffleNet的整体架构：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p4.png" />
</div>

主要分为三个阶段：

+ 每个阶段的第一个block的步长为2，下一阶段的通道翻倍

+ 每个阶段内的除步长其他超参数保持不变

+ 每个ShuffleNet unit的bottleneck通道数为输出的1/4(和ResNet设置一致)

这里主要是给出一个baseline。在ShuffleNet Unit中，参数g控制逐点卷积的连接稀疏性(即分组数)，对于给定的限制下，越大的g会有越多的输出通道，这帮助我们编码信息。

定制模型需要满足指定的预算，我们可以简单的使用放缩因子s控制通道数，ShuffleNet s×即表示通道数放缩到s倍。

#### 4.Experiment


实验在ImageNet的分类集上做评估，大多数遵循ResNeXt的设置，除了两点：

+ 权重衰减从1e-4降低到了4e-5

+ 数据增强使用较少的aggressive scale 增强

这样做的原因是小型网络在训练过程通常会遇到欠拟合而不是过拟合问题。

**On the Importance of Pointwise Group Convolutions**

为了评估逐点卷积的重要性，比较相同复杂度下组数从1到8的ShuffleNet模型，同时我们通过放缩因子s控制网络宽度，扩展为3种：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p5.png" />
</div>

从结果来看，有组卷积的一致比没有组卷积(g=1)的效果要好。注意组卷积可允许获得更多通道的信息，我们假设性能的收益源于更宽的特征映射，这帮助我们编码更多信息。并且，较小的模型的特征映射通道更少，这意味着能多的从特征映射上获取收益。

表2还显示，对于一些模型，随着g增大，性能上有所下降。意味组数增加，每个卷积滤波器的输入通道越来越少，损害了模型表示能力。

值得注意的是，对于小型的ShuffleNet 0.25×，组数越大性能越好，这表明对于小模型更宽的特征映射更有效。受此启发，在原结构的阶段3删除两个单元，即表2中的`arch2`结构，放宽对应的特征映射，明显新的架构效果要好很多。

**Channel Shuffle vs. No Shuffle**

Shuffle操作是为了实现多个组之间信息交流，下表表现了有无Shuffle操作的性能差异：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p6.png" />
</div>

在三个不同复杂度下带Shuffle的都表现出更优异的性能，尤其是当组更大(arch2,g=8)，具有shuffle操作性能提升较多，这表现出Shuffle操作的重要性

**Comparison with Other Structure Units**

我们对比不同unit之间的性能差异，使用表1的结构，用各个unit控制阶段2-4之间的Shuffle unit，调整通道数保证复杂度类似。

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p7.png" />
</div>

可以看到ShuffleNet的表现是比较出色的。有趣的是，我们发现特征映射通道和精度之间是存在直观上的关系，以38MFLOPs为例，VGG-like, ResNet, ResNeXt, Xception-like, ShuffleNet模型在阶段4上的输出通道为50, 192, 192, 288, 576，这是和精度的变化趋势是一致的。我们可以在给定的预算中使用更多的通道，通常可以获得更好的性能。

上述的模型不包括GoogleNet或Inception结构，因为Inception涉及到太多超参数了，做为参考，我们采用了一个类似的轻量级网络PVANET。结果如下：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p8.png" />
</div>

ShuffleNet模型效果要好点.

**Comparison with MobileNets and Other Frameworks**

与MobileNet和其他模型相比：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p9.png" />
</div>

相比于不同深度的模型对比，可以看到我们的模型要比MobileNet的效果要好，这表明ShuffleNet的有效性主要来源于高效的结构设计，而不是深度。

**Generalization Ability**

我们在MS COCO目标检测任务上测试ShuffleNet的泛化和迁移学习能力，以Faster RCNN为例：

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p10.png" />
</div>

ShuffleNet的效果要比同等条件下的MobileNet效果要好，我们认为收益源于ShuffleNet的设计。

**Actual Speedup Evaluation**

评估ShuffleNet在ARM平台的移动设备上的推断速度。

<div align=center>
<img src="zh-cn/img/shufflenet/v1/p11.png" />
</div>

三种分辨率输入做测试，由于内存访问和其他开销，原本理论上4倍的加速降低到了2.6倍左右。

#### 5.Conclusion

论文针对现多数有效模型采用的逐点卷积存在的问题，提出了组卷积和通道混洗的处理方法，并在此基础上提出了一个ShuffleNet unit，后续对该单元做了一系列的实验验证，证明ShuffleNet的结构有效性。

#### 6.Code


layer:

```python
import tensorflow as tf
import numpy as np


############################################################################################################
# Convolution layer Methods
def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p('conv', x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o


def grouped_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                   initializer=tf.contrib.layers.xavier_initializer(), num_groups=1, l2_strength=0.0, bias=0.0,
                   activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
                   is_training=True):
    with tf.variable_scope(name) as scope:
        sz = x.get_shape()[3].value // num_groups
        conv_side_layers = [
            conv2d(name + "_" + str(i), x[:, :, :, i * sz:i * sz + sz], w, num_filters // num_groups, kernel_size,
                   padding,
                   stride,
                   initializer,
                   l2_strength, bias, activation=None,
                   batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=dropout_keep_prob,
                   is_training=is_training) for i in
            range(num_groups)]
        conv_g = tf.concat(conv_side_layers, axis=-1)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_g, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_g
            else:
                conv_a = activation(conv_g)

        return conv_a


def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_p(name='conv', x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a


############################################################################################################
# ShuffleNet unit methods

def shufflenet_unit(name, x, w=None, num_groups=1, group_conv_bottleneck=True, num_filters=16, stride=(1, 1),
                    l2_strength=0.0, bias=0.0, batchnorm_enabled=True, is_training=True, fusion='add'):
    # Paper parameters. If you want to change them feel free to pass them as method parameters.
    activation = tf.nn.relu

    with tf.variable_scope(name) as scope:
        residual = x
        bottleneck_filters = (num_filters // 4) if fusion == 'add' else (num_filters - residual.get_shape()[
            3].value) // 4

        if group_conv_bottleneck:
            bottleneck = grouped_conv2d('Gbottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                        padding='VALID',
                                        num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                        activation=activation,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = channel_shuffle('channel_shuffle', bottleneck, num_groups)
        else:
            bottleneck = conv2d('bottleneck', x=x, w=None, num_filters=bottleneck_filters, kernel_size=(1, 1),
                                padding='VALID', l2_strength=l2_strength, bias=bias, activation=activation,
                                batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            shuffled = bottleneck
        padded = tf.pad(shuffled, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        depthwise = depthwise_conv2d('depthwise', x=padded, w=None, stride=stride, l2_strength=l2_strength,
                                     padding='VALID', bias=bias,
                                     activation=None, batchnorm_enabled=batchnorm_enabled, is_training=is_training)
        if stride == (2, 2):
            residual_pooled = avg_pool_2d(residual, size=(3, 3), stride=stride, padding='SAME')
        else:
            residual_pooled = residual

        if fusion == 'concat':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters - residual.get_shape()[3].value,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(tf.concat([residual_pooled, group_conv1x1], axis=-1))
        elif fusion == 'add':
            group_conv1x1 = grouped_conv2d('Gconv1x1', x=depthwise, w=None,
                                           num_filters=num_filters,
                                           kernel_size=(1, 1),
                                           padding='VALID',
                                           num_groups=num_groups, l2_strength=l2_strength, bias=bias,
                                           activation=None,
                                           batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            residual_match = residual_pooled
            # This is used if the number of filters of the residual block is different from that
            # of the group convolution.
            if num_filters != residual_pooled.get_shape()[3].value:
                residual_match = conv2d('residual_match', x=residual_pooled, w=None, num_filters=num_filters,
                                        kernel_size=(1, 1),
                                        padding='VALID', l2_strength=l2_strength, bias=bias, activation=None,
                                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
            return activation(group_conv1x1 + residual_match)
        else:
            raise ValueError("Specify whether the fusion is \'concat\' or \'add\'")


def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name) as scope:
        n, h, w, c = x.shape.as_list()
        x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output


############################################################################################################
# Fully Connected layer Methods

def __dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
              bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        __variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output


def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_p(name='dense', x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,
                              bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training, epsilon=1e-5)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o


############################################################################################################
# Pooling Methods

def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling', padding='VALID'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding=padding,
                          name=name)


############################################################################################################
# Utilities for layers

def __variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w


# Summaries for variables
def __variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
```

model:

```python
import tensorflow as tf
from layers import shufflenet_unit, conv2d, max_pool_2d, avg_pool_2d, dense, flatten


class ShuffleNet:
    """ShuffleNet is implemented here!"""
    MEAN = [103.94, 116.78, 123.68]
    NORMALIZER = 0.017

    def __init__(self, args):
        self.args = args
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None

        # A number stands for the num_groups
        # Output channels for conv1 layer
        self.output_channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960], '4': [272, 544, 1088],
                                '8': [384, 768, 1536], 'conv1': 24}

        self.__build()

    def __init_input(self):
        batch_size = self.args.batch_size if self.args.train_or_test == 'train' else 1
        with tf.variable_scope('input'):
            # Input images
            self.X = tf.placeholder(tf.float32,
                                    [batch_size, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            # Classification supervision, it's an argmax. Feel free to change it to one-hot,
            # but don't forget to change the loss from sparse as well
            self.y = tf.placeholder(tf.int32, [batch_size])
            # is_training is for batch normalization and dropout, if they exist
            self.is_training = tf.placeholder(tf.bool)

    def __resize(self, x):
        return tf.image.resize_bicubic(x, [224, 224])

    def __stage(self, x, stage=2, repeat=3):
        if 2 <= stage <= 4:
            stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None,
                                          num_groups=self.args.num_groups,
                                          group_conv_bottleneck=not (stage == 2),
                                          num_filters=
                                          self.output_channels[str(self.args.num_groups)][
                                              stage - 2],
                                          stride=(2, 2),
                                          fusion='concat', l2_strength=self.args.l2_strength,
                                          bias=self.args.bias,
                                          batchnorm_enabled=self.args.batchnorm_enabled,
                                          is_training=self.is_training)
            for i in range(1, repeat + 1):
                stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i),
                                              x=stage_layer, w=None,
                                              num_groups=self.args.num_groups,
                                              group_conv_bottleneck=True,
                                              num_filters=self.output_channels[
                                                  str(self.args.num_groups)][stage - 2],
                                              stride=(1, 1),
                                              fusion='add',
                                              l2_strength=self.args.l2_strength,
                                              bias=self.args.bias,
                                              batchnorm_enabled=self.args.batchnorm_enabled,
                                              is_training=self.is_training)
            return stage_layer
        else:
            raise ValueError("Stage should be from 2 -> 4")

    def __init_output(self):
        with tf.variable_scope('output'):
            # Losses
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y, name='loss'))
            self.loss = self.regularization_loss + self.cross_entropy_loss

            # Optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)
                # This is for debugging NaNs. Check TensorFlow documentation.
                self.check_op = tf.add_check_numerics_ops()

            # Output and Metrics
            self.y_out_softmax = tf.nn.softmax(self.logits)
            self.y_out_argmax = tf.argmax(self.y_out_softmax, axis=-1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))

        with tf.name_scope('train-summary-per-iteration'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.accuracy)
            self.summaries_merged = tf.summary.merge_all()

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()

        with tf.name_scope('Preprocessing'):
            red, green, blue = tf.split(self.X, num_or_size_splits=3, axis=3)
            preprocessed_input = tf.concat([
                tf.subtract(blue, ShuffleNet.MEAN[0]) * ShuffleNet.NORMALIZER,
                tf.subtract(green, ShuffleNet.MEAN[1]) * ShuffleNet.NORMALIZER,
                tf.subtract(red, ShuffleNet.MEAN[2]) * ShuffleNet.NORMALIZER,
            ], 3)
        x_padded = tf.pad(preprocessed_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        conv1 = conv2d('conv1', x=x_padded, w=None, num_filters=self.output_channels['conv1'], kernel_size=(3, 3),
                       stride=(2, 2), l2_strength=self.args.l2_strength, bias=self.args.bias,
                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training,
                       activation=tf.nn.relu, padding='VALID')
        padded = tf.pad(conv1, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
        max_pool = max_pool_2d(padded, size=(3, 3), stride=(2, 2), name='max_pool')
        stage2 = self.__stage(max_pool, stage=2, repeat=3)
        stage3 = self.__stage(stage2, stage=3, repeat=7)
        stage4 = self.__stage(stage3, stage=4, repeat=3)
        global_pool = avg_pool_2d(stage4, size=(7, 7), stride=(1, 1), name='global_pool', padding='VALID')

        logits_unflattened = conv2d('fc', global_pool, w=None, num_filters=self.args.num_classes,
                                    kernel_size=(1, 1),
                                    l2_strength=self.args.l2_strength,
                                    bias=self.args.bias,
                                    is_training=self.is_training)
        self.logits = flatten(logits_unflattened)

        self.__init_output()

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
```
*来源：https://github.com/MG2033/ShuffleNet.git*


### 3.2 (ShuffleNet V2)ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

#### 0.摘要

目前，神经网络架构的设计的评估都是依托于不直接的FLOPs(每秒浮点运算次数)来评价计算的复杂度，本文利用直观的评价例如，运行速度，内存消耗，平台特性等，此外论文还提出了新的模型框架：ShuffleNet V2。

#### 1.介绍

模型的评估中除了accuracy还有计算的复杂度，现实任务中是在有限的计算资源的条件下寻找尽可能准确的模型，这激发了一些模型的产生比如： Xception,MobileNet,MobileNet v2,ShuffleNet,CondenseNet,这些网络大部分都使用了分组卷积和深度可分卷积。

FLOPs是间接的评价指标，它只是近似而不能等价于一些直接的评价比如：速度和延迟，如下图显示，仅使用FLOPs是不能有效的衡量计算复杂度的。

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p1.png" />
</div>


FLOPs和speed的区别主要如下：第一： MAC(memory access cost)内存访问成本，在分组卷积计算中MAC的成本是非常大的，对于一些强的计算设备：比如GPU，这仍然是一个瓶颈，，这个因素在模型设计中不能不被考虑。另一个是并行化在相同FLOPs下谁的并行化更强谁的计算将变的更快。

第二是平台，相同的FLOPs运算速度却不一样，比如最新版的CUDNN优化了3X3conv,我们不能直观的就人为3X3卷积要比1X1卷积慢9倍。

综上所述，两个主要的影响网络架构设计的指标必须考虑： 首先,直接的评价（比如：速度），间接的评价（比如FLOPs)，其次,这些评价必须指明运行的平台。基于这些准则，论文设计了ShuffleNet V2。section 2: 给出了几个模型结构评价的guidelines，这些评价依托于不同的平台（GPU和ARM），section 3: ShuffleNet V2的架构； section 4: 提供了一些实验结果，section 5:结论。

#### 2.Practical Guidelines for Efficient Network Design

我们的模型比大部分开源的模型效率要高，

- GPU. NVIDIA GeForce GTX 1080 Ti,CUDNN 7.0 

- ARM. Qualcomm Snapdragon 810.


**G1)卷积层的输入输出特征通道数对MAC(memory access cost，内存访问消耗时间)的影响**

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p2.png" />
</div>

**结论1**：卷积层的输入和输出特征通道数相等时，MAC最小，此时模型速度最快。

**G2)卷积的group操作对MAC的影响**

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p3.png" />
</div>

**​​​​​​​结论2**：过多的group操作会增大MAC，从而使模型速度变慢。


**G3)模型设计的分支数量对模型速度的影响**

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p4.png" />
</div>

​​​​​​​**结论3**：模型中的分支数量越少，模型速度越快。

**G4)element-wise(逐点点积)操作对模型速度的影响**

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p5.png" />
</div>

**结论4**：element-wise操作所带来的时间消耗远比在FLOPs上的体现的数值要多，因此要尽可能减少element-wise操作。depthwise convolution具有低FLOPs、高MAC的特点。

**结论**

综上所述，高效的网络架构应该，1）使用"balance" conv;2) Be a ware of the cost of using group convs;3)Reduce the degree of fragmentation; 4)Reduce the element-wise operations.

#### 3.ShuffleNet V2: an Efficient Architecture

**ShuffleNet v1和ShuffleNet v2的构建块结构对比**

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p6.png" />
</div>

+ 结论1：增加了一个channel split操作；

+ 结论2：取消了1X1卷积层中的group操作；

+ 结论3：channel shuffle的操作移到了concat后面；

+ 结论4：将element-wise add操作替换成concat。

**channel split操作**

1.feature channels的个数是c，分割成两个batch,c-c1和c1(c1=0.5或1)两个channel batch(G3)

2.一个channel batch 用来做shortcut的identity操作，另一个channel batch 用来做另一个分支的卷积操作（G1,G2)

3.经过卷积操作后,再concat在一起，保持了channel个数的不变（G1)

4.经过shuffling后进入下一个unit.

**ShuffleNet v2的具体网络结构示意图**

每个stage都是由Fig.3(c)(d)所示的构建块组成，构建块的具体数量对应下表中的Repeat列

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p7.png" />
</div>

#### 4.一些模型在速度、精度、FLOPs上的详细对比

实验中不少结果都和前面的实验结论吻合，比如MobileNet v1速度较快，主要原因是因为简单的网络结构，没有太多复杂的支路结构。

<div align=center>
<img src="zh-cn/img/shufflenet/v2/p8.png" />
</div>

#### 5.code

utils.py:

```python
import os
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x
```

shufflenetv2.py

```python
import numpy as np
from keras.utils import plot_model
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dense
from keras.models import Model
import keras.backend as K
from utils import block


def ShuffleNetV2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(224,224,3),
                 load_model=None,
                 num_shuffle_units=[3,7,3],
                 bottleneck_ratio=1,
                 classes=1000):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                      data_format=K.image_data_format())
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ShuffleNetV2(include_top=True, input_shape=(224, 224, 3), bottleneck_ratio=1)
    plot_model(model, to_file='shufflenetv2.png', show_layer_names=True, show_shapes=True)


    pass
```
*来源：https://github.com/opconty/keras-shufflenetV2.git*

------

### 4.EffNet: An Efficient Structure for Convolutional Neural Networks

#### 0.摘要

随着CNN在用户产品应用上不断的增加，对模型能够高效运行在嵌入式，移动设备上的需求逐渐热化。模型轻量化因此变成一个热门研究话题，也涌现出如网络二值化，改进卷积层设计等方向的研究。本文研究的方式是后者，即提出一种新颖的卷积设计，能够显著的减轻计算负担，且性能远胜当前最好的模型(对比MobileNet,ShuffleNet),该模型称为EffNet.

*关键词：CNN,计算高效，实时inference*

#### 1.介绍

近些年出现许多用于在小的硬件平台上做实时推断的方法研究例如对训练后的网络进行剪枝，还有一种是将32bit的网络值转化为二值化模型；最新的方法是关注神经元的互联性和普通卷积层的自然特性。

3X3的卷积设计目前作为标准的卷积结构，称为很多优化方案的选择。MobileNet和ShuffleNet的做法是在4维tensor的不同维度上对计算进行分离，从而解决计算量大的问题，但这两个网络仍然存在未解决的问题：第一个问题是两个模型在大型网络上表现突出，能使迷行减小以及更加高效，然而在小的网络结构中结果却很一般。第二个问题是，两个模型结构对流入网络的数据会产生很大的瓶颈，这种瓶颈在高度冗余的网络中可能被认为是无关紧要的，但如我们实验显示，他可能会对小的模型产生影响。而我们的模型设计的网络能够部署大的网络结构到低容量的硬件，也能够增强现有模型的效率。

#### 2.相关工作

**超参数优化**，提出一种CNN和SVM的贝叶斯优化框架，通过最大化增加模型精度的概率。这种方法需要初始化合适，且受限于搜索空间。基于增强学习的LSTM来优化超参数来改进速度和精度，这类方法不受搜索空间限制，但需要额外步骤。

另一类方法是对模型进行后处理，即**模型剪枝**，基于精度损失代价最小化的修剪算法。这类方法的问题是：开发的pipeline需要额外的阶段用于特定超参数的优化。此外，当网络结构发生变化，模型需要额外的fine-tuning.

另外一种模型后处理压缩的方式是固定点模型参数量化，即数值精度小于常规的32bit float型，或者是直接**二值化网络**。这类方法的问题是：尽管模型推断快了，但对比它的baseline，精度下降了较多，因此这类方法的诉求性不高。

最近与本文相似的工作，都探寻了常规卷积操作的本质。其中包含维度级的卷积操作分离。相比原始操作，FLOPs显著减少。Rethink Inception paper将3x3卷积分离成3x1和1x3,.MobileNet做了进一步延伸，将分离用在了通道级，显著减少了FLOPs，主要计算量转到了pointwise层。最后ShuffleNet通过同ReNeXt类似的方法将pointwise层进行分组来处理pointwise层的FLOPs装载，是的精度损失相当小。

#### 3.构建块来增加模型效率

构造了一种通用的EffNet block这种合适的解决方案形式。

<div align=center>
<img src="zh-cn/img/effnet/p1.png" />
</div>

 首先,将MoblieNet的3×3的depthwise convolution层分解为两个3×1,1×3depthwise convolution层，这样便可以在第一层之后就采用pool操作,从而减少第二层的计算量. 
 如上图所示,在第一个卷积层之后,使用2X1 max pooling操作，在第二个卷积层之后,用2×1,stride=1的卷积核代替1×1pointwise convolution，这样计算量相同,但是可以有更高的精度.

**Bottleneck 结构**

对输入通道缩减系数8倍（即baseline中最原始输入是32通道，最后输出的256通道）。一个ShuffleNet块使用的缩减系数是4。而窄的模型没有足够的通道做大幅减少。我们实验证明精度的损失与较适度的缩减成比例。因此，我们提出瓶颈系数用2,此外我们发现使用空间卷积是有效的（看下图），其深度乘法器为2，第一个depthwise卷积层数量也翻了一番。

<div align=center>
<img src="zh-cn/img/effnet/p2.png" />
</div>


**Strides 和 Pooling**

MobileNet和ShuffleNet的模块中depthwise空间卷积层采用的步长为2, 我们实验表明这种操作有两个问题。第一个问题：我们多次实验证明与最大池化相比，stride=2有精度下降。

此外，给空间卷积层添加最大池化不允许在它缩减到输入尺寸4分之1前给数据进行编码。然而，越早的最大池化意味后续计算开销更小。为了维持早些最大池化的优势，且避免数据压缩过严重，我们提出采用可分离的池化（separable pooling）。同可分离卷积类似，首先在第一次dw 1x3的卷积后用2x1的池化核（相应的stride,我的理解是2x1，其中的1d mp指的是一维的mp操作，即1x2或2x1，而非2x2），然后是3x1的depthwise卷积，接下来是2x1xch的卷积来替代1x1xch的pointwise卷积，stride是1d的，这么做的好处是应用了池化可以减少计算量，且由避免数据压缩严重，精度下降更少。


**Separable Convolutions**

用连续的3x1和1x3替代3x3卷积的思想，并结合我们的可分离池化卷积核的思想。

**Residual Connnections**

本文分析了使用这个连接对小网络中高压缩率精度损失的影响。

**Group Convolutions**

本文未使用，考虑精度损失的问题，尽管它在减少计算量上有很大优势。

**Addressing the First Layer**

MobileNet和ShuffleNet都未对第一个层做替换更改，他们认为第一个层产生的计算代价不高。我们认为每个地方的优化都有意义，在优化完其他层后，发现第一层显得有点大，我们将第一层替换成EffNet block,结果减少了约30%的计算量。

#### 4.EffNet 模型

**4.1 数据压缩**

我们构建了个对数据压缩比较明显的网络结构来做实验。实验发现大的bottleneck对精度影响比较大，Table 1列出了在Cifar10上网络的数据维度信息。

**4.2 EffNet Block**

我们设计的efficient 卷积block作为基本模块来替代之前的普通卷积层，且不再局限于slim网络。将3x3的depthwise卷积分割成两个线性层，1x3和3x1. 在对第一个空间层后进行池化，为第二层减少了计算量。见Table1和图1，在第一个depthwise卷积后应用2X1最大池化核，在第二次下采样时，我们使用2x1的卷积核来替代常规的pointwise，该步操作的FLOPs可能相当，但得到的精确度更好些。

#### 5.实验

<div align=center>
<img src="zh-cn/img/effnet/p3.png" />
</div>

<div align=center>
<img src="zh-cn/img/effnet/p4.png" />
</div>

<div align=center>
<img src="zh-cn/img/effnet/p5.png" />
</div>

<div align=center>
<img src="zh-cn/img/effnet/p6.png" />
</div>

<div align=center>
<img src="zh-cn/img/effnet/p7.png" />
</div>


#### code

```python
from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *


def get_post(x_in):
    x = LeakyReLU()(x_in)
    x = BatchNormalization()(x)
    return x

def get_block(x_in, ch_in, ch_out):
    x = Conv2D(ch_in,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False)(x_in)
    x = get_post(x)

    x = DepthwiseConv2D(kernel_size=(1, 3), padding='same', use_bias=False)(x)
    x = get_post(x)
    x = MaxPool2D(pool_size=(2, 1),
                  strides=(2, 1))(x) # Separable pooling

    x = DepthwiseConv2D(kernel_size=(3, 1),
                        padding='same',
                        use_bias=False)(x)
    x = get_post(x)

    x = Conv2D(ch_out,
               kernel_size=(2, 1),
               strides=(1, 2),
               padding='same',
               use_bias=False)(x)
    x = get_post(x)

    return x


def Effnet(input_shape, nb_classes, include_top=True, weights=None):
    x_in = Input(shape=input_shape)

    x = get_block(x_in, 32, 64)
    x = get_block(x, 64, 128)
    x = get_block(x, 128, 256)

    if include_top:
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model
```

*来源：https://github.com/arthurdouillard/keras-effnet.git*

------



### 5.SqueezeNet：AlexNet-Level Accuracy with 50x Fewer Paramenters and <0.5MB Model Size


#### 0.摘要

深卷积神经网络 (CNNs) 最近的研究主要集中在提高精度。对于给定的精度级别, 通常可以确定多个 CNN 体系结构, 以达到该精度级别。同样的精度, 较小的 CNN 架构提供至少三优点: 

(1) 较小的 CNNs 在分布式训练过程中需要跨服务器进行更少的通信。 

(2) 较小的 CNNs 需要更少的带宽, 将一个新的模型从云端导出到自动汽车。

(3) 较小的 CNNs 提供了在 fpga 和其他内存有限的硬件上部署的可行性。 

综上， 我们提出了一个叫做 SqueezeNet 的小型 CNN 架构。SqueezeNet 在 ImageNet 上达到 AlexNet 级精度, 参数减少50x(即参数减少50倍，下同)。此外, 使用模型压缩技术, 我们可以将 SqueezeNet 压缩到小于 0.5MB (比AlexNet小510×). 

SqueezeNet 体系结构可在此处下载: https://github.com/DeepScale/SqueezeNet


#### 1.介绍

最近对深卷积神经网络 (CNNs) 的研究主要集中在提高计算机视觉数据集的精确度上。对于给定的精度级别, 通常可以用不同的 CNN 体系结构来实现了该精度级别。而具有更少参数的 CNN 体系结构具有以下几个优点:

+ **更高效的分布式训练**。服务器之间的通信是分布式 CNN 训练的可扩展性的重要限制因素。对于分布式数据并行训练, 通信开销与模型中的参数数目成正比 (Iandola 等, 2016)。简而言之, 小模型由于需要较少的通信量而得以快速地训练。

+ **向客户端导出新模型时的开销更小**。为了优化自主驾驶技术, 特斯拉等公司定期将新的训练模型从服务器复制到客户的车上。此做法通常称为远程更新。消费者报告发现, 特斯拉的自动驾驶仪半自主驱动功能的安全性随着最近的更新而逐步提高 (消费者报告, 2016)。然而, 今天典型的 CNN/DNN 模型频繁的远程更新可能需要大量的数据传输。与 AlexNet相比, 这将需要从云服务器传输240MB的数据量到汽车上。较小的模型只需要传输较少的数据量, 这使得频繁更新变得更加可行。

+ **可行的 FPGA 和嵌入式部署**。 fpga 通常有小于 10MB的 onchip 内存, 并且不存在外部存储。因此, 一个足够小的模型可以直接存储在 fpga 上, 而不是被内存带宽所限制 (邱等, 2016), 而视频帧流通过 FPGA 实时。此外, 当在特定于应用程序的集成电路 (ASICs) 上部署 CNNs 时, 一个足够小的模型可以直接存储在芯片上, 而较小的模型可以使 ASIC 适合于较小的模型。例如, Xilinx Vertex-7 FPGA 的芯片内存最大为 8.5 mb (即 68 Mbits), 不提供外部存储空间。

较小的 CNN 架构有太多的优点。考虑到这一点, 我们的研究目标聚焦于在准确度不下降的情况下设计一个需要更少参数的CNN网络结构，我们发现了这样的一个体系结构, 我们称之为SqueezeNet。此外, 我们提出了一个更具有流程化的方法来搜索新的 CNN 架构的设计空间。

论文的其余部分按如下方式组织。在第2部分中, 我们回顾相关的工作。然后, 在3和4节中, 我们描述和评估 SqueezeNet 体系结构。之后, 我们将注意力转向了解 CNN 结构设计方案如何影响模型的大小和准确性。我们通过探索 SqueezeNet 体系结构的设计空间来获得这种理解。在5节中, 我们设计了CNN 微体系结构的空间探索, 我们将其定义为各个层和模块的组织和维度。在6节中, 我们对CNN 宏观体系结构进行了设计空间探索, 我们将其定义为 CNN 中层次结构的高层组织。最后, 我们在7节总结。简而言之, 3 和4节对于 CNN 研究人员以及只想将 SqueezeNet 应用于新应用程序的从业者都很有用。其余部分针对的是打算设计自己的 CNN 架构的高级研究人员。

#### 2.相关工作

**2.1 模型压缩**

我们工作的首要目标是确定一个模型, 它的参数很少, 同时保持精度。为了解决这个问题, 可行的做法是采取现有的 CNN 模型, 并以有损的方式压缩它。事实上, 在一些研究团体已经开始围绕着这个思路进行了探索，并发表了几种方法。一个由Denton et al提出的相当简单的方法,是将奇异值分解 (SVD) 应用于预训练CNN模型 (2014)。Han et发名了一种网络修剪方法，即从一个预训练模型开始, 然后用零替换低于某个阈值的参数, 形成一个稀疏矩阵, 最后在稀疏 CNN 上执行一些训练迭代 (han等, 2015b)。最近, han et通过将网络修剪与量化 (8 比特或更少) 和霍夫曼编码结合在一起, 以创建一种称为深压缩的方法 (EIE, 2015a), 并进一步设计了一个名为EIE的硬件加速器，它直接运行在压缩的模型之上, 实现了提高模型运行速度和节省大量运算资源的效果。


**2.2 CNN 微结构**

从LeCun et al.在二十世纪八十年代下旬推广CNNs 数字识别应用以来 (LeCun 等, 1989)，卷积网络已在人工神经网络中使用了至少25年在神经网络中, 卷积过滤器通常有3个维度, 以高度、宽度和通道为关键尺寸。当应用到图像时, CNN 过滤器通常在第一层 中有3个通道(即 RGB) , 并且在随后的每一层中,过滤器的通道数与 Li −1 CNN卷积的个数相同。早期的工作由 LeCun et al。(LeCun et 等, 1989) 使用 5x5xChannels[1] 卷积, 最近的 VGG (Simonyan & Zisserman, 2014) 体系结构广泛使用3x3卷积模型, 如NIN和 GoogLeNet 在某些层中使用1x1 卷积。

而在CNNs网络结构越来越深的大趋势下, 手动选择每个层的过滤尺寸变得很麻烦。为了解决这一问题, 提出了由多个卷积层组成的各种更高级别的构建块或模块。例如, GoogLeNet 文件提出了初始化模块, 它由许多不同的卷积组成, 通常包括1x1 和 3x3, 加上有时 5x5 (Szegedy et, 2014), 有时1x3 和 3x1 (Szegedy 等)。2015). 然后将许多这样的模块组合起来, 也许还有其他的(ad-hoc)层来组成一个完整的网络. 我们使用CNN 微体系结构【CNN microarchitecture】一词来引用各个模块的特定组织和维度。

**2.3 CNN宏观结构**

虽然微体系结构是指单个层和模块, 但我们将CNN macroarchitecture定义为多个模块的系统级组织, 使其成为一个端到端的 CNN 架构。

也许在最近的文献中广泛研究的 CNN macroarchitecture 主题是网络中深度(即层数) 的影响。Simoyan 和 Zisserman 提出了层数大概在12到19层的VGG (Simonyan & Zisserman, 2014) 网络, 并在论文中指出, 更深的网络在ImageNet-1k 数据集上回获得更高的准确性 (Den等, 2009)。k. He et. 提出了更深层次的 CNNs ,网络层数多达30层, 同样在ImageNet数据集上也获得了更高的精度 (He等, 2015a)。

跨多个层或模块的连接选择是 CNN macroarchitectural 研究的一个新兴领域。残差网络 (ResNet) (He等, 2015b) 和HighwayNetworks (Srivastava等, 2015)都建议使用跳过多个层的连接, 例如将第3层的激活层连接到第6层的激活层上。我们将这些连接称为旁路连接。对于一个34层的CNN网络结构，ResNet 的作者提供了有旁路连接和没有旁路连接的比较，添加旁路连接使Top-5 ImageNet数据集上的准确度增加了2个百分点。


**2.4 网略设计空间探索**

神经网络 (包括深度网络和卷积神经网络) 具有很大的设计空间, 比如说microarchitectures、macroarchitectures的设计和其他超参数的选择。自然而然地, 很多研究机构希望凭借直觉得到关于这些因素如何影响神经网络的准确性 (即设计空间的形状)。神经网络设计空间探索的大部分工作都侧重于开发自动化的方法, 以找到更高精度的神经网络体系结构。这些自动化方法包括贝叶斯优化 (Snoek et, 2012), 模拟退火 (Ludermir 等, 2006), 随机搜索 (Bergstra & Bengio, 2012) 和遗传算法 (Stanley & Miikkulainen, 2002)。值得赞扬的是，每一篇论文都提供了一个案例，在这个案例中，提出的DSE方法产生了一个NN体系结构，与一个具有代表性的基础神经网络相比，它的精确度的确更高。然而, 这些论文并没有试图提供关于神经网络设计空间形状的直觉。在本文的后面, 我们避开了自动化的方法-相反, 我们通过重构 CNNs 的方式, 这样就就可以做A/B的比较, 从而可以探索出CNN 架构是如何影响模型的大小和准确性的。

在下面的章节中, 我们首先提出和评估了 SqueezeNet 网络结构, 并没有模型压缩。然后, 我们探讨了微体系结构和 宏观体系结构中的设计选择对 SqueezeNet 型 CNN 架构的影响。

#### 3 SQUEEZENET: 使用少量参数保持精度
在本节中, 我们首先概述了 CNN 体系结构的设计策略, 这些参数很少。然后, 我们介绍Fire模块这个我们的新构建块, 以构建整个CNN网络结构。最后, 我们使用我们的设计策略来构造SqueezeNet, 它主要由Fire模块组成。

**3.1 结构设计策略**

本文的首要目标是确定在保持准确性的同时, 有几个参数的 CNN 架构。为了实现这一点, 我们在设计 CNN 架构时采用了三个主要策略:

**策略 1**. 用1x1 滤镜替换3x3 滤镜。考虑到一定数量的卷积的预算, 我们将选择大量使用1x1卷积, 因为1x1 卷积的参数比3x3 过滤器少了 9X.

**策略 2**. 减少3x3 卷积输入通道的数量。假设有一个卷积层, 它完全由3x3 卷积组成。此层中参数的总数量为：(输入通道数) * (过滤器数) * (3 * 3)。因此, 为了在 CNN 中得到更少的参数, 不仅要减少3x3 过滤器的数量 (参见上面的策略 1), 还要减少3x3 卷积中输入通道的数量。我们使用squeeze层将输入通道的数量减少, 在下一节中我们将对其进行描述。

**策略 3**. 在网络中延迟下采样的时间, 以便卷积层具有较大的特征图。在卷积网络中, 每个卷积层输出一个特征图, 这些特征图的高度和宽度由以下内容控制: (1) 输入数据的大小 (如256x256 图像) 和 (2)在CNN 体系结构中缩减像素采样的层的选择。最常见的情况是, 下采样通过在某些卷积或池层中设置 ( > 1) 在 CNN 体系结构中进行设计 (例如 (Szegedy 等), 2014;Simonyan & Zisserman, 2014;Krizhevsky 等, 2012))。如果前边在网络中有很大的步长, 那么大多数图特征入将有小的激活映射。 反之, 如果网络中的前边的层都有1的步长, 并且超过1的步长集中在网络的后半部分 , 则网络中的许多层将具有大的激特征图。**我们的直觉是, 在其他不变的情况下，大的特征图 (由延迟下采样产生) 可以导致更高的分类精度**。的确, K.He和 h. Sun 将延迟下采样率应用到四种不同的 CNN 体系结构中, 在每种情况下, 延迟下采样都会导致分类精度变高 (He& Sun, 2015).【这里所说的下采样应该就是指池化】

策略1和2是关于在尽可能保持模型准确度地情况下减少 CNN 的参数数量,。策略3是关于在有限的参数数量下最大化精度。接下来, 我们描述的Fire模块, 将使我们能够成功地使用战略 1, 2 和3。

**3.2 Fire Model**

我们定义Fire模块如下。一个Fire模块包括: 一个squeeze层 (只有1x1 卷积), 将其放入一个具有1x1 和3x3 卷积组合的expand层中（图1）。在Fire模块中随意使用1x1 过滤器是应用3.1节中的策略1。在一个Fire模块中有三个超参数: s1x1, e1x1和 e3x3。在Fire模块中, s1x1 是squeeze层 (所有 1x1) 中的过滤器数, e1x1是1x1 卷积在expand层的数量, e3x3 3x3卷积在expand层的数量,。当我们使用Fire模块时, 我们设置 s1x1 小于 (e1x1 e + 3x3 ), 因此, expand层有助于限制3x3卷积中输入通道的数量即3.1节中的策略 2。 

<div align=center>
<img src="zh-cn/img/squeezenet/p1.png" />
</div>

*图 1: Microarchitectural 视图: 在Fire模块中组织卷积结构。在这个例子中, s1x1 = 3, e1x1 4, e 3x3 = 4。我们只展示卷积, 并没有展示激活层.*

+ 1.squeeze conv layer使用1X1卷积filter即策略1

+ 2.expand layer使用1X1和3X3filter的组合

+ 3.Fire Module使用3个可调用的超参数，s1X1（squeeze conv layer中的1X1filter个数）e1X1（expand layer 中的1X1filter个数) ，e3X3（expand layer中3X3filter的个数)

+ 4.Fire module过程中，令s1X1 < e1X1 + e3X3，这样squeeze可以限制输入通道数量即策略2


**3.3 SQUEEZENET 体系结构**

我们现在描述了 SqueezeNet CNN 的架构。我们在图2中说明了 SqueezeNet 从一个独立的卷积层 (conv1) 开始, 后跟8个Fire模块 (fire2-9), 最后 conv 层 (conv10) 结束。从开始到网络的末端，我们逐渐增加每个Fire模块的卷积的数量。

squeezeNet 在层 conv1、fire4、fire8 和 conv10 之后执行最大池化, 其步长为 2;这些相对较晚地执行池化操作是在执行3.1节的策略3。我们在表1中展示了完整的 SqueezeNet 体系结构。

<div align=center>
<img src="zh-cn/img/squeezenet/p2.png" />
</div>

每个Fire module中的filter数量逐渐增加，并在conv1,fire4,fire8和conv10这几层之后使用步长为2的max-pooling即将池化放在相对靠后的位置（策略3）

**3.3.1 其他SQUEEZENET 细节**

为了简洁起见, 我们省略了表1和图2中有关 SqueezeNet 的详细信息和设计选项的数量。我们提供以下这些设计选择。这些选择背后的直觉可以在下面引用的论文中找到。 

（1）为了使 1x1 和 3x3 filter输出的结果有相同的尺寸，在expand modules中，给3x3 filter的原始输入添加一个像素的边界（zero-padding）。 

（2）squeeze 和 expand layers中都是用ReLU作为激活函数 

（3）在fire9 module之后，使用Dropout，比例取50% 

（4）注意到SqueezeNet中没有全连接层，这借鉴了Network in network的思想 

（5）训练过程中，初始学习率设置为0.04，在训练过程中线性降低学习率。更多的细节参见本项目在github中的配置文件。 

（6）由于Caffee中不支持使用两个不同尺寸的filter，在expand layer中实际上是使用了两个单独的卷积层（1x1 filter 和 3x3 filter），最后将这两层的输出连接在一起，这在数值上等价于使用单层但是包含两个不同尺寸的filter。 
在github上还有SqueezeNet在其他框架下的实现：MXNet、Chainer、Keras、Torch。


<div align=center>
<img src="zh-cn/img/squeezenet/p3.png" />
</div>

*表 1: SqueezeNet 的体系结构尺寸。(此表的格式来自Inception2论文 )*

参数计算方法，以fire2为例：

+ max-pool/2 输出为55x55x96

+ fire2: 16个1x1x96 filter,之后将输出分别送入expand层中的1x1x16(64个)，3x3x16(64个)进行处理，对3X3进行padding,为了使长宽相同，这样就得到55x55x64大小的feature map.

+ 将这两个feature map链接到一起，得到55x55x128大小的feature map

参数计算：（1x1x96+1)x16+(1x1x16+1)X64+(3x3x16+1)x64=11920个


#### 4.评估SqueezeNet

接下来，我们对SqueezeNet网络进行一个评估 。在2.1 节中提到的 CNN 模型压缩论文中, 目标是压缩一个 AlexNet 网络, 它使用 ImageNet (Deng et al.2009) (ILSVRC 2012) 数据集训练后可以对图像进行分类。因此, 在评估 SqueezeNet 时, 我们使用 AlexNet[4] 和相关的模型压缩结果作为比较的基准。

在表2中, 我们将最近的模型压缩结果 和SqueezeNet网络做一个对比。可以看到：SVD 方法可以将 AlexNet 模型压缩为以前的5x, 同时会使top-1 数据集上的精度降低到 56.0% (丹顿 et, 2014)。网络修剪实现了模型尺寸的9x 降低, 同时保持了 top-1数据集上 57.2%的精度和top-5数据集上 80.3% 的精度 (Han等人, 2015b)。深压缩达到35x 的模型尺寸压缩比率, 同时仍然保持以往的精度 (Han等, 2015a)。现在, 使用 SqueezeNet网络, 我们实现了减少50X 的模型压缩比率, 同时满足或超过 AlexNet 的 top-1 和 top-5 的准确性。

我们似乎已经超过了目前模型压缩所取得的最新成果: 即使使用未压缩的32位值来表示模型, SqueezeNet 也在保持或超过原本正确率的基础上有一个1.4× 的模型压缩比，这相比于目前模型压缩所取得的最新成果还要好一些。

同时，直到现在, 一个开放的问题是:是易于压缩的小模型, 还是小模型 “需要” 由稠密浮点值提供的所有代表力？为了找出, 我们应用了深压缩 (韩等, 2015a) 表 2: 比较 SqueezeNet 模型压缩方法。通过模型大小, 我们的意思是存储经过培训的模型中所有参数所需的字节数。

SqueezeNet, 使用33% 稀疏[5] 和8位量化. 这将生成一个 0.66 MB 的模型 (363× 小于32位 AlexNet), 并具有与 AlexNet 等效的精度。此外, 在 SqueezeNet 上应用6位量化和33% 稀疏度的深压缩, 我们生成一个0.47MB 模型 (510× 小于32位 AlexNet), 具有等效的精度。我们的小模型确实可以压缩。

此外, 这些结果表明, 深压缩 (韩等, 2015a) 不仅在 CNN 的体系结构具有许多参数 (如 AlexNet 和 VGG), 但它也能够压缩已经紧凑, 完全卷积 SqueezeNet建筑。通过10×压缩 SqueezeNet 的深层压缩, 同时保留基线精度。总而言之: 通过将 CNN 的体系结构创新 (SqueezeNet) 与最先进的压缩技术 (深压缩) 结合在一起, 我们实现了一个 510× 在模型大小上的缩减, 与基线相比, 精确度没有降低。

最后, 请注意, 深压缩 (han等, 2015b) 使用码书作为其方案的一部分, 将 CNN 参数量化为6或8位精度。因此, 在大多数初级商品处理器上, 使用在深压缩中开发的方案, 以8位量化或的速度加速的速度是不是微不足道的. 但是, 韩et。开发的自定义硬件-有效推理引擎 (EIE) -可以更有效地计算码书量化 CNNs (汉族等人, 2016a)。此外, 在我们发布 SqueezeNet 后的几个月中, p Gysel 开发了一个称为Ristretto的策略, 用于线性量化 SqueezeNet 为8位 (Gysel, 2016)。具体地说, Ristretto 在8位中进行计算, 并在8位数据类型中存储参数和激活。在 SqueezeNet 推理中, 使用8位计算的 Ristretto 策略, 在使用8位而不是32位数据类型时, Gysel 观察到精度下降的小于1的百分比.

<div align=center>
<img src="zh-cn/img/squeezenet/p4.png" />
</div>


#### 5.CNN 微体系结构设计空间探索

到目前为止, 我们已经提出了小模型的结构设计策略, 遵循这些原则来创建 SqueezeNet, 并发现 SqueezeNet 比 AlexNet 小 50x的同时保持了相同的精度。然而, SqueezeNet 甚至是其他模型知识广阔的未探索的 CNN 架构的设计空间里的一种。现在, 在5和6节中, 我们探讨了设计空间的几个方面。我们将此体系结构探索分为两个主要主题: microarchitectural 探测(每个模块层的维度和配置) 和macroarchitectural 探测(模块的高级端到端组织和其他层)。

在本节中, 我们设计并执行实验, 目的是提供关于 microarchitectural 设计空间形状的直觉, 就我们在3.1 节中提出的设计策略而言。请注意, 我们在这里的目标不是在每个实验中实现最大的精确度, 而是要了解 CNN 架构选择对模型大小和准确性的影响。

<div align=center>
<img src="zh-cn/img/squeezenet/p5.png" />
</div>

*图3：微结构设计空间探索*

**5.1 CNN 微体系结构参数**

<div align=center>
<img src="zh-cn/img/squeezenet/p6.png" />
</div>


**5.2 压缩比(SR)**

<div align=center>
<img src="zh-cn/img/squeezenet/p7.png" />
</div>

**5.3 训练时关闭1X1和3X3卷积核**

<div align=center>
<img src="zh-cn/img/squeezenet/p8.png" />
</div>



#### 6.CNN 宏观体系结构设计空间探索

到目前为止, 我们已经探索了微体系结构层面的设计空间, 即CNN网络各个模块的内容。现在, 我们在 macroarchitecture 级别上探讨了有关Fire模块之间高层连接的设计决策。灵感来自 ResNet (He等, 2015b), 我们探索了三种不同的体系结构:

+ Vanilla SqueezeNet (按前一节). 
+ SqueezeNet 在某些Fire模块之间进行简单的旁路连接。 
+ SqueezeNet 在Fire模块之间使用复杂的旁路连接。

我们在图2中画出了这三种 SqueezeNet 的变体。

我们的简单旁路体系结构在3、5、7和9的Fire模块附近添加旁路连接, 要求这些模块在输入和输出之间学习残差函数。与 ResNet 一样, 要实现围绕 Fire3 的旁路连接, 我们将输入设置为 Fire4 等于 (Fire2 + 输出 Fire3 的输出), 其中 + 运算符为数组加法。这改变了正规化应用于这些消防Fire模块的参数, 并且, 根据 ResNet, 可以提高最终的准确度。

一个限制是, 在简单的情况下, 输入通道的数量和输出通道的数量必须相同;因此, 只有一半的Fire模块可以有简单的旁路连接, 如图2的中间图所示。当无法满足 “相同数量的通道” 要求时, 我们使用复杂旁路连接, 如图2的右侧所示。虽然一个简单的旁路是 “只是一个导线,” 我们定义一个复杂的旁路作为旁路, 包括一个1x1 卷积层与数量的过滤器设置等于数量的输出通道。需要注意的是, 复杂的旁路连接会向模型中添加额外的参数, 而简单旁路连接则不会。

除了改变正规化我, 我们还可以比较直观地看到： 增加旁路连接将有助于减轻squeeze层引入的瓶颈。例如：在 SqueezeNet中, 挤压比 (SR) 是 0.125, 这意味着每个squeeze层的输出通道比expand层少8倍。由于这种严重的通道数减少, 只有很少的信息可以通过expand层。但是, 通过将旁路连接添加到 SqueezeNet网络中, 我们打开了信息的通道, 使信息可以在不同的squeeze层之间传输。

我们按照图2中的三种结构训练了 SqueezeNet网络， 并比较了表3中的精度和模型大小。我们修正了微体系结构以匹配 SqueezeNet, 如表1在整个探索中所述。复杂和简单的旁路连接相比于基础的SqueezeNet结构，准确性得以改善。有趣的是, 简单的旁路使得精确度的提高比复杂的旁路更高。

#### 7.总结

本文对卷积神经网络的设计空间探索提出了更严格的方法。针对这个目标, 我们提出了 SqueezeNet——一个 CNN 体系结构, 它的参数比 AlexNet少50×, 并且在 ImageNet 上保持 AlexNet 级别的准确性。我们还将 SqueezeNet 压缩到小于 0.5MB, 或比不进行任何压缩的 AlexNet小510倍。自从我们在2016年发布这篇论文作为一份技术报告以来, Song han和他的合作者对 SqueezeNet 和模型压缩进行了进一步的实验。使用一种新的方法, 称为Dense-Sparse-Dense (DSD) (Han等, 2016b), han et al.在训练过程中使用模型压缩作为 regularizer, 以进一步提高准确性, 在 ImageNet-1k 上生成一组压缩的 SqueezeNet 参数, 其精度是1.2 百分点, 而且还产生一组未压缩的 SqueezeNet 参数与表2中的结果相比。我们在本文的开头提到, 小模型更适于 fpga 上的芯片实现。自从我们发布了 SqueezeNet 模型后, Gschwend 开发了 SqueezeNet 的变体, 并在 FPGA 上实现了它 (Gschwend, 2016)。正如我们所预料的那样, Gschwend 能够完全在 FPGA 内存储 SqueezeNet 样模型的参数, 并消除了对负载模型参数的非芯片内存访问的需要。

在本文的背景下, 我们将 ImageNet 作为目标数据集。然而, 将 ImageNet 训练的 CNN 表示法应用于各种应用, 如细粒度对象识别, 图像标识 (Iandola 等, 2015), 并生成关于图像语义 (方等, 2015)。ImageNettrained CNNs 也适用于一些有关自主驾驶的应用, 包括在图像中的行人和车辆检测 (Iandola 等, 2014;Girshick 等, 2015;阿什拉夫等, 2016) 和视频 (陈等, 2015b), 以及分割道路的形状 (Badrinarayanan 等, 2015)。我们认为 SqueezeNet 将是一个很好的候选人 CNN 架构的各种应用, 特别是那些小模型的大小是重要的。

SqueezeNet 是我们在广泛探索 CNN 体系结构设计空间时发现的几种新 CNNs 之一。我们希望 SqueezeNet 能激励读者考虑和探索 CNN 架构设计空间中广泛的可能性, 并以更系统的方式进行探索。

**关于本论文的几点思考：**

在项目中用过这个模型，确实很给力，在准确度不怎么变化的情况下可以使模型的参数减少很多。以往的压缩模型设计思路，都是在现有模型的基础上进行一些有损压缩，修修补补，但作者在这里另辟蹊径，直接根据他制定的三种可以使模型参数减少的策略设计了一种全新的模型，而且效果还不错。

对于大多数人来说，只要使用这个模型就够了，但是论文中作者花大量篇幅介绍的压缩模型的设计思路还是值得设计模型的研究员们借鉴的。


#### code

```python
import tensorflow as tf
import numpy as np
import cv2


class SqueezeNet(object):
    def __init__(self, session, alpha, optimizer=tf.train.GradientDescentOptimizer, squeeze_ratio=1):
        if session:
            self.session = session
        else:
            self.session = tf.Session()

        self.dropout   = tf.placeholder(tf.float32)
        self.target    = tf.placeholder(tf.float32, [None, 1000])
        self.imgs      = tf.placeholder(tf.float32, [None, 224, 224, 3])

        self.alpha = alpha
        self.sq_ratio  = squeeze_ratio
        self.optimizer = optimizer

        self.weights = {}
        self.net = {}

        self.build_model()
        self.init_opt()
        self.init_model()

    def build_model(self):
        net = {}

        # Caffe order is BGR, this model is RGB.
        # The mean values are from caffe protofile from DeepScale/SqueezeNet github repo.
        self.mean = tf.constant([123.0, 117.0, 104.0],
                                dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images = self.imgs-self.mean

        net['input'] = images

        # conv1_1
        net['conv1'] = self.conv_layer('conv1', net['input'],
                              W=self.weight_variable([3, 3, 3, 64], name='conv1_w'), stride=[1, 2, 2, 1])

        net['relu1'] = self.relu_layer('relu1', net['conv1'], b=self.bias_variable([64], 'relu1_b'))
        net['pool1'] = self.pool_layer('pool1', net['relu1'])

        net['fire2'] = self.fire_module('fire2', net['pool1'], self.sq_ratio * 16, 64, 64)
        net['fire3'] = self.fire_module('fire3', net['fire2'], self.sq_ratio * 16, 64, 64,   True)
        net['pool3'] = self.pool_layer('pool3', net['fire3'])

        net['fire4'] = self.fire_module('fire4', net['pool3'], self.sq_ratio * 32, 128, 128)
        net['fire5'] = self.fire_module('fire5', net['fire4'], self.sq_ratio * 32, 128, 128, True)
        net['pool5'] = self.pool_layer('pool5', net['fire5'])

        net['fire6'] = self.fire_module('fire6', net['pool5'], self.sq_ratio * 48, 192, 192)
        net['fire7'] = self.fire_module('fire7', net['fire6'], self.sq_ratio * 48, 192, 192, True)
        net['fire8'] = self.fire_module('fire8', net['fire7'], self.sq_ratio * 64, 256, 256)
        net['fire9'] = self.fire_module('fire9', net['fire8'], self.sq_ratio * 64, 256, 256, True)

        # 50% dropout
        net['dropout9'] = tf.nn.dropout(net['fire9'], self.dropout)
        net['conv10']   = self.conv_layer('conv10', net['dropout9'],
                               W=self.weight_variable([1, 1, 512, 1000], name='conv10', init='normal'))
        net['relu10'] = self.relu_layer('relu10', net['conv10'], b=self.bias_variable([1000], 'relu10_b'))
        net['pool10'] = self.pool_layer('pool10', net['relu10'], pooling_type='avg')

        avg_pool_shape        = tf.shape(net['pool10'])
        net['pool_reshaped']  = tf.reshape(net['pool10'], [avg_pool_shape[0],-1])
        self.fc2              = net['pool_reshaped']
        self.logits           = net['pool_reshaped']

        self.probs = tf.nn.softmax(self.logits)
        self.net   = net

    def init_opt(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))
        self.optimize = self.optimizer(self.alpha).minimize(self.cost)

    def init_model(self):
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def bias_variable(self, shape, name, value=0.1):
        initial = tf.constant(value, shape=shape)
        self.weights[name] = tf.Variable(initial)
        return self.weights[name]

    def weight_variable(self, shape, name=None, init='xavier'):
        if init == 'variance':
            initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
        elif init == 'xavier':
            initial = tf.get_variable('W' + name, shape, initializer=tf.contrib.layers.xavier_initializer())
        else:
            initial = tf.Variable(tf.random_normal(shape, stddev=0.01), name='W'+name)

        self.weights[name] = initial
        return self.weights[name]

    def relu_layer(self, layer_name, layer_input, b=None):
        if b:
            layer_input += b
        relu = tf.nn.relu(layer_input)
        return relu

    def pool_layer(self, layer_name, layer_input, pooling_type='max'):
        if pooling_type == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 13, 13, 1],
                              strides=[1, 1, 1, 1], padding='VALID')
        elif pooling_type == 'max':
            pool = tf.nn.max_pool(layer_input, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
        return pool

    def conv_layer(self, layer_name, layer_input, W, stride=[1, 1, 1, 1]):
        return tf.nn.conv2d(layer_input, W, strides=stride, padding='SAME')

    def fire_module(self, layer_name, layer_input, s1x1, e1x1, e3x3, residual=False):
        """ Fire module consists of squeeze and expand convolutional layers. """
        fire = {}

        shape = layer_input.get_shape()

        # squeeze
        s1_weight = self.weight_variable([1, 1, int(shape[3]), s1x1], layer_name + '_s1')

        # expand
        e1_weight = self.weight_variable([1, 1, s1x1, e1x1], layer_name + '_e1')
        e3_weight = self.weight_variable([3, 3, s1x1, e3x3], layer_name + '_e3')

        fire['s1'] = self.conv_layer(layer_name + '_s1', layer_input, W=s1_weight)
        fire['relu1'] = self.relu_layer(layer_name + '_relu1', fire['s1'],
                                        b=self.bias_variable([s1x1], layer_name + '_fire_bias_s1'))

        fire['e1'] = self.conv_layer(layer_name + '_e1', fire['relu1'], W=e1_weight)
        fire['e3'] = self.conv_layer(layer_name + '_e3', fire['relu1'], W=e3_weight)
        fire['concat'] = tf.concat([tf.add(fire['e1'], self.bias_variable([e1x1],
                                                           name=layer_name + '_fire_bias_e1' )),
                                    tf.add(fire['e3'], self.bias_variable([e3x3],
                                                           name=layer_name + '_fire_bias_e3' ))], 3)

        if residual:
            fire['relu2'] = self.relu_layer(layer_name + 'relu2_res', tf.add(fire['concat'],layer_input))
        else:
            fire['relu2'] = self.relu_layer(layer_name + '_relu2', fire['concat'])

        return fire['relu2']

    def save_model(self, path):
        """
        Save the neural network model.
        :param path: path where will be stored
        :return: path if success
        """
        saver = tf.train.Saver(self.weights)
        save_path = saver.save(self.session, path)
        return save_path

    def load_model(self, path):
        """
        Load neural network model from path.
        :param path: path where is network located.
        :return: None
        """
        saver = tf.train.Saver(self.weights)
        saver.restore(self.session, path)

if __name__ == '__main__':
    sess = tf.Session()
    alpha= tf.placeholder(tf.float32)
    net  = SqueezeNet(sess, alpha)

    img1 = cv2.imread('./images/architecture.png')#, mode='RGB')
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (224, 224))
    prob = sess.run(net.probs, feed_dict={net.net['input']: [img1], net.dropout:1.0})
    print(prob)
    net.save_model('./test.ckpt')
    net.load_model('./test.ckpt')
```

*来源：https://github.com/Dawars/SqueezeNet-tf.git*
