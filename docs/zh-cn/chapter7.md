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

------

### 4.EffNet

------



### 5.SqueezeNet

------



