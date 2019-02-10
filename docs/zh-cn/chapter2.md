## Inception
------


### 1.NIN
------

1X1的卷积核，我们在深度学习之CNN中解释了为什么卷积核往往就是奇数，并且着重介绍了1X1的卷积核的作用，本节将重复介绍《Network in NetWork》 paper中关于1x1卷积核的作用，并为Inception中的应用做铺垫。

NIN的paper主要对CNN结构进行改进，是得CNN能够学习更加抽象和有效的非线性特征。 统的卷积层只是将前一层的特征进行了线性组合，然后经过一个非线性激活。而在文章中，作者提出了使用一个微小的神经网络(主要是多层感知器)来替代普通的卷积过程，当然，这个微小的神经网络也和卷积的权值共享一样，即对于相同的特征层是相同的。传统的卷积过程和微网络分别如图所示：

<div align=center>
<img src="zh-cn/img/NIN/1.png" />
</div>

+  由图可知，mlpconv=convolution+mlp（图中为2层的mlp）。

+ 在实现上,mlpconv=convolution+1×1convolution+1×1convolution（2层的mlp） 

+ 实现NIN就是在原来的卷积后再加上1X1的卷积即可，如下图，有三个NIN层，那么第一个NIN的实现应该是conv1[3X3], conv2[1X1],conv3[1X1]

<div align=center>
<img src="zh-cn/img/NIN/2.png" />
</div>


作者之所以进行这样的改进，主要是因为，传统的卷积层只是一个线性的过程，而且，层次比较深的网络层是对于浅层网络层学习到的特征的整合，因此，在对特征进行高层次整合之前，进行进一步的抽象是必要的。

对于选择多层感知器作为微网络结构，主要是处于下面两个方面的考虑：

+ 多层感知器也是使用BP算法进行训练的，可以与CNN进行整合；

+ 多层感知器也可以作为一个深层结构，也包含了特征重用的思想。

除此之外，文中作者使用了Global Average Pooling，来替代全连接层，这个主要是为了使得最后一个多层感知卷积层获得的每一个特征图能够对应于一个输出类别。
使用全局平均池化的优点有两个：

+ 使用全局平均池化能够强化特征图与类别的关系；

+ 全局平均池化没有参数需要进行优化，因此，可以避免在这一层出现过拟合。

全局平均池化的过程是对于每一个特征图求出其平均数，然后将这些平均数组成一个特征向量，输入到softmax层中。如下图所示：

<div align=center>
<img src="zh-cn/img/NIN/3.jpg" />
</div>


**Reference:**

[1]. Network In Network


NIN 这种思想被抽象成1X1的卷积核，这种1X1的卷积核被大量的应用在GoogleNet中。


------

Google Inception Ne首次出现在ILSVRC 2014的比赛中，就以较大的优势取得第一名，那一届的比赛中的Inception Net通常被称为Inception V1.接着后边又出现了其他版本的Inception包括(v2,v3,v4),接下来我们就看一下基本的Inception结构及每一个版本的Inception的改进，最后通过Tendorflow实现一个Inception网络结构。

### 2.GoogleNet(Inception V1)

基本的网络结构，是把每个卷积核作用的结构进行堆叠，形成新的更深chanel的结构，如下图所示：

<div align=center>
<img src="zh-cn/img/Inception/1.png" /> 
</div>

为了降低计算的复杂度和参数个数引入了1X1的卷积核：


<div align=center>
<img src="zh-cn/img/Inception/2.png" /> 
</div>

论文中对应的结构为：


<div align=center>
<img src="zh-cn/img/Inception/3.png" /> 
</div>

GoogleLeNet的结构如下图所示：


<div align=center>
<img src="zh-cn/img/Inception/4.png" /> 
</div>

Inception V1参数少且效果好的原因除了模型层数更深，表达能力更强之外，还有两点：一是去除了最后的全连接层，用NIN中的全局平均池化（参数少，泛化能力将会变强），二的原因是Inception V1中精心设计的Inception Module模块提高了参数的利用率。实际上这一部分可以借鉴与NIN中的思想，相当于在大网络中套了很多个子网络。除此之外InceptionV1网络层的中间层也会有输出。

Inception V1有22层的深度，除了最后一层的输出，中间节点的分类效果也不错，因此在Inception Net中，还是用了辅助分类节点，即将中间层的输出作为分类，并按照一定的权重加（比如0.3)加到最终的分类结果，这样相当于做了模型融合，就像在论文应引入的网络俚语一样：

<div align=center>
<img src="zh-cn/img/Inception/5.png" /> 
</div>

除此之外，Google Inception Net还有一个大家族，包括：

+ 2014年9月：Going Deeper with Convolutions提出的Inception V1（top-5的错误率6.67%）
+ 2015年2月：Batch Normalization:Accelerating Deep Network Training by Reducing Internal Covariate提出的Inception V2（Top-5的错误率是4.8%）
+ 2015年12月：Rethinking the Inception Architecture for Computer Vision提出的Inception V3（top-5错误率3.5%）
+ 2016年2月：Inception-v4,Inception-ResNet and the Impact of Residual Commentions on Learning提出的Inception V4(top-5错误率3.08%）

下面我们就以依次来看一下这几个网络结构。


### 3.Inception V2

Inceptin V2对V1做了如下的改变：首先在网络结构上，用两个3X3的卷积代替5X5的大卷积(用以降低参数量并减轻过拟合),同时和提出了著名的Batch Normalization（BN),下面我们细致的看一下这两个变化：

关于卷积核的变化：


<div align=center>
<img src="zh-cn/img/Inception/6.png" /> 
</div>

关于BN:

BN是一种非常有效的正则化方法，可以让大型卷积网络的训练速度加快很多倍，同时收敛后的分类准确率也可以大幅提高。BN在用于神经网络某层时，会对每一个mini-batch数据的内部进行标准化处理，使得输出规范到N(0,1)的正态分布，减少了内部神经元分布的改变(Internal Covariate Shift)。论文中提到，传统的深度神经网络在训练时，每一层的输入的分布都在变化，导致训练变得困难，我们只能使用一个很小的学习率来解决这个问题。而对于每一层使用BN之后，就可以有效的解决这个问题。因为BN某种意义上还起到了正则化的作用，因此可以减少或取消Dropout，简化网络结构。

BN算法:


<div align=center>
<img src="zh-cn/img/Inception/7.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/Inception/8.png" /> 
</div>

------


### 4.Inception V3

Inception V3网络主要有两方面的改进：一是引入了Factorization into small convolutions的思想，将一个较大的二维卷积拆成两个较小的一维卷积，比如将7X7的卷积拆成1X7和7X1的卷积，或者将3X3的卷积拆成1X3或3x1的卷积,如下图所示：

<div align=center>
<img src="zh-cn/img/Inception/9.png" /> 
</div>

一方面节约了大量的参数，加速运算减轻了过拟合；同时增加了一层非线性扩展模型的表达能力。论文指出，这种非对称的卷积结构拆分，其结果比对称的拆分为几个相同的小卷积核效果更明显，可以处理更多，更丰富的空间特征，增加特征多样性。

另一方面Inception V3优化了Inception Module的结构，现在Inception module有35X35,17X17,8X8三种不同的结构(如下图所示),这些Inception Module只在网络的后部分出现，前部还是普通的卷积层。并且Inception V3除在Inception Module中使用分支，还在分支中使用了分支（8X8)的结构中，可以说是Network in Network in Network。

<div align=center>
<img src="zh-cn/img/Inception/10.png" /> 
</div>

Inception V3模型总共46层，由11个Inception模块组成，有96个卷积层，我们在第6部分将实现一个Inception V3的模型。

------  

### 5.Inception V4

Inception V4主要利用残差链接(Residual Connection）来改进v3结构，代表作有Inception-ResNet-v1,Inception-ResNet-v2,inception-v4。Inception-ResNet的改进就是使用以上三部分中的Inception module来替换ResNet shortcut中额1X1conv+conv,如下图所示:

<div align=center>
<img src="zh-cn/img/Inception/11.png" /> 
</div>

+ 将Inception Module与Resdual Connection结合，提出了Inception-ResNet-v1,Inception-ResNet-v2,使得训练加速收敛更快，精度更高。

+ 设计了更深的Inception-v4版本，效果和Inception-ResNet-v2相当

+ 网络输入大小和V3相同买还是299X299

下面我们看这几种网路结构


Inception V4:

<div align=center>
<img src="zh-cn/img/Inception/12.png" /> 
</div>

Inception-ResNet V1:

<div align=center>
<img src="zh-cn/img/Inception/13.png" /> 
</div>

Inception-ResNet v2:

<div align=center>
<img src="zh-cn/img/Inception/14.png" /> 
</div>

------

### 6.Tensorflow实现Inception

本节我们将实现Inception V3结构，关于Google Inception的网络结构的Tnesorflow实现可以参考Tensorflow的官方示例代码。下面我们使用tensorflow-sim工具来更加简洁的实现一个卷积层

```python
# 直接使用Tnesorflow原始API实现卷积层

with tf.variable_scope(scope_name):
	weights = tf.get_variable("weights",...)
	biases = tf.get_variable("bias",...)
	conv = tf.nn.conv2d(...)
relu = tf.nn.rrlu(tf.nn.bias_add(conv,biases))

import tensorflow as tf
slim = tf.contrib.slim
# 使用tensorflow-slim实现卷积层。通过tensorflow-slim可以在一行中实现一个卷积层
# 的前向传播算法。slim.conv2d函数的有三个参数是必要的，第一个参数为输入节点矩阵
# 第二个参数为当前卷积层过滤器的深度，第三个参数是过滤器的尺寸。可选的参数有过滤器
# 移动的步长，是否使用全0填充，激活函数的选择以及变量的命名空间等。

net = slim.conv2d(input,32,[3,3])
# slim.arg_scope函数可以用于设置默认的参数取值，slim.arg_scope函数的第一个参数是
# 一个函数列表，在这个列表中的函数将使用默认的参数取值。比如通过下面的定义，调用
# slim.conv2d(net,320,[1,1])函数时会自动加上stride=1和padding='SAME'的参数。如果在
# 函数调用时指定了stride,那么这里设置的默认值就不会再使用，通过这种方式可以进一步减少
# 冗余的代码

with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding="SAME"):
	...
	# 此处省略Inception v3中的其他结构，而直接实现Inception Module部分的结构，其他结构
	# 写法是相同的。
	net = 上一层的输出节点矩阵

	# 为一个Inception Module声明一个统一的命名空间
	with tf.variable_scope("Maxed_7c"):
		# 给Inception Module中的每条路径声明一个命名空间
		with tf.variable_scope("Branch_0"):
			# 实现一个过滤器边长为1，深度为320的卷积层
			branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')

		# Inception Module中的第2条路径，这条路径上的结构本身也是一个Inception结构
		with yf.variable_scope("Branch_1"):
			branch_1 = slim.conv2d(net,384,[1,1],scope="Conv2d_0a_1X1")
			# tf.concat函数可以将多个矩阵拼接起来。tf.concat函数的第一个参数指定了
			# 拼接的维度，这里给出的“3”表示矩阵是在深度这个维度上进行拼接，这就是
			# Inception Module中的拼接方式

			branch_1 = tf.concat(3,[
				# 注意此处的两层卷积的输入都是branch_1,而不是net
				slim.conv2d(branch_1,384,[1,3],scope="Conv2d_0b_1X3"),
				slim.conv2d(branch_1,384,[3,1],scope="Conv2d_0b_3X1") ])

		# Inception Module中的第3条路径，此时计算路径也是一个Inception结构
		with tf.variable_scope("Branch_2"):
			branch_2 = slim.conv2d(net,448,[1,1],scope="Conv2d_0a_1X1")
			branch_2 = slim.conv2d(branch_2,384,[3,3],scope="Cnv2d_0b_3X3")
			branch_2 = tf.concat(3,[
				slim.conv2d(branch_2,384,[1,3],scope="Conv2d_0c_1X3")
				slim.conv2d(branch_2,384,[3,1],scope="Conv2d_0d_3X1")
				])

		# Inception Module中额第4条路径.
		with tf.variable_scope("Branch_3"):
			branch_3 = slim.avg_pool2d(net,[3,3],scope="AvgPool_0a_3X3")
			branch_3 = slim.conv2d(branch_3,192,[1,1],scope="Conv2d_0b_1X1")
		#当前Inception Module的最后输出是由上面四个计算结果拼接得到的
		net = tf.concat(3,[branch_0,branch_1,branch_2,brnch_3])
```

<div align=center>
<img src="zh-cn/img/Inception/15.png" /> 
</div>
