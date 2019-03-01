## N种卷积

研读了大量的CNN相关的paper,一直想总结Deep Learning中常用的几种卷积， Kunlun Bai 近日发布一篇介绍深度学习的卷积文章，用浅显易懂的方式介绍了深度学习领域的各种卷积及其优势。本节介绍深度学习中常用的卷积结构。

1. 卷积与互相关
2. 深度学习中的简单卷积（单通道和多通道）
3. 1D和3D卷积
4. 1X1卷积
5. 转置卷积（反卷积）
6. 空洞(扩张)卷积(Dilated Conv),多孔(带洞)卷积(Atrous Conv)
7. 可分卷积（空间可分卷积，深度可分卷积）
8. 分组卷积



### 1.卷积与互相关

在信号处理，图像，文本等非结构化的数据处理领域，卷积都是一种使用广泛的技术。在深度学习领域，CNN这种模型架构就得名于这种技术。深度学习领域的卷积，本质是信号或图像处理领域内的互相关（cross-correlation）。

在信号或图像处理领域，卷积的定义如下：

<div align=center>
<img src="zh-cn/img/conv12/p1.png" /> 
</div>

其定义是两个函数中一个函数经过**反转**和**位移**后再相乘得到的积的积分，可视化展示了这种思想：

<div align=center>
<img src="zh-cn/img/conv12/p2.png" /> 
</div>

函数 g 是过滤器。它被反转后再沿水平轴滑动。在每一个位置，我们都计算 f 和反转后的 g 之间相交区域的面积。这个相交区域的面积就是特定位置出的卷积值。

统计学中简单的定义相关性就是通过函数之间的滑动点积或滑动內积，互相关中的过滤器不经过反转。f 与 g 之间的交叉区域即是互相关。下图展示了卷积与互相关之间的差异。

<div align=center>
<img src="zh-cn/img/conv12/p3.png" /> 
</div>

在深度学习中，卷积中的过滤器不经过反转。严格来说，这是互相关。我们本质上是执行逐元素乘法和加法。但在深度学习中，直接将其称之为卷积更加方便。这没什么问题，因为过滤器的权重是在训练阶段学习到的。如果上面例子中的反转函数 g 是正确的函数，那么经过训练后，学习得到的过滤器看起来就会像是反转后的函数 g。因此，在训练之前，没必要像在真正的卷积中那样首先反转过滤器。

### 2.深度学习中通常意义的卷积

可参见3.2节或链接  <https://dataxujing.github.io/深度学习之CNN/>

### 3.1D和3D卷积

1.1D卷积

<div align=center>
<img src="zh-cn/img/conv12/p4.png" /> 
</div>
图片来源:Andrew Ng deeplearning.ai from Coursera 

2.3D卷积

深度学习中常用的对一个3D体积执行卷积，但通常而言，仍称为2D卷积。这是在3D体积数据上的2D卷积，卷积核的深度和输入深度（channel的个数）是一至的，这个3D过滤器仅沿两个方向移动（图像的高和宽）。这种操作的输出是一张2D图像（仅有一个通道）

而3D卷积怎么做呢？其卷积核深度小于输入深度（卷积核的大小小于通道大小）。因此，3D过滤器可以在所有三个方向（图像的宽，高，深度（通道））上移动。在每个位置，逐元素的乘法和加法都会提供一个数值。因为过滤器是划过了一个3D空间排布。

<div align=center>
<img src="zh-cn/img/conv12/p5.png" /> 
</div>

3D 卷积可以描述 3D 空间中目标的空间关系。对某些应用（比如生物医学影像中的 3D 分割/重构）而言，这样的 3D 关系很重要，比如在 CT 和 MRI 中，血管之类的目标会在 3D 空间中蜿蜒曲折。

<div align=center>
<img src="zh-cn/img/conv12/p6.png" /> 
</div>

### 4.1X1卷积

略，详细的参考 《Network in Network》

### 5.转置卷积（反(去)卷积）

对于很多网络架构的很多应用而言，我们往往需要进行与普通卷积方向相反的转换，即我们希望执行上采样。例子包括生成高分辨率图像以及将低维特征图映射到高维空间，比如在自动编码器或形义分割中。（在后者的例子中，形义分割首先会提取编码器中的特征图，然后在解码器中恢复原来的图像大小，使其可以分类原始图像中的每个像素。）在DCGAN中的生成器将会用随机值转变为一个全尺寸(full-size)的图片，这个时候就需要用到转置卷积。


实现上采样的传统方法是应用插值方案或人工创建规则。而神经网络等现代架构则倾向于让网络自己自动学习合适的变换，无需人类干预。为了做到这一点，我们可以使用转置卷积

转置卷积（transposed Convolutions）又名反卷积（deconvolution）或是分数步长卷积（fractially straced convolutions）。反卷积（Deconvolution）的概念第一次出现是Zeiler在2010年发表的论文Deconvolutional Networks中。需要指出「反（去）卷积（deconvolution）」这个名称并不是很合适，因为转置卷积并非信号/图像处理领域定义的那种真正的去卷积。从技术上讲，信号处理中的去卷积是卷积运算的逆运算。但这里却不是这种运算。因此，某些作者强烈反对将转置卷积称为去卷积。人们称之为去卷积主要是因为这样说很简单。后面我们会介绍为什么将这种运算称为转置卷积更自然且更合适。

**一般卷积的过程:**

<div align=center>
<img src="zh-cn/img/conv12/p7.png" /> 
</div>

卷积核大小是3X3，stride是2，padding是1的普通卷积，通过对应关系我们发现输入元素a仅于第一个输出元素1有关，而输入元素b和输出元素1和2均有关。同理可以看到其他元素的关系，那么进行转置卷积时，依然应该保持这个链接关系不变。

**转置卷积的过程：**

我们需要将上图中绿色的特征图作为输入，蓝色的特征图作为输出，并且保证连接关系不变。怎么才能达到这个效果呢？我们可以先用0给绿色特征图做插值，插值的个数就是使相邻两个绿色元素的间隔为卷积的步长，同时边缘也需要进行与插值数量相等的补0。如下图： 

<div align=center>
<img src="zh-cn/img/conv12/p8.png" /> 
</div>

注意，这时候卷积核的滑动步长就不是2了，而是1，步长体现在了插值补0的过程中。

我们一直都可以使用直接的卷积实现转置卷积。对于下图的例子，我们在一个 2×2 的输入（周围加了 2×2 的单位步长的零填充）上应用一个 3×3 核的转置卷积。上采样输出的大小是 4×4。

<div align=center>
<img src="zh-cn/img/conv12/p9.png" /> 
</div>

有趣的是，通过应用各种填充和步长，我们可以将同样的 2×2 输入图像映射到不同的图像尺寸。下面，转置卷积被用在了同一张 2×2 输入上（输入之间插入了一个零，并且周围加了 2×2 的单位步长的零填充），所得输出的大小是 5×5。

<div align=center>
<img src="zh-cn/img/conv12/p10.png" /> 
</div>

**为什么叫转置卷积**

1.卷积的矩阵乘法

<div align=center>
<img src="zh-cn/img/conv12/p11.png" /> 
</div>

输入平展为 16×1 的矩阵，并将卷积核转换为一个稀疏矩阵（4×16）。然后，在稀疏矩阵和平展的输入之间使用矩阵乘法。之后，再将所得到的矩阵（4×1）转换为 2×2 的输出。

现在，如果我们在等式的两边都乘上矩阵的转置 CT，并借助「正交阵：一个矩阵与其转置矩阵的乘法得到一个单位矩阵」这一性质，那么我们就能得到公式 CT x Small = Large，如下图所示。

<div align=center>
<img src="zh-cn/img/conv12/p12.png" /> 
</div>

这里可以看到，我们执行了从小图像到大图像的上采样。

转置卷积的算术解释可参阅：

+ <https://arxiv.org/abs/1603.07285>
+ <https://github.com/vdumoulin/conv_arithmetic>
+ <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>

**反卷积网络(Deconvolutional Networks)**

反卷积的用处还挺广的，涉及到 
- visualization 
- pixel-wiseprediction 
- unsupervised learning 

都会用到反卷积的结构。比如Deconvolutional Network做图片的unsupervised feature learning，ZF-Net论文中的卷积网络可视化，FCN网络中的upsampling，GAN中的Generative图片生成。Matthew D. Zeiler这位大牛在反卷积上做了很大贡献，先后发表多篇有关反卷积的paper

- 提出反卷积概念：Zeiler M D, Krishnan D, Taylor G W, et al. Deconvolutional networks[C]. Computer Vision and Pattern Recognition, 2010. 

- 无监督学习： Zeiler M D, Taylor G W, Fergus R, etal. Adaptive deconvolutional networks for mid and high level featurelearning[C]. International Conference on Computer Vision, 2011.以无监督学习形式提出“反卷积”概念。 

- 卷积网络可视化 ： Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[C]. European Conference on Computer Vision, 2013. 


深度网络结构是由多个单层网络叠加而成的，而常见的单层网络按照编码解码情况可以分为下面3类：

+ 既有encoder部分也有decoder部分：比如常见的RBM系列（由RBM可构成的DBM, DBN等），autoencoder系列(以及由其扩展的sparse autoencoder, denoise autoencoder, contractive autoencoder, saturating autoencoder等)。
+ 只包含decoder部分：比如sparse coding, 和deconvolution network.
+ 只包含encoder部分，那就是普通的feed-forward network.

Deconvolution Network的中文名字是反卷积网络，那么什么是反卷积呢？其概念从字面就很容易理解，假设A=BXC 表示的是：B和C的卷积是A，也就是说已知B和C，求A这一过程叫做卷积。那么如果已知A和B求C或者已知A和C求B，则这个过程就叫做反卷积了: deconvolution.

<div align=center>
<img src="zh-cn/img/conv12/p41.png" width='120%' height='120%' /> 
</div>

<div align=center>
<img src="zh-cn/img/conv12/p42.png" width='120%' height='120%'  /> 
</div>

<div align=center>
<img src="zh-cn/img/conv12/p43.png" width='120%' height='120%'  /> 
</div>

<div align=center>
<img src="zh-cn/img/conv12/p39.png" /> 
</div>


值得注意的反卷积虽然存在，但是在深度学习中并不常用。而转置卷积虽然又名反卷积，却不是真正意义上的反卷积。因为根据反卷积的数学含义，通过反卷积可以将通过卷积的输出信号，完全还原输入信号。而事实是，转置卷积只能还原shape大小，而不能还原value。你可以理解成，至少在数值方面上，转置卷积不能实现卷积操作的逆过程。所以说转置卷积与真正的反卷积有点相似，因为两者产生了相同的空间分辨率。但是又名反卷积（deconvolutions）的这种叫法是不合适的，因为它不符合反卷积的概念。


### 6.Dilated Conv/Atrous Conv

扩张卷积由这几篇paper引入：

+ [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)

+ [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)

+ [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

+ [DeepLab V2](https://arxiv.org/pdf/1412.7062v3.pdf)

+ [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

+ [Deeplab v3+](http://cn.arxiv.org/abs/1802.02611)


**空洞卷积:Dilated Convolutions**

Dilated Convolutions，翻译为扩张卷积或空洞卷积。扩张卷积与普通的卷积相比，除了卷积核的大小以外，还有一个扩张率(dilation rate)参数，主要用来表示扩张的大小。扩张卷积与普通卷积的相同点在于，卷积核的大小是一样的，在神经网络中即参数数量不变，区别在于扩张卷积具有更大的感受野。感受野是卷积核在图像上看到的大小，例如3×3卷积核的**感受野**大小为9。

<div align=center>
<img src="zh-cn/img/conv12/p33.png" /> 
</div>


在空洞卷积中有个重要的参数叫rate，这个参数代表了空洞的大小。
要理解空洞概念和如何操作可以从两个角度去看。

1）从原图角度，所谓空洞就是在原图上做采样。采样的频率是根据rate参数来设置的，当rate为1时候，就是原图不丢失任何信息采样，此时卷积操作就是标准的卷积操作，当rate>1，比如2的时候，就是在原图上每隔一（rate-1）个像素采样，如图b，可以把红色的点想象成在原图上的采样点，然后将采样后的图像与kernel做卷积，这样做其实变相增大了感受野。

2）从kernel角度去看空洞的话就是扩大kernel的尺寸，在kernel中，相邻点之间插入rate-1个零，然后将扩大的kernel和原图做卷积 ，这样还是增大了感受野。

(a) 普通卷积，1-dilated convolution，卷积核的感受野为3×3=9。

(b) 扩张卷积，2-dilated convolution，卷积核的感受野为7×7=49。

(c) 扩张卷积，4-dilated convolution，卷积核的感受野为15×15=225。

从上图中可以看出，卷积核的参数个数保持不变，感受野的大小随着“dilation rate”参数的增加呈指数增长。

论文《Multi-scale context aggregation by dilated convolutions》的作者用多个扩张卷积层构建了一个网络，其中扩张率 l 每层都按指数增大。由此，有效的感受野大小随层而指数增长，而参数的数量仅线性增长。

<div align=center>
<img src="zh-cn/img/conv12/p36.png" /> 
</div>

扩展卷积在保持参数个数不变的情况下增大了卷积核的感受野，同时它可以保证输出的特征映射（feature map）的大小保持不变。一个扩张率为2的3×3卷积核，感受野与5×5的卷积核相同，但参数数量仅为9个，是5×5卷积参数数量的36%。扩张卷积在**图像分割**、**语音合成**、**机器翻译**、**目标检测**中都有应用。


**多孔卷积:Atrous Convolutions**

Atrous 卷积，就是带洞的卷积，卷积核是稀疏的。

<div align=center>
<img src="zh-cn/img/conv12/p34.png" /> 
</div>

上图（b）是带洞卷积，可以跳着选，隔一个加一个。

下图中第三个示例（c），就是带洞卷积。

<div align=center>
<img src="zh-cn/img/conv12/p35.png" /> 
</div>

带洞卷积减少了核的大小，可以达到节省内存的作用。

而且带洞卷积的有效性基于一个假设：紧密相邻的像素几乎相同，全部纳入属于冗余，不如跳H(hole size)个取一个。

自己感觉空洞卷积和多孔卷积是同一个东西，但是来自于不同的文章，《Deformable Convolutional Networks》paper验证了我们的判断。解决的问题也主要在图像中的语义分割上。具体的参阅相关paper。


### 7. 可分卷积（深度可分卷积,空间可分卷积）

**深度可分卷积**

可分卷积在深度学习领域要常用得多（MobileNet V1, Xception）。深度卷积和1X1卷积。

还是回到普通的卷积：

<div align=center>
<img src="zh-cn/img/conv12/p18.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/conv12/p19.png" /> 
</div>

深度可分卷积：

第一步：我们不使用 2D 卷积中大小为 3×3×3 的单个过滤器，而是分开使用 3 个核。每个过滤器的大小为 3×3×1。每个核与输入层的一个通道卷积（仅一个通道，而非所有通道！）。每个这样的卷积都能提供大小为 5×5×1 的映射图。然后我们将这些映射图堆叠在一起，创建一个 5×5×3 的图像。经过这个操作之后，我们得到大小为 5×5×3 的输出。

第二步，为了扩展深度，我们应用一个核大小为 1×1×3 的 1×1 卷积。将 5×5×3 的输入图像与每个 1×1×3 的核卷积，可得到大小为 5×5×1 的映射图。

<div align=center>
<img src="zh-cn/img/conv12/p20.png" /> 
</div>

因此，在应用了 128 个 1×1 卷积之后，我们得到大小为 5×5×128 的层。

<div align=center>
<img src="zh-cn/img/conv12/p21.png" /> 
</div>

通过这两个步骤，深度可分卷积也会将输入层（7×7×3）变换到输出层（5×5×128）。


<div align=center>
<img src="zh-cn/img/conv12/p22.png" /> 
</div>

深度可分卷积效率上要比普通的卷积高。因为乘法的计算次数变少了，具体的可以参考MobileNet V1这个算法。小模型的话对算法的准确度上会有很大的影响！

还有就是要注意深度可分卷积与Inception结构的区别和联系，其本质就是使交叉通道相关性和空间相关性充分解耦。详细的参考Xception

example: MobileNet V1 （测试效果并不理想，如果效率没问题不建议使用！）

<div align=center>
<img src="zh-cn/img/conv12/p23.png" /> 
</div>

每一步都添加BN和Relu

用kivy实现安卓端开发，并调用保存的tensorflow lite 模型，确实效率没问题，但是准确率上会打折扣！

<div align=center>
<img src="zh-cn/img/conv12/p24.png" /> 
</div>

**空间可分卷积**

空间可分卷积操作的是图像的 2D 空间维度，即高和宽。从概念上看，空间可分卷积是将一个卷积分解为两个单独的运算。对于下面的示例，3×3 的 Sobel 核被分成了一个 3×1 核和一个 1×3 核。

<div align=center>
<img src="zh-cn/img/conv12/p25.png" /> 
</div>

在卷积中，3×3 核直接与图像卷积。在空间可分卷积中，3×1 核首先与图像卷积，然后再应用 1×3 核。这样，执行同样的操作时仅需 6 个参数，而不是 9 个。

此外，使用空间可分卷积时所需的矩阵乘法也更少。给一个具体的例子，5×5 图像与 3×3 核的卷积（stride=1，padding=0）要求在 3 个位置水平地扫描核（还有 3 个垂直的位置）。总共就是 9 个位置，表示为下图中的点。在每个位置，会应用 9 次逐元素乘法。总共就是 9×9=81 次乘法。

<div align=center>
<img src="zh-cn/img/conv12/p26.png" /> 
</div>

另一方面，对于空间可分卷积，我们首先在 5×5 的图像上应用一个 3×1 的过滤器。我们可以在水平 5 个位置和垂直 3 个位置扫描这样的核。总共就是 5×3=15 个位置，表示为下图中的点。在每个位置，会应用 3 次逐元素乘法。总共就是 15×3=45 次乘法。现在我们得到了一个 3×5 的矩阵。这个矩阵再与一个 1×3 核卷积，即在水平 3 个位置和垂直 3 个位置扫描这个矩阵。对于这 9 个位置中的每一个，应用 3 次逐元素乘法。这一步需要 9×3=27 次乘法。因此，总体而言，空间可分卷积需要 45+27=72 次乘法，少于普通卷积。

<div align=center>
<img src="zh-cn/img/conv12/p27.png" /> 
</div>

尽管空间可分卷积能节省成本，但深度学习却很少使用它。一大主要原因是并非所有的核都能分成两个更小的核。如果我们用空间可分卷积替代所有的传统卷积，那么我们就限制了自己在训练过程中搜索所有可能的核。这样得到的训练结果可能是次优的。

### 8.分组卷积

AlexNet 论文（<https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>）在 2012 年引入了分组卷积。实现分组卷积的主要原因是让网络训练可在 2 个内存有限（每个 GPU 有 1.5 GB 内存）的 GPU 上进行。下面的 AlexNet 表明在大多数层中都有两个分开的卷积路径。这是在两个 GPU 上执行模型并行化（当然如果可以使用更多 GPU，还能执行多 GPU 并行化）。

<div align=center>
<img src="zh-cn/img/conv12/p28.png" /> 
</div>

这里我们介绍一下分组卷积的工作方式。首先，典型的 2D 卷积的步骤如下图所示。在这个例子中，通过应用 128 个大小为 3×3×3 的过滤器将输入层（7×7×3）变换到输出层（5×5×128）。推广而言，即通过应用 Dout 个大小为 h x w x Din 的核将输入层（Hin x Win x Din）变换到输出层（Hout x Wout x Dout）。

<div align=center>
<img src="zh-cn/img/conv12/p29.png" /> 
</div>

在分组卷积中，过滤器会被分为不同的组。每一组都负责特定深度的典型 2D 卷积。下面的例子能让你更清楚地理解。

<div align=center>
<img src="zh-cn/img/conv12/p30.png" /> 
</div>

上图展示了具有两个过滤器分组的分组卷积。在每个过滤器分组中，每个过滤器的深度仅有名义上的 2D 卷积的一半。它们的深度是 Din/2。每个过滤器分组包含 Dout/2 个过滤器。第一个过滤器分组（红色）与输入层的前一半（[:, :, 0:Din/2]）卷积，而第二个过滤器分组（橙色）与输入层的后一半（[:, :, Din/2:Din]）卷积。因此，每个过滤器分组都会创建 Dout/2 个通道。整体而言，两个分组会创建 2×Dout/2 = Dout 个通道。然后我们将这些通道堆叠在一起，得到有 Dout 个通道的输出层。

1.分组卷积与深度卷积

你可能会注意到分组卷积与深度可分卷积中使用的深度卷积之间存在一些联系和差异。如果过滤器分组的数量与输入层通道的数量相同，则每个过滤器的深度都为 Din/Din=1。这样的过滤器深度就与深度卷积中的一样了。

深度卷积并不会改变层的深度。在深度可分卷积中，层的深度之后通过 1×1 卷积进行扩展。

2.分组卷积有几个优点

第一个优点是高效训练。因为卷积被分成了多个路径，每个路径都可由不同的 GPU 分开处理，所以模型可以并行方式在多个 GPU 上进行训练。相比于在单个 GPU 上完成所有任务，这样的在多个 GPU 上的模型并行化能让网络在每个步骤处理更多图像。[关于tensorflow的并行参考如下链接]

+ <https://dataxujing.github.io/TensorFlow分布式最佳实践/>
+ <https://dataxujing.github.io/TensorFlow分布式并行-2/>
+ <https://dataxujing.github.io/TensorFlow分布式并行-1/>
+ <https://dataxujing.github.io/TensorFlow-GPU-并行/>
+ <https://dataxujing.github.io/TensorFlow-多线程输入数据处理框架/>

在训练非常深的神经网络时，分组卷积会非常重要，正如在 ResNeXt 中那样。

<div align=center>
<img src="zh-cn/img/conv12/p32.png" /> 
</div>
图片来源: https://arxiv.org/abs/1611.05431

In particular, a 101-layer ResNeXt is able to achieve better accuracy than ResNet-200 but has only 50% complexity.

第二个优点是模型会更高效，即模型参数会随过滤器分组数的增大而减少。在之前的例子中，完整的标准 2D 卷积有 h x w x Din x Dout 个参数。具有 2 个过滤器分组的分组卷积有 (h x w x Din/2 x Dout/2) x 2 个参数。参数数量减少了一半。

第三个优点有些让人惊讶。分组卷积也许能提供比标准完整 2D 卷积更好的模型。另一篇出色的博客已经解释了这一点：<https://blog.yani.io/filter-group-tutorial>, 这样显著地减少网络中的参数数量能使其不容易过拟合，因此，一种类似正则化的效果让优化器可以学习得到更准确更高效的深度网络。

此外，每个过滤器分组都会学习数据的一个独特表征。正如 AlexNet 的作者指出的那样，过滤器分组似乎会将学习到的过滤器结构性地组织成两个不同的分组——黑白过滤器和彩色过滤器。

<div align=center>
<img src="zh-cn/img/conv12/p31.png" /> 
</div>

除此之外还有类似于像平展卷积,混洗分组卷积,逐点分组卷积等，不是很常用的卷积操作，感兴趣可自己阅读相关的paper。