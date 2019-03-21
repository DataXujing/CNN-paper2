## SENet

------

### 0.摘要

卷积神经网络建立在卷积运算的基础上，通过融合局部感受野内的空间信息和通道信息来提取信息特征。为了提高网络的表示能力，许多现有的工作已经显示出增强空间编码的好处。在这项工作中，专注于通道，并提出了一种新颖的架构单元，称之为“Squeeze-and-Excitation”（SE）块，通过显式地建模通道之间的相互依赖关系，自适应地重新校准通道式的特征响应。通过将这些块堆叠在一起，可以构建SENet架构，在具有挑战性的数据集中可以进行泛化地非常好。关键的是，SE块以微小的计算成本为现有的最先进的深层架构产生了显著的性能改进。SENets是ILSVRC 2017分类提交的基础，它赢得了第一名，并将top-5错误率显著减少到2.251%，相对于2016年的获胜成绩取得了25%的相对改进。

### 1.引言

卷积神经网络（CNNs）已被证明是解决各种视觉任务的有效模型[19,23,29,41]。对于每个卷积层，沿着输入通道学习一组滤波器来表达局部空间连接模式。换句话说，期望卷积滤波器通过融合空间信息和信道信息进行信息组合，而受限于局部感受野。通过叠加一系列非线性和下采样交织的卷积层，CNN能够捕获具有全局感受野的分层模式作为强大的图像描述。最近的工作已经证明，网络的性能可以通过显式地嵌入学习机制来改善，这种学习机制有助于捕捉空间相关性而不需要额外的监督。Inception架构推广了一种这样的方法[14,39]，这表明网络可以通过在其模块中嵌入多尺度处理来取得有竞争力的准确度。最近的工作在寻找更好地模型空间依赖[1,27]，结合空间注意力[17]。

<div align=center>
<img src="zh-cn/img/senet/p1.png" />
</div>

<div align=center>
<img src="zh-cn/img/senet/p2.png" />
</div>

SE网络可以通过简单地堆叠SE构建块的集合来生成。SE块也可以用作架构中任意深度的原始块的直接替换。然而，虽然构建块的模板是通用的，正如我们6.3节中展示的那样，但它在不同深度的作用适应于网络的需求。在前面的层中，它学习以类不可知的方式激发信息特征，增强共享的较低层表示的质量。在后面的层中，SE块越来越专业化，并以高度类特定的方式响应不同的输入。因此，SE块进行特征重新校准的好处可以通过整个网络进行累积。

新CNN架构的开发是一项具有挑战性的工程任务，通常涉及许多新的超参数和层配置的选择。相比之下，上面概述的SE块的设计是简单的，并且可以直接与现有的最新架构一起使用，其卷积层可以通过直接用对应的SE层来替换从而进行加强。另外，如第四节所示，SE块在计算上是轻量级的，并且在模型复杂性和计算负担方面仅稍微增加。为了支持这些声明，我们开发了一些SENets，即SE-ResNet，SE-Inception，SE-ResNeXt和SE-Inception-ResNet，并在ImageNet 2012数据集[30]上对SENets进行了广泛的评估。此外，为了证明SE块的一般适用性，我们还呈现了ImageNet之外的结果，表明所提出的方法不受限于特定的数据集或任务。

使用SENets，我们赢得了ILSVRC 2017分类竞赛的第一名。我们的表现最好的模型集合在测试集上达到了2.251%的top-5错误率。与前一年的获奖者（2.991%的top-5错误率）相比，这表示25%的相对改进。我们的模型和相关材料已经提供给研究界。

### 2.近期工作

**深层架构**。大量的工作已经表明，以易于学习深度特征的方式重构卷积神经网络的架构可以大大提高性能。VGGNets[35]和Inception模型[39]证明了深度增加可以获得的好处，明显超过了ILSVRC 2014之前的方法。批标准化（BN）[14]通过插入单元来调节层输入稳定学习过程，改善了通过深度网络的梯度传播，这使得可以用更深的深度进行进一步的实验。He等人[9,10]表明，通过重构架构来训练更深层次的网络是有效的，通过使用基于恒等映射的跳跃连接来学习残差函数，从而减少跨单元的信息流动。最近，网络层间连接的重新表示[5,12]已被证明可以进一步改善深度网络的学习和表征属性。

另一种研究方法探索了调整网络模块化组件功能形式的方法。可以用分组卷积来增加基数（一组变换的大小）[13,43]以学习更丰富的表示。多分支卷积可以解释为这个概念的概括，使得卷积算子可以更灵活的组合[14,38,39,40]。跨通道相关性通常被映射为新的特征组合，或者独立的空间结构[6,18]，或者联合使用标准卷积滤波器[22]和1×1卷积，然而大部分工作的目标是集中在减少模型和计算复杂度上面。这种方法反映了一个假设，即通道关系可以被表述为具有局部感受野的实例不可知的函数的组合。相比之下，我们声称为网络提供一种机制来显式建模通道之间的动态、非线性依赖关系，使用全局信息可以减轻学习过程，并且显著增强网络的表示能力。

**注意力和门机制**。从广义上讲，可以将注意力视为一种工具，将可用处理资源的分配偏向于输入信号的信息最丰富的组成部分。这种机制的发展和理解一直是神经科学社区的一个长期研究领域[15,16,28]，并且近年来作为一个强大补充，已经引起了深度神经网络的极大兴趣[20,25]。注意力已经被证明可以改善一系列任务的性能，从图像的定位和理解[3,17]到基于序列的模型[2,24]。它通常结合门功能（例如softmax或sigmoid）和序列技术来实现[11,37]。最近的研究表明，它适用于像图像标题[4,44]和口头阅读[7]等任务，其中利用它来有效地汇集多模态数据。在这些应用中，它通常用在表示较高级别抽象的一个或多个层的顶部，以用于模态之间的适应。高速网络[36]采用门机制来调节快捷连接，使得可以学习非常深的架构。王等人[42]受到语义分割成功的启发，引入了一个使用沙漏模块[27]的强大的trunk-and-mask注意力机制。这个高容量的单元被插入到中间阶段之间的深度残差网络中。相比之下，**我们提出的SE块是一个轻量级的门机制**，专门用于以计算有效的方式对通道关系进行建模，并设计用于增强整个网络中模块的表示能力。


### 3.Squeeze-and-Excitation块

<div align=center>
<img src="zh-cn/img/senet/p3.png" />
</div>


#### 3.1.Squeeze:全局信息嵌入

为了解决利用通道依赖性的问题，我们首先考虑输出特征中每个通道的信号。每个学习到的滤波器都对局部感受野进行操作，因此变换输出U的每个单元都无法利用该区域之外的上下文信息。在网络较低的层次上其感受野尺寸很小，这个问题变得更严重。

<div align=center>
<img src="zh-cn/img/senet/p4.png" />
</div>

讨论。转换输出U可以被解释为局部描述子的集合，这些描述子的统计信息对于整个图像来说是有表现力的。特征工程工作中[31,34,45]普遍使用这些信息。我们选择最简单的全局平均池化，同时也可以采用更复杂的汇聚策略。


#### 3.2.Excitation:自适应重新校正

<div align=center>
<img src="zh-cn/img/senet/p5.png" />
</div>

讨论。激活作为适应特定输入描述符z的通道权重。在这方面，SE块本质上引入了以输入为条件的动态特性，有助于提高特征辨别力。


#### 3.3.模型：SE-Inception和SE-ResNet

<div align=center>
<img src="zh-cn/img/senet/p6.png" />
</div>

<div align=center>
<img src="zh-cn/img/senet/p7.png" />
</div>

*图2.最初的Inception模块架构(左)和SE-Inception模块架构(右)*

<div align=center>
<img src="zh-cn/img/senet/p8.png" />
</div>

<div align=center>
<img src="zh-cn/img/senet/p9.png" />
</div>

*图3.最初的Residual模块架构(左)和SE-ResNet模块架构(右)*


### 4.模型和计算复杂度

SENet通过堆叠一组SE块来构建。实际上，它是通过用原始块的SE对应部分（即SE残差块）替换每个原始块（即残差块）而产生的。我们在表1中描述了SE-ResNet-50和SE-ResNeXt-50的架构。

<div align=center>
<img src="zh-cn/img/senet/p10.png" />
</div>

*表1.(左)ResNet-50，(中)SE-ResNet-50，(右)具有32×4d模板的SE-ResNeXt-50。在括号内列出了残差构建块特定参数设置的形状和操作，并且在外部呈现了一个阶段中堆叠块的数量。fc后面的内括号表示SE模块中两个全连接层的输出维度。*


### 5.实现

在训练过程中，我们遵循标准的做法，使用随机大小裁剪[39]到224×224像素（299×299用于Inception-ResNet-v2[38]和SE-Inception-ResNet-v2）和随机的水平翻转进行数据增强。输入图像通过通道减去均值进行归一化。另外，我们采用[32]中描述的数据均衡策略进行小批量采样，以补偿类别的不均匀分布。网络在我们的分布式学习系统“ROCS”上进行训练，能够处理大型网络的高效并行训练。使用同步SGD进行优化，动量为0.9，小批量数据的大小为1024（在4个服务器的每个GPU上分成32张图像的子批次，每个服务器包含8个GPU）。初始学习率设为0.6，每30个迭代周期减少10倍。使用[8]中描述的权重初始化策略，所有模型都从零开始训练100个迭代周期。

### 6.结论

在本文中，我们提出了SE块，这是一种新颖的架构单元，旨在通过使网络能够执行动态通道特征重新校准来提高网络的表示能力。大量实验证明了SENets的有效性，其在多个数据集上取得了最先进的性能。此外，它们还提供了一些关于以前架构在建模通道特征依赖性上的局限性的洞察，我们希望可能证明SENets对其它需要强判别性特征的任务是有用的。最后，由SE块引起的特征重要性可能有助于相关领域，例如为了压缩的网络修剪。

### Reference

[1] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.

[2] T. Bluche. Joint line segmentation and transcription for end-to-end handwritten paragraph recognition. In NIPS, 2016.

[3] C.Cao, X.Liu, Y.Yang, Y.Yu, J.Wang, Z.Wang, Y.Huang, L. Wang, C. Huang, W. Xu, D. Ramanan, and T. S. Huang. Look and think twice: Capturing top-down visual attention with feedback convolutional neural networks. In ICCV, 2015.

[4] L. Chen, H. Zhang, J. Xiao, L. Nie, J. Shao, W. Liu, and T. Chua. SCA-CNN: Spatial and channel-wise attention in convolutional networks for image captioning. In CVPR, 2017.

[5] Y. Chen, J. Li, H. Xiao, X. Jin, S. Yan, and J. Feng. Dual path networks. arXiv:1707.01629, 2017.

[6] F. Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017.

[7] J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman. Lip reading sentences in the wild. In CVPR, 2017.

[8] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In ICCV, 2015.

[9] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[10] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[11] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 1997.

[12] G. Huang, Z. Liu, K. Q. Weinberger, and L. Maaten. Densely connected convolutional networks. In CVPR, 2017.

[13] Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi. Deep roots: Improving CNN efficiency with hierarchical filter groups. In CVPR, 2017.

[14] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[15] L. Itti and C. Koch. Computational modelling of visual attention. Nature reviews neuroscience, 2001.

[16] L. Itti, C. Koch, and E. Niebur. A model of saliency-based visual attention for rapid scene analysis. IEEE TPAMI, 1998.

[17] M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu. Spatial transformer networks. In NIPS, 2015.

[18] M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. In BMVC, 2014.

[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

[20] H. Larochelle and G. E. Hinton. Learning to combine foveal glimpses with a third-order boltzmann machine. In NIPS, 2010.

[21] H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In ICML, 2009.

[22] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv:1312.4400, 2013.

[23] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[24] A. Miech, I. Laptev, and J. Sivic. Learnable pooling with context gating for video classification. arXiv:1706.06905, 2017.

[25] V. Mnih, N. Heess, A. Graves, and K. Kavukcuoglu. Recurrent models of visual attention. In NIPS, 2014.

[26] V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In ICML, 2010.

[27] A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.

[28] B. A. Olshausen, C. H. Anderson, and D. C. V. Essen. A neurobiological model of visual attention and invariant pattern recognition based on dynamic routing of information. Journal of Neuroscience, 1993.

[29] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[30] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet large scale visual recognition challenge. IJCV, 2015.

[31] J. Sanchez, F. Perronnin, T. Mensink, and J. Verbeek. Image classification with the fisher vector: Theory and practice. RR-8209, INRIA, 2013.

[32] L. Shen, Z. Lin, and Q. Huang. Relay backpropagation for effective learning of deep convolutional neural networks. In ECCV, 2016.

[33] L. Shen, Z. Lin, G. Sun, and J. Hu. Places401 and places365 models. https://github.com/lishen-shirley/ Places2-CNNs, 2016.

[34] L. Shen, G. Sun, Q. Huang, S. Wang, Z. Lin, and E. Wu. Multi-level discriminative dictionary learning with application to large scale image classification. IEEE TIP, 2015.

[35] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[36] R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. In NIPS, 2015.

[37] M. F. Stollenga, J. Masci, F. Gomez, and J. Schmidhuber. Deep networks with internal selective attention through feedback connections. In NIPS, 2014.

[38] C.Szegedy, S.Ioffe, V.Vanhoucke, and A.Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv:1602.07261, 2016.

[39] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[40] C.Szegedy, V.Vanhoucke, S.Ioffe, J.Shlens, and Z.Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[41] A. Toshev and C. Szegedy. DeepPose: Human pose estimation via deep neural networks. In CVPR, 2014.

[42] F. Wang, M. Jiang, C. Qian, S. Yang, C. Li, H. Zhang, X. Wang, and X. Tang. Residual attention network for image classification. In CVPR, 2017.

[43] S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.

[44] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov, R. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. In ICML, 2015.

[45] J. Yang, K. Yu, Y. Gong, and T. Huang. Linear spatial pyramid matching using sparse coding for image classification. In CVPR, 2009.

[46] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson. How transferable are features in deep neural networks? In NIPS, 2014.

[47] X. Zhang, Z. Li, C. C. Loy, and D. Lin. Polynet: A pursuit of structural diversity in very deep networks. In CVPR, 2017.

[48] B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba. Places: A 10 million image database for scene recognition. IEEE TPAMI, 2017.


------
------

## ResNeXt：Aggregated Residual Transformations for Deep Neural Networks - CVPR2017

------

### 1.引言

提出了 ResNeXt 架构，该架构依然采用堆叠构建块的方式构建。构建块内部采用分支结构。分支的数目称为 “基数”，这里提到一个名词cardinality,原文的解释是 the size of the set of transformations,如下面图形中所示，右边是 cardinality=32 的样子。作者们认为，增加分支的数量比增加深度、宽度更高效。ResNeXt 在 ILSVRC 2016 分类比赛中获第二名。作者在 ImageNet-5K分类、COCO探测数据集上进行了实验，性能超过了 ResNet。

作者提出ResNeXt的主要原因在于:传统的要提高模型的准确率，都是通过 加深 或 加宽 网络，但是随着超参数数量的增加(比如 channels数，filter size等等)，网络设计的难度和计算开销也会增加。因此本文提出的 ResNeXt结构可以在不增加参数复杂度的前提下

+ 提高准确率
+ 减少超参数数量(得益于子模块的拓扑结构)

**核心思想**，作者在论文中首先提到VGG，VGG采用 堆叠网络 来实现，之前的 ResNet 也借用了这样的思想。
之后提到了Inception系列网络，简单说就是 split-transform-merge 的策略，但是存在一个问题:
网络的超参数设定的针对性比较强，当应用在别的数据集上需要修改许多参数，因此可扩展性一般.
作者同时采用 VGG 的 堆叠思想 和 Inception 的 split-transform-merge 的思想，但是 可扩展性比较强. 可以认为在增加准确率的同时基本不改变或降低模型的复杂度。

> 这里提到一个名词cardinality,原文的解释是 the size of the set of transformations,如下图 Fig1 右边是 cardinality=32 的样子:

<div align=center>
<img src="zh-cn/img/senet/p11.png" />
</div>

参数计算,假设在不使用偏置的情况下:

```shell
# A block of ResNet 
256x1x64 + 64x3x3x64 + 64x1x256 = 69632
 
# A block of ResNeXt with cardinality
(256x1x4 + 4x4x3x3 + 4x256) x 32 = 70144
```
两者参数数量差不多，但是后面作者有更加精妙的实现。

注意:

+ 每个被聚合的拓扑结构都是一样的(这也是和 Inception 的差别，减轻设计负担)
+ 原文点明了增加 `cardinality` 比增加深度和宽度更有效，这句话的实验结果在后面有展

> Experiments demonstrate that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider.

> In particular, a 101-layer ResNeXt is able to achieve better accuracy than ResNet-200 but has only 50% complexity.


### 2.相关工作

**多分支卷积网络**： Inception 就是精心设计的多分支结构。ResNet 可以被看作一个两分支结构，一个分支是 identity mapping。深度神经决策森林是树状多分支网络，学习多个分离的函数。

**分组卷积**： 分组卷积最早可以追溯到 AlexNet。AlexNet 中分组卷积主要是为了用两块 GPU 来分布式训练。分组卷积的一个特例就是 Channel-wise 卷积。

**压缩卷积网络**： 卷积分解（在空间 and/or 通道层面）是一种常用的卷积网络冗余、加速、压缩网络的常用技术。相比于压缩，作者希望有更强的表示能力。
多模型集成： 对多个单独训练的网络进行平均是一种提高准确率的办法（在识别类比赛中广泛使用）。因为ResNet采用 additive behaviors，有人将 ResNet 理解为 一系列浅层网络 的集成。作者采用 加法 来聚合一系列的变换。但是作者认为将 ResNeXt 看作集成是不准确的，因为各个分支是同时训练的。


### 3.方法

下图列举了 ResNet-50 和 ResNeXt-50 的内部结构，另外最后两行说明二者之间的参数复杂度差别不大。

<div align=center>
<img src="zh-cn/img/senet/p12.png" />
</div>

主要遵从了两个原则：

+ feature map 大小不变时，标准堆叠
+ 当 feature map 的大小减半，则通道数增加一倍


本文提出的新的 block，举全连接层（Inner product）的例子：

<div align=center>
<img src="zh-cn/img/senet/p13.png" />
</div>

<div align=center>
<img src="zh-cn/img/senet/p14.png" />
</div>


下面展示了三种相同的 ResNeXt blocks。（a） 就是前面所说的aggregated residual transformations。 （b） 则采用两层卷积后 concatenate，再卷积，有点类似 Inception-ResNet，只不过这里的 paths 都是相同的拓扑结构。（c）采用的是grouped convolutions，这个 group 参数就是 caffe 的 convolusion 层的 group 参数，用来限制本层卷积核和输入 channels 的卷积，最早应该是 AlexNet 上使用，可以减少计算量。这里（c）采用32个 group，每个 group 的输入输出 channels 都是4，最后把channels合并。这张图的(c)和Fig 1的左边图很像，差别在于(c)的中间 filter 数量（此处为128，而Fig 1中为64）更多。作者在文中明确说明这三种结构是严格等价的，并且用这三个结构做出来的结果一模一样，论文中展示的是（c）的结果，因为（c） 的结构比较简洁而且速度更快。

<div align=center>
<img src="zh-cn/img/senet/p15.png" />
</div>

### 4.实施细节

+ Table 1 中 conv3、conv4、conv5 的下采样过程采用 stride 为 2 的 3x3 卷积。
+ 使用 8 块 GPU 来训练模型
+ 优化器：SGD
+ momentum：0.9
+ batch size：256 （每块 GPU 上 32）
+ weight decay：0.0001
+ 初始学习速率：0.1
+ 学习速率衰减策略同：[11]
+ 测试时从 短边为 256 的图像中裁出一个 224x224 的图像进行测试
+ 代码实现基于 Fig 3(c) ，并且在卷积后加BN+ReLU，在 shortcut 加和只有使用ReLU。Fig 3的三种形式是等价的，之所以选择(c)来实现，因为它更简单、代码更高效。



### Reference

[code-keras](https://github.com/titu1994/Keras-ResNeXt)

[code-tensorflow](https://github.com/taki0112/ResNeXt-Tensorflow)

[code-caffe](https://github.com/soeaver/caffe-model/tree/master/cls/resnext)


## Keras实现SENet-ResNeXt

------

```python
from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense

from keras.layers import Concatenate, concatenate
from keras.layers import Add, add
from keras.layers import Multiply, multiply

from keras import backend as K


class SEResNeXt(object):
    def __init__(self, size=96, num_classes=10, depth=64, reduction_ratio=4, num_split=8, num_block=3):
        self.depth = depth # number of channels
        self.ratio = reduction_ratio # ratio of channel reduction in SE module
        self.num_split = num_split # number of splitting trees for ResNeXt (so called cardinality)
        self.num_block = num_block # number of residual blocks
        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3
        self.model = self.build_model(Input(shape=(size,size,3)), num_classes)

    def conv_bn(self, x, filters, kernel_size, stride, padding='same'):
        '''
        Combination of Conv and BN layers since these always appear together.
        '''
        x = Conv2D( filters=filters, kernel_size=[kernel_size, kernel_size]
                               , strides=[stride, stride], padding=padding )(x)
        x = BatchNormalization()(x)
        
        return x
    
    def activation(self, x, func='relu'):
        '''
        Activation layer.
        '''
        return Activation(func)(x)
    
    def channel_zeropad(self, x):
        '''
        Zero-padding for channle dimensions.
        Note that padded channles are added like (Batch, H, W, 2/x + x + 2/x).
        '''
        shape = list(x.shape)
        y = K.zeros_like(x)
        
        if self.channel_axis == 3:
            y = y[:, :, :, :shape[self.channel_axis]//2]
        else:
            y = y[:, :shape[self.channel_axis]//2, :, :]
        
        return concatenate([y, x, y], self.channel_axis)
    
    def channel_zeropad_output(self, input_shape):
        '''
        Function for setting a channel dimension for zero padding.
        '''
        shape = list(input_shape)
        shape[self.channel_axis] *= 2

        return tuple(shape)
    
    def initial_layer(self, inputs):
        '''
        Initial layers includes {conv, BN, relu}.
        '''
        x = self.conv_bn(inputs, self.depth, 3, 1)
        x = self.activation(x)
        
        return x
    
    def transform_layer(self, x, stride):
        '''
        Transform layer has 2 {conv, BN, relu}.
        '''
        x = self.conv_bn(x, self.depth, 1, 1)
        x = self.activation(x)
        
        x = self.conv_bn(x, self.depth, 3, stride)
        x = self.activation(x)
        
        return x
        
    def split_layer(self, x, stride):
        '''
        Parallel operation of transform layers for ResNeXt structure.
        '''
        splitted_branches = list()
        for i in range(self.num_split):
            branch = self.transform_layer(x, stride)
            splitted_branches.append(branch)
        
        return concatenate(splitted_branches, axis=self.channel_axis)
    
    def squeeze_excitation_layer(self, x, out_dim):
        '''
        SE module performs inter-channel weighting.
        '''
        squeeze = GlobalAveragePooling2D()(x)
        
        excitation = Dense(units=out_dim // self.ratio)(squeeze)
        excitation = self.activation(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = self.activation(excitation, 'sigmoid')
        excitation = Reshape((1,1,out_dim))(excitation)
        
        scale = multiply([x,excitation])
        
        return scale
    
    def residual_layer(self, x, out_dim):
        '''
        Residual block.
        '''
        for i in range(self.num_block):
            input_dim = int(np.shape(x)[-1])
            
            if input_dim*2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            
            subway_x = self.split_layer(x, stride)
            subway_x = self.conv_bn(subway_x, out_dim, 1, 1)
            subway_x = self.squeeze_excitation_layer(subway_x, out_dim)
            
            if flag:
                pad_x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
                pad_x = Lambda(self.channel_zeropad, output_shape=self.channel_zeropad_output)(pad_x)
            else:
                pad_x = x
            
            x = self.activation(add([pad_x, subway_x]))
                
        return x
    
    def build_model(self, inputs, num_classes):
        '''
        Build a SENet model.
        '''
        x = self.initial_layer(inputs)
        
        x = self.residual_layer(x, out_dim=64)
        x = self.residual_layer(x, out_dim=128)
        x = self.residual_layer(x, out_dim=256)
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=num_classes, activation='softmax')(x)
        
        return Model(inputs, x)
```
