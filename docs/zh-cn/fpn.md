## FPN:Feature Pyramid Networks for Object Detection

论文地址：https://arxiv.org/abs/1612.03144

### 1.前言与摘要

这篇论文主要使用特征金字塔网络来融合多层特征，改进了CNN特征提取。论文在Fast/Faster R-CNN上进行了实验，在COCO数据集上刷到了第一的位置，意味着其在小目标检测上取得了很大的进步。论文整体思想比较简单，但是实验部分非常详细和充分。

特征金字塔是多尺度目标检测系统中的一个基本组成部分。近年来深度学习目标检测却有意回避这一技巧，部分原因是特征金字塔在计算量和用时上很敏感（一句话，太慢）。这篇文章，作者利用了深度卷积神经网络固有的多尺度、多层级的金字塔结构去构建特征金字塔网络。使用一种自上而下的侧边连接，在所有尺度构建了高级语义特征图，这种结构就叫特征金字塔网络（FPN）。其在特征提取上改进明显，把FPN用在Faster R-CNN上，在COCO数据集上，一举超过了目前所有的单模型（single-model）检测方法，而且在GPU上可以跑到5帧.

### 2.概述

多尺度目标检测是计算机视觉领域的一个基础且具挑战性的课题。在图像金字塔基础上构建的特征金字塔（featurized image pyramids ,Figure1[a]）是传统解决思路，具有一定意义的尺度不变性。直观上看，这种特性使得模型可以检测大范围尺度的图像。

Featurized image pyramids 主要在人工特征中使用，比如DPM就要用到它产生密集尺度的样本以提升检测水平。目前人工特征式微，深度学习的CNN特征成为主流，CNN特征的鲁棒性很好，刻画能力强。即使如此，仍需要金字塔结构去进一步提升准确性，尤其在多尺度检测上。金字塔结构的优势是其产生的特征每一层都是语义信息加强的，包括高分辨率的低层。

对图像金字塔每一层都处理有很大的局限性，首先运算耗时会增加4倍，训练深度网络的时候太吃显存，几乎没法用，即使用了，也只能在检测的时候。因为这些原因，Fast/Faster R-CNN 都没使用featurized image pyramids。

当然，图像金字塔并不是多尺度特征表征的唯一方式，CNN计算的时候本身就存在多级特征图（feature map hierarchy），且不同层的特征图尺度就不同，形似金字塔结构（Figure1[b]）。结构上虽不错，但是前后层之间由于不同深度（depths）影响，语义信息差距太大，主要是高分辨率的低层特征很难有代表性的检测能力。

SSD方法在借鉴利用featurized image pyramid上很是值得说，为了避免利用太低层的特征，SSD从偏后的conv4_3开始，又往后加了几层，分别抽取每层特征，进行综合利用（Figure1[c]）。但是SSD对于高分辨率的底层特征没有再利用，而这些层对于检测小目标很重要。 

<div align=center>
<img src="zh-cn/img/fpn/p1.png" />
</div>

这篇论文的特征金字塔网络（Figure1[d]）做法很简单，如下图所示。把低分辨率、高语义信息的高层特征和高分辨率、低语义信息的低层特征进行自上而下的侧边连接，使得所有尺度下的特征都有丰富的语义信息。这种结构是在CNN网络中完成的，和前文提到的基于图片的金字塔结构不同，而且完全可以替代它。

<div align=center>
<img src="zh-cn/img/fpn/p2.png" />
</div>

上图展示了4种利用特征的形式：

（a）图像金字塔，即将图像做成不同的scale，然后不同scale的图像生成对应的不同scale的特征。这种方法的缺点在于增加了时间成本。有些算法会在测试时候采用图像金字塔。

（b）像SPP net，Fast RCNN，Faster RCNN是采用这种方式，即仅采用网络最后一层的特征。

（c）像SSD（Single Shot Detector）采用这种多尺度特征融合的方式，没有上采样过程，即从网络不同层抽取不同尺度的特征做预测，这种方式不会增加额外的计算量。作者认为SSD算法中没有用到足够低层的特征（在SSD中，最低层的特征是VGG网络的conv4_3），而在作者看来足够低层的特征对于检测小物体是很有帮助的。

（d）本文作者是采用这种方式，顶层特征通过上采样和低层特征做融合，而且每层都是独立预测的。


### 3.特征金字塔网络(FPN)

论文的目标是利用CNN的金字塔层次结构特性（具有从低到高级的语义），构建具有高级语义的特征金字塔。得到的特征金字塔网络（FPN）是通用的，但在论文中，作者先在RPN网络和Fast R-CNN中使用这一成果，也将其用在instance segmentation proposals 中。

该方法将任意一张图片作为输入，以全卷积的方式在多个层级输出成比例大小的特征图，这是独立于CNN骨干架构（本文为ResNets）的。具体结构如图Figure2。 

<div align=center>
<img src="zh-cn/img/fpn/p3.png" />
</div>

**自下而上的路径**： CNN的前馈计算就是自下而上的路径，特征图经过卷积核计算，通常是越变越小的，也有一些特征层的输出和原来大小一样，称为“相同网络阶段”（same network stage ）。对于本文的特征金字塔，作者为每个阶段定义一个金字塔级别， 然后选择每个阶段的最后一层的输出作为特征图的参考集。 这种选择是很自然的，因为每个阶段的最深层应该具有最强的特征。具体来说，对于ResNets，作者使用了每个阶段的最后一个残差结构的特征激活输出。将这些残差模块输出表示为`{C2, C3, C4, C5}`，对应于conv2，conv3，conv4和conv5的输出，并且注意它们相对于输入图像具有`{4, 8, 16, 32}`像素的步长。考虑到内存占用，没有将conv1包含在金字塔中。

**自上而下的路径和横向连接**： 自上而下的路径（the top-down pathway ）是如何去结合低层高分辨率的特征呢？方法就是，把更抽象，语义更强的高层特征图进行**上采样**，然后把该特征横向连接（lateral connections ）至前一层特征，因此高层特征得到加强。值得注意的是，横向连接的两层特征在空间尺寸上要相同。这样做应该主要是为了利用底层的定位细节信息。

Figure 3显示连接细节。把高层特征做2倍上采样（最邻近上采样法），然后将其和对应的前一层特征结合（前一层要经过`1 * 1`的卷积核才能用，目的是改变channels，应该是要和后一层的channels相同），**横向结合方式就是做像素间的加法**。重复迭代该过程，直至生成最精细的特征图。迭代开始阶段，作者在C5层后面加了一个`1 * 1`的卷积核来产生最粗略的特征图，最后，作者用`3 * 3`的卷积核去处理已经融合的特征图（为了消除上采样的混叠效应），以生成最后需要的特征图。`{C2, C3, C4, C5}`层对应的融合特征层为`{P2, P3, P4, P5}`，对应的层空间尺寸是相通的。 

<div align=center>
<img src="zh-cn/img/fpn/p4.png" />
</div>

*图3 构建模块说明了横向连接和自顶向下路径，通过加法合并*

金字塔结构中所有层级共享分类层（回归层），就像featurized image pyramid 中所做的那样。作者固定所有特征图中的维度（通道数，表示为d）。作者在本文中设置`d = 256`，因此所有额外的卷积层（比如P2）具有256通道输出。 这些额外层没有用非线性（博主：不知道具体所指），而非线性会带来一些影响。


**FPN的两种构架**

*第一种为megred rcnn：*


<div align=center>
<img src="zh-cn/img/fpn/p10.png" />
</div>

*第二种为shared rcnn：*

<div align=center>
<img src="zh-cn/img/fpn/p11.png" />
</div>


### 4.实际应用

本文方法在理论上在CNN中是通用的，作者将其首先应用到了RPN和Fast R-CNN中，应用中尽量做较小幅度的修改。

#### 1.Faster R-CNN+Resnet-101

要想明白FPN如何应用在RPN和Fast R-CNN（合起来就是Faster R-CNN），首先要明白Faster R-CNN+Resnet-101的结构，这部分在是论文中没有的。

直接理解就是把Faster-RCNN中原有的VGG网络换成ResNet-101，ResNet-101结构如下图：

<div align=center>
<img src="zh-cn/img/fpn/p5.png" />
</div>

<div align=center>
<img src="zh-cn/img/fpn/p7.jpg" />
</div>

Faster-RCNN利用`conv1`到`onv4_x`的91层为共享卷积层，然后从`conv4_x`的输出开始分叉，一路经过`RPN`网络进行区域选择，另一路直接连一个`ROI Pooling`层，把`RPN`的结果输入`ROI Pooling`层，映射成`7*7`的特征。然后所有输出经过`conv5_x`的计算，这里`conv5_x`起到原来全连接层（fc）的作用。最后再经分类器和边框回归得到最终结果。整体框架用下图表示： 

<div align=center>
<img src="zh-cn/img/fpn/p6.png" />
</div>

#### 2.RPN中的特征金字塔网络

RPN是Faster R-CNN中用于区域选择的子网络(具体原理可以参考本书中的Faster R-CNN网络)。RPN是在一个`13*13*256`的特征图上应用9种不同尺度的anchor，本篇论文另辟蹊径，把特征图弄成多尺度的，然后固定每种特征图对应的anchor尺寸，很有意思。也就是说，`作者在每一个金字塔层级应用了单尺度的anchor`，`{P2, P3, P4, P5, P6}`分别对应的anchor尺度为$\{32^2, 64^2, 128^2, 256^2, 512^2 \}$，当然目标不可能都是正方形，本文仍然使用三种比例`{1:2, 1:1, 2:1}`，所以金字塔结构中共有`15`种anchors。这里,我们尝试画一下修改后的RPN结构：

<div align=center>
<img src="zh-cn/img/fpn/p8.png" />
</div>

训练中，把重叠率（IoU）高于0.7的作为正样本，低于0.3的作为负样本。特征金字塔网络之间有参数共享，其优秀表现使得所有层级具有相似程度的语义信息。具体性能在实验中评估。

每个level的feature P2，P3，P4，P5，P6只对应一种scale，比例还是3个比例


<div align=center>
<img src="zh-cn/img/fpn/p9.png" />
</div>



#### 3.Fast R-CNN 中的特征金字塔网络

Fast R-CNN的具体原理可以参考本文Fast R-CNN ，其中很重要的是ROI Pooling层，需要对不同层级的金字塔制定不同尺度的ROI。

这里要把视角转换一下，想象成有一种图片金字塔在起作用。我们知道，ROI Pooling层使用region proposal的结果和中间的某一特征图作为输入，得到的结果经过分解后分别用于分类结果和边框回归。

然后作者想的是，不同尺度的ROI，使用不同特征层作为ROI pooling层的输入，大尺度ROI就用后面一些的金字塔层，比如P5；小尺度ROI就用前面一点的特征层，比如P4。那怎么判断ROI改用那个层的输出呢？这里作者定义了一个系数Pk，其定义为

$$k = \[k_0 +\log_2(\sqrt{wh}/224\]$$

224是ImageNet的标准输入，$k_0$是基准值，设置为5，代表P5层的输出（原图大小就用P5层），w和h是ROI区域的长和宽，假设ROI是`112*112`的大小，那么$k = k_0-1 = 5-1 = 4$，意味着该ROI应该使用P4的特征层。k值应该会做取整处理，防止结果不是整数。

然后，因为作者把conv5也作为了金字塔结构的一部分，那么从前全连接层的那个作用怎么办呢？这里采取的方法是增加两个1024维的轻量级全连接层，然后再跟上分类器和边框回归。作者认为这样还能使速度更快一些。草图如下：

<div align=center>
<img src="zh-cn/img/fpn/p12.png" />
</div>


### 5.为什么FPN能够很好的处理小目标

<div align=center>
<img src="zh-cn/img/fpn/p13.png" />
</div>

*FPN处理小目标*

如上图所示，FPN能够很好地处理小目标的主要原因是：

- FPN可以利用经过top-down模型后的那些上下文信息（高层语义信息）；
- 对于小目标而言，FPN增加了特征映射的分辨率（即在更大的feature map上面进行操作，这样可以获得更多关于小目标的有用信息），如图中所示；

<div align=center>
<img src="zh-cn/img/fpn/p14.png" />
</div>

*FPN实例分割结果*


### 6.Reference

References
[1] E. H. Adelson, C. H. Anderson, J. R. Bergen, P. J. Burt, and J. M. Ogden. Pyramid methods in image processing. RCA engineer, 1984.

[2] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.

[3] Z. Cai, Q. Fan, R. S. Feris, and N. Vasconcelos. A unified multi-scale deep convolutional neural network for fast object detection. In ECCV, 2016.

[4] J. Dai, K. He, Y. Li, S. Ren, and J. Sun. Instance-sensitive fully convolutional networks. In ECCV, 2016.

[5] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005.

[6] P. Dollar, R. Appel, S. Belongie, and P. Perona. Fast feature pyramids for object detection. TPAMI, 2014.

[7] P.F.Felzenszwalb,R.B.Girshick,D.McAllester,andD.Ramanan. Object detection with discriminatively trained part-based models. TPAMI, 2010.

[8] G.GhiasiandC.C.Fowlkes.Laplacianpyramidreconstruction and refinement for semantic segmentation. In ECCV, 2016.

[9] S. Gidaris and N. Komodakis. Object detection via a multi-region & semantic segmentation-aware CNN model. In ICCV, 2015.

[10] S. Gidaris and N. Komodakis. Attend refine repeat: Active box proposal generation via in-out localization. In BMVC, 2016.

[11] R. Girshick. Fast R-CNN. In ICCV, 2015.

[12] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[13] B.Hariharan,P.Arbelaez,R.Girshick,andJ.Malik.Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.

[14] K. He, G. Gkioxari, P. Dollar, and R. Girshick. Mask r-cnn. arXiv:1703.06870, 2017.

[15] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV. 2014.

[16] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[17] S. Honari, J. Yosinski, P. Vincent, and C. Pal. Recombinator networks: Learning coarse-to-fine feature aggregation. In CVPR, 2016.

[18] T. Kong, A. Yao, Y. Chen, and F. Sun. Hypernet: Towards accurate region proposal generation and joint object detection. In CVPR, 2016.

[19] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

[20] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. 
Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[21] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dolla虂r, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[22] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed. SSD: Single shot multibox detector. In ECCV, 2016.

[23] W. Liu, A. Rabinovich, and A. C. Berg. ParseNet: Looking wider to see better. In ICLR workshop, 2016.

[24] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[25] D. G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 2004.

[26] A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.

[27] P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to segment object candidates. In NIPS, 2015.

[28] P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Dolla虂r. Learning to refine object segments. In ECCV, 2016.

[29] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[30] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. PAMI, 2016.

[31] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional networks for biomedical image segmentation. In MIC- CAI, 2015.

[32] H. Rowley, S. Baluja, and T. Kanade. Human face detection in visual scenes. Technical Report CMU-CS-95-158R, Carnegie Mellon University, 1995.

[33] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

[34] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.

[35] A. Shrivastava, A. Gupta, and R. Girshick. Training region-based object detectors with online hard example mining. In CVPR, 2016.

[36] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[37] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders. Selective search for object recognition. IJCV, 2013.

[38] R. Vaillant, C. Monrocq, and Y. LeCun. Original approach for the localisation of objects in images. IEE Proc. on Vision, Image, and Signal Processing, 1994.

[39] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.

[40] S. Zagoruyko, A. Lerer, T.-Y. Lin, P. O. Pinheiro, S. Gross, S. Chintala, and P. Dolla虂r. A multipath network for object detection. In BMVC, 2016. 10.
