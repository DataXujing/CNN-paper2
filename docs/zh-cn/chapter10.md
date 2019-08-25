## 图像分割

在计算机视觉中，图像分割是个非常重要且基础的研究方向。简单来说，图像分割（image segmentation）就是根据某些规则把图片中的像素分成不同的部分（加不同的标签）。

图像分割中的一些常见的术语有：superpixels（超像素）、Semantic Segmentation（语义分割）、Instance Segmentation（实例分割）、Panoptic Segmentation（全景分割）。他们之间到底有什么区别呢？

**superpixels（超像素）**： 第一次听说这个超像素很容易理解错误，以为是在普通的像素基础上继续像微观细分，如果这样理解就恰好理解反了，其实超像素是一系列像素的集合，这些像素具有类似的颜色、纹理等特征，距离也比较近。用超像素对一张图片进行分割的结果见下图，其中每个白色线条区域内的像素集合就是一个超像素。需要注意的是，超像素很可能把同一个物体的不同部分分成多个超像素。

<div align=center>
<img src="zh-cn/img/chapter10/p1.gif" />
</div>

超像素最早的定义来自2003年 Xiaofeng Ren等人的一篇论文《Learning a Classification Model for Segmentation》。

其中超像素中比较常用的一种方法是SLIC（simple linear iterative clustering），是Achanta 等人2010年提出的一种思想简单、实现方便的算法，将彩色图像转化为CIELAB颜色空间和XY坐标下的5维特征向量，然后对5维特征向量构造距离度量标准，对图像像素进行局部聚类的过程。SLIC算法能生成紧凑、近似均匀的超像素，在运算速度，物体轮廓保持、超像素形状方面具有较高的综合评价，比较符合人们期望的分割效果。

<div align=center>
<img src="zh-cn/img/chapter10/p2.png" />
</div>

**Semantic Segmentation（语义分割）**：语义分割还是比较常见的，就是把图像中每个像素赋予一个类别标签（比如汽车、建筑、地面、天空等），比如下图就把图像分为了草地（浅绿）、人（红色）、树木（深绿）、天空（蓝色）等标签，用不同的颜色来表示。

不过这种分割方式存在一些问题，比如如果一个像素被标记为红色，那就代表这个像素所在的位置是一个人，但是如果有两个都是红色的像素，这种方式无法判断它们是属于同一个人还是不同的人。也就是说语义分割只能判断类别，无法区分个体。

<div align=center>
<img src="zh-cn/img/chapter10/p3.jpg" />
</div>

但很多时候我们更需要个体信息，想要区分出个体怎么办呢？

**Instance Segmentation（实例分割）**:实例分割方式有点类似于物体检测，不过物体检测一般输出的是 bounding box，实例分割输出的是一个mask。

实例分割和上面的语义分割也不同，它不需要对每个像素进行标记，它只需要找到感兴趣物体的边缘轮廓就行，比如下图中的人就是感兴趣的物体。该图的分割方法采用了一种称为Mask R-CNN的方法。我们可以看到每个人都是不同的颜色的轮廓，因此我们可以区分出单个个体。

<div align=center>
<img src="zh-cn/img/chapter10/p4.jpg" />
</div>

**Panoptic Segmentation（全景分割）**:最后说说全景分割，它是语义分割和实例分割的结合。如下图所示，每个像素都被分为一类，如果一种类别里有多个实例，会用不同的颜色进行区分，我们可以知道哪个像素属于哪个类中的哪个实例。比如下图中黄色和红色都属于人这一个类别里，但是分别属于不同的实例（人），因此我们可以通过mask的颜色很容易分辨出不同的实例。

<div align=center>
<img src="zh-cn/img/chapter10/p5.jpg" />
</div>


+ 超像素（superpixels）

+ 语义分割（semantic segmentation）

+ 实例分割（instance segmentation）

+ 全景分割（panoptic segmentation）

相关论文可以参考：<https://github.com/mrgloom/awesome-semantic-segmentation>


------

### 1.语义分割

主要有：U-Net、SegNet、DeepLab系列、FCN、ENet、ICNet、ShelfNet、BiseNet、DFN和CCNet等网络，我们这里着重介绍**FCN,U-Net和DeepLab系列**

### FCN:Fully Convolutional Networks for Semantic Segmentation

论文地址：https://arxiv.org/abs/1411.4038

#### 1.核心思想与预备知识

该论文包含了当下CNN的三个思潮：

- 不含全连接层(fc)的全卷积(fully conv)网络。可适应任意尺寸输入。 
- 增大数据尺寸的反卷积(deconv转置卷积)层。能够输出精细的结果。 
- 结合不同深度层结果的跳级(skip)结构。同时确保鲁棒性和精确性。
- 损失函数是在最后一层的 spatial map上的 pixel 的 loss 和，在每一个 pixel 使用 softmax loss 
使用 skip 结构融合多层（3层）输出，底层网络应该可以预测更多的位置信息，因为他的感受野小可以看到小的 pixels上采样 lower-resolution layers 时，如果采样后的图因为 padding 等原因和前面的图大小不同，使用 crop ，当裁剪成大小相同的，spatially aligned ，使用 concat 操作融合两个层 

CNN与FCN:

- 通常cnn网络在卷积之后会接上若干个全连接层，将卷积层产生的特征图（feature map）映射成为一个固定长度的特征向量。一般的CNN结构适用于图像级别的分类和回归任务，因为它们最后都期望得到输入图像的分类的概率，如ALexNet网络最后输出一个1000维的向量表示输入图像属于每一类的概率。
- FCN对图像进行像素级的分类，从而解决了语义级别的图像分割问题。与经典的CNN在卷积层使用全连接层得到固定长度的特征向量进行分类不同，FCN可以接受任意尺寸的输入图像，采用反卷积层对最后一个卷基层的特征图（feature map）进行上采样，使它恢复到输入图像相同的尺寸，从而可以对每一个像素都产生一个预测，同时保留了原始输入图像中的空间信息，最后奇偶在上采样的特征图进行像素的分类。
- 全卷积网络(FCN)是从抽象的特征中恢复出每个像素所属的类别。即从图像级别的分类进一步延伸到像素级别的分类。
- FCN将传统CNN中的全连接层转化成一个个的卷积层。如下图所示，在传统的CNN结构中，前5层是卷积层，第6层和第7层分别是一个长度为4096的一维向量，第8层是长度为1000的一维向量，分别对应1000个类别的概率。FCN将这3层表示为卷积层，卷积核的大小(通道数，宽，高)分别为`(4096,1,1)`,`(4096,1,1)`,`(1000,1,1)`. 所有的层都是卷积层，故称为全卷积网络。 

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p1.png" />
</div>


#### 2.网络结构

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p3.jpg" />
</div>

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p2.jpg" />
</div>

网络结构如上。输入可为任意尺寸图像彩色图像；输出与输入尺寸相同，深度为：`20类目标+背景=21`(在PASCAL数据集上进行的，PASCAL一共20类)

#### 3.全卷积-提取特征

如第2部分上图所示，虚线上半部分为全卷积网络。(`蓝：卷积`，`绿：max pooling`). 对于不同尺寸的输入图像，各层数据的尺寸`(height，width)`相应变化，深度（channel）不变。 
这部分由深度学习分类问题中经典网络AlexNet修改而来。只不过，把最后两个全连接层（fc）改成了卷积层。论文中，达到最高精度的分类网络是VGG16，但提供的模型基于AlexNet,此处使用AlexNet便于绘图。

全连接层转换为卷积层：在两种变换中，将全连接层转化为卷积层在实际运用中更加有用。假设一个卷积神经网络的输入是 `224x224x3`的图像，一系列的卷积层和下采样层将图像数据变为尺寸为 `7x7x512` 的激活数据体。AlexNet使用了两个尺寸为`4096`的全连接层，最后一个有`1000`个神经元的全连接层用于计算分类评分。我们可以将这3个全连接层中的任意一个转化为卷积层：
针对第一个连接区域是[7x7x512]的全连接层，令其滤波器尺寸为F=7，这样输出数据体就为`[1x1x4096]`了。针对第二个全连接层，令其滤波器尺寸为`F=1`，这样输出数据体为`[1x1x4096]`。对最后一个全连接层也做类似的，令其`F=1`，最终输出为`[1x1x1000]`


#### 4.逐像素预测

虚线下半部分中，分别从卷积网络的不同阶段，以卷积层（蓝色×3）预测深度为21的分类结果。

```
例：第一个预测模块 
输入16*16*4096，卷积模板尺寸1*1，输出16*16*21。 
```
相当于对每个像素施加一个全连接层，从4096维特征，预测21类结果。

采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。具体过程：

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p4.jpg" />
</div>

经过多次卷积和pooling以后，得到的图像越来越小，分辨率越来越低。其中图像到 

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p5.jpg" />
</div>

最后的输出是21张heatmap经过`upsampling`变为原图大小的图片，为了对每个像素进行分类预测label成最后已经进行语义分割的图像，这里有一个小trick，就是最后通过逐个像素地求其在21张图像该像素位置的最大数值描述（概率）作为该像素的分类。因此产生了一张已经分类好的图片，如下图右侧有狗狗和猫猫的图。


#### 5.UnPooling Unsampling和Deconvolution的区别

> unpooling是填充0，unsampling是填充相同的值

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p6.png" />
</div>

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p7.gif" />
</div>

上图是full卷积，full卷积：`输入(蓝色2*2大小的图形)为N1*N1,卷积核（灰色的3*3）大小为N2*N2，卷积后图像大小为N1+N2-1（绿色4*4）`

图像的deconvolution实现过程：

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p8.png" />
</div>

输入：`2X2`， 卷积核`4X4`，滑动步长：`3`，输出`7X7`

输入与输出的关系为：`outputSize = (input - 1) * stride + kernel_size`

1.先对每一个像素做full卷积，卷积后输出大小为`1+4-1=4`，得到`4*4`大小的特征图（`2*2大小分开卷积，相当于4个1*1的图形做卷积`）

2.对四个特征图进行步长为3的fusion(相加)，步长为3是指每隔3个像素进行fusion，重叠部分进行相加，即输出的第一行第四列是由红色特征图的第一行第四列与绿色特征图第一行第一列相加得到，其他的依此类推

#### 6.反卷积(转置卷积)说明

反卷积(deconvolutional)运算的参数和CNN的参数一样是在训练FCN模型的过程中通过BP算法学习得到。反卷积层也是卷积层，不关心input大小，滑窗卷积后输出output。deconv并不是真正的deconvolution（卷积的逆变换），最近比较公认的叫法应该是transposed convolution，deconv的前向传播就是conv的反向传播。反卷积参数: 利用卷积过程filter的转置（先水平翻转，再竖直方向上翻转filter）作为计算卷积前的特征图。蓝色是反卷积层的input，绿色是output

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p9.png" />
</div>

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p10.png" />
</div>

#### 7.跳级(skip)结构

对CNN的结果做处理，得到了dense prediction，而作者在试验中发现，得到的分割结果比较粗糙，所以考虑加入更多前层的细节信息，也就是把倒数第几层的输出和最后的输出做一个fusion，就是对应元素相加

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p11.png" />
</div>

- 如上图所示：对原图进行卷积conv1、pool1后图像缩小为1/2；对图像进行第二次卷积conv2、pool2后图像缩小为1/4；对图像进行第三次卷积conv3、pool3后图像缩小为1/8，此时保留pool3的featuremap；对图像进行第四次卷积conv4、pool4后图像缩小为1/16，此时保留pool4的featuremap；对图像进行第五次卷积conv5、pool5后图像缩小为1/32，然后把原来CNN操作过程中的全连接编程卷积操作的conv6、conv7，图像的featuremap的大小依然为原图的1/32,此时图像不再叫featuremap而是叫heatmap。

- 其实直接使用前两种结构就已经可以得到结果了，这个上采样是通过反卷积（deconvolution）实现的，对第五层的输出（32倍放大）反卷积到原图大小。但是得到的结果还不够精确，一些细节无法恢复。于是将第四层的输出和第三层的输出也依次反卷积，分别需要16倍和8倍上采样，结果过也更精细一些了。这种做法的好处是兼顾了local和global信息。



#### 8.训练

训练过程分为四个阶段，也体现了作者的设计思路，值得研究。

*第1阶段*

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p13.png" />
</div>

以经典的分类网络为初始化。最后两级是全连接（红色），参数弃去不用。

*第2阶段*

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p14.png" />
</div>

从特征小图（`16*16*4096`）预测分割小图（`16*16*21`），之后直接升采样为大图。 
反卷积（橙色）的步长为`32`，这个网络称为`FCN-32s`。 
这一阶段使用单`GPU`训练约需`3`天。

*第3阶段*

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p15.png" />
</div>

升采样分为两次完成（`橙色×2`）。 
在第二次升采样前，把第4个pooling层（绿色）的预测结果（蓝色）融合进来。使用跳级结构提升精确性。 
第二次反卷积步长为`16`，这个网络称为`FCN-16s`, 
这一阶段使用单`GPU`训练约需`1`天。

*第4阶段*

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p16.png" />
</div>

升采样分为三次完成（`橙色×3`）。 
进一步融合了第3个pooling层的预测结果。 
第三次反卷积步长为`8`，记为`FCN-8s`。 
这一阶段使用单`GPU`训练约需`1`天。

较浅层的预测结果包含了更多细节信息。比较`2,3,4`阶段可以看出，跳级结构利用浅层信息辅助逐步升采样，有更精细的结果。 

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p17.png" />
</div>

其他参数:`minibatch：20张图片`,`learning rate：0.001`,`初始化： 分类网络之外的卷积层参数初始化为0`,`
反卷积参数初始化为bilinear插值`,`最后一层反卷积固定位bilinear插值不做学习`

#### 9.FCN的缺点

（1）得到的结果还是不够精细。进行8倍上采样虽然比32倍的效果好了很多，但是上采样的结果还是比较模糊和平滑，对图像中的细节不敏感。

（2）对各个像素进行分类，没有充分考虑像素与像素之间的关系。忽略了在通常的基于像素分类的分割方法中使用的空间规整（spatial regularization）步骤，缺乏空间一致性。

<div align=center>
<img src="zh-cn/img/chapter10/FCN/p12.png" />
</div>

#### 10.参考文献
[1] C. M. Bishop. Pattern recognition and machine learning,page 229. Springer-Verlag New York, 2006. 6

[2] J. Carreira, R. Caseiro, J. Batista, and C. Sminchisescu. Semantic segmentation with second-order pooling. In ECCV,2012. 9

[3] D. C. Ciresan, A. Giusti, L. M. Gambardella, and J. Schmidhuber.Deep neural networks segment neuronal membranes in electron microscopy images. In NIPS, pages 2852–2860,2012. 1, 2, 4, 7

[4] J. Dai, K. He, and J. Sun. Convolutional feature masking for joint object and stuff segmentation. arXiv preprint arXiv:1412.1283, 2014. 9

[5] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang,E. Tzeng, and T. Darrell. DeCAF: A deep convolutional activation feature for generic visual recognition. In ICML, 2014.1, 2

[6] D. Eigen, D. Krishnan, and R. Fergus. Restoring an image taken through a window covered with dirt or rain. In Computer Vision (ICCV), 2013 IEEE International Conference on, pages 633–640. IEEE, 2013. 2

[7] D. Eigen, C. Puhrsch, and R. Fergus. Depth map prediction from a single image using a multi-scale deep network. arXiv preprint arXiv:1406.2283, 2014. 2

[8] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2011 (VOC2011) Results. 

[9] C. Farabet, C. Couprie, L. Najman, and Y. LeCun. Learning hierarchical features for scene labeling. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 2013. 1, 2, 4,7, 8

[10] P. Fischer, A. Dosovitskiy, and T. Brox. Descriptor matching with convolutional neural networks: a comparison to SIFT.CoRR, abs/1405.5769, 2014. 1

[11] Y. Ganin and V. Lempitsky. N4-fields: Neural network nearest neighbor fields for image transforms. In ACCV, 2014. 1,2, 7

[12] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Computer Vision and Pattern Recognition,2014. 1, 2, 7

[13] A. Giusti, D. C. Cires¸an, J. Masci, L. M. Gambardella, and J. Schmidhuber. Fast image scanning with deep max-pooling convolutional neural networks. In ICIP, 2013. 3, 4

[14] S. Gupta, P. Arbelaez, and J. Malik. Perceptual organization and recognition of indoor scenes from RGB-D images. In CVPR, 2013. 8

[15] S. Gupta, R. Girshick, P. Arbelaez, and J. Malik. Learning rich features from RGB-D images for object detection and segmentation. In ECCV. Springer, 2014. 1, 2, 8

[16] B. Hariharan, P. Arbelaez, L. Bourdev, S. Maji, and J. Malik.Semantic contours from inverse detectors. In International Conference on Computer Vision (ICCV), 2011. 7

[17] B. Hariharan, P. Arbel´aez, R. Girshick, and J. Malik. Simultaneous detection and segmentation. In European Conference on Computer Vision (ECCV), 2014. 1, 2, 4, 5, 7, 8

[18] B. Hariharan, P. Arbel´aez, R. Girshick, and J. Malik. Hypercolumns for object segmentation and fine-grained localization.In Computer Vision and Pattern Recognition, 2015.2

[19] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV, 2014. 1, 2

[20] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick,S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv preprint

arXiv:1408.5093, 2014. 7

[21] J. J. Koenderink and A. J. van Doorn. Representation of local geometry in the visual system. Biological cybernetics,55(6):367–375, 1987. 6

[22] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012. 1, 2, 3, 5

[23] Y. LeCun, B. Boser, J. Denker, D. Henderson, R. E. Howard,W. Hubbard, and L. D. Jackel. Backpropagation applied to hand-written zip code recognition. In Neural Computation,1989. 2, 3

[24] Y. A. LeCun, L. Bottou, G. B. Orr, and K.-R. M¨uller. Efficient backprop. In Neural networks: Tricks of the trade,pages 9–48. Springer, 1998. 7

[25] C. Liu, J. Yuen, and A. Torralba. Sift flow: Dense correspondence across scenes and its applications. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 33(5):978–994, 2011.8

[26] J. Long, N. Zhang, and T. Darrell. Do convnets learn correspondence?In NIPS, 2014. 1

[27] S. Mallat. A wavelet tour of signal processing. Academic press, 2nd edition, 1999. 4

[28] O. Matan, C. J. Burges, Y. LeCun, and J. S. Denker. Multidigit recognition using a space displacement neural network.In NIPS, pages 488–495. Citeseer, 1991. 2

[29] R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler,R. Urtasun, and A. Yuille. The role of context for object detection and semantic segmentation in the wild. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 891–898. IEEE, 2014. 9

[30] F. Ning, D. Delhomme, Y. LeCun, F. Piano, L. Bottou, and P. E. Barbano. Toward automatic phenotyping of developing embryos from videos. Image Processing, IEEE Transactions on, 14(9):1360–1371, 2005. 1, 2, 4, 7

[31] P. H. Pinheiro and R. Collobert. Recurrent convolutional neural networks for scene labeling. In ICML, 2014. 1, 2,4, 7, 8

[32] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.1, 2, 4

[33] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 8

[34] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR,abs/1409.1556, 2014. 1, 2, 3, 5

[35] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A.Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842,2014. 1, 2, 3, 5

[36] J. Tighe and S. Lazebnik. Superparsing: scalable nonparametric image parsing with superpixels. In ECCV, pages 352–365. Springer, 2010. 8

[37] J. Tighe and S. Lazebnik. Finding things: Image parsing with regions and per-exemplar detectors. In CVPR, 2013. 8

[38] J. Tompson, A. Jain, Y. LeCun, and C. Bregler. Joint training of a convolutional network and a graphical model for human pose estimation. CoRR, abs/1406.2984, 2014. 2

[39] L. Wan, M. Zeiler, S. Zhang, Y. L. Cun, and R. Fergus. Regularization of neural networks using dropconnect. In Proceedings of the 30th International Conference on Machine Learning (ICML-13), pages 1058–1066, 2013. 4

[40] R. Wolf and J. C. Platt. Postal address block location using a convolutional locator network. Advances in Neural Information Processing Systems, pages 745–745, 1994. 2

[41] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014,pages 818–833. Springer, 2014. 2

[42] N. Zhang, J. Donahue, R. Girshick, and T. Darrell. Partbased r-cnns for fine-grained category detection. In Computer Vision–ECCV 2014, pages 834–849. Springer, 2014.1



------

### 2.实例分割

主要有：FCIS、DeepMask、Mask R-CNN, Mask Scoring R-CNN 和 PANet,YOLACT 等网络，我们这里着重介绍**Mask R-CNN,Mask Scoring R-CNN和YOLACT**

#### Mask R-CNN

!> 请移驾到 [Mask R-CNN章节](/zh-cn/mask-r-cnn)

#### Mask Scoring R-CNN

!> 论文地址：https://arxiv.org/abs/1903.00241?context=cs

##### 0.摘要：

让深度网络意识到自己预测的质量是一个有趣但重要的问题。在大多数实例分割的任务中，实例分类的置信度被用作大多数实例分割框架中的掩模质量分数。然而，掩模质量（量化为实例的掩模与其GT之间的IoU）通常与分类得分没有很好的相关性。在本文中，我们研究了这个问题，并提出了Mask Scoring R-CNN，**它包含一个network block来学习预测实例掩码的质量。所提出的network block将实例特征和对应的预测掩模一起用于对掩模IoU进行回归**。掩模评分策略校准掩模质量和掩模评分之间的未对准，并通过在COCO AP评估,通过对COCO数据集的广泛评估，Mask Scoring R-CNN为不同的模型带来了一致且显着的增益，并且优于最先进的Mask R-CNN。我们希望我们简单有效的方法将为改进实例细分提供新的方向。方法源代码地址https://github.com/zjhuang22/maskscoring_rcnn。

##### 1.Introduction

深度网络正在极大地推动计算机视觉的发展，导致一系列最先进的任务，包括分类，目标检测，语义分割等。从计算机视觉深度学习的发展，我们可以观察到深度网络正逐渐从图像级预测到区域/盒级预测，像素级预测和实例/掩模级预测逐渐增长。进行细粒度预测的能力不仅需要更详细的标签，还需要更精细的网络设计。

在本文中，我们关注实例分割的问题，这是目标检测的更深下一步，从粗略的盒级实例识别转移到精确的像素级分类。具体来说，这项工作提出了一种新的方法来对实例分割假设进行评分，这对于例如分割评估非常重要。原因在于大多数评估指标是根据假设得分定义的，更精确的得分有助于更好地表征模型性能。例如，精确召回（P-R）曲线和平均精度（AP）通常用于具有挑战性的实例分割数据集COCO。如果一个实例分割假设未被正确评分，则可能被错误地视为假阳性或假阴性，导致AP减少。

但是，在大多数实例分割算法中，例如Mask R-CNN和MaskLab，实例掩码的分数与盒级分类置信度共享，这是由应用于提议特征的分类器预测的。 使用分类置信度来测量掩模质量是不合适的，因为它仅用于区分提议的语义类别，并且不知道实例掩码的实际质量和完整性。 分类置信度和掩模质量之间的不对准在图1中示出，其中实例分割假设获得准确的盒级定位结果和高分类分数，但是相应的掩模是不准确的。 显然，使用这种分类分数对掩模进行评分往往会降低评估结果。


<div align=center>
<img src="zh-cn/img/msrcnn/p1.png" />
</div>

*Fig1：实例分割的示例性案例，其中边界框与GT具有高重叠并且在掩模不够好的 情况下具有高分类分数。Mask R-CNN和我们提出的MS R-CNN预测的分数被附加在它们相应的边界框上方。左侧四幅图像显示出良好的检测结果，具有高分类分数但掩模质量低。我们的方法旨在解决这个问题。最右边的图像显示了具有高分类分数的良好掩模的情况。我们的方法将重新训练高分。可以看出，我们模型预测的分数可以更好地解释实际的mask质量。*

与之前旨在获得更准确的实例定位或分割掩膜的方法不同，我们的方法侧重于对mask进行评分。为了实现这一目标，我们的模型学习每个掩膜的分数而不是使用其分类分数。为清楚起见，我们称学习得分为掩码得分。

受实例分割的AP度量的启发，该实例分割使用预测掩模与其GT掩模之间的像素级IoU来描述实例分割质量，我们建议网络直接学习IoU。 在本文中，该IoU表示为MaskIoU。**一旦我们在测试阶段获得预测的MaskIoU，则通过将预测的MaskIoU和分类得分相乘来重新评估掩模得分**。 因此，掩码分数代表语义类别和实例掩码完整性。

学习MaskIoU与候选框分类或掩码预测完全不同，因为它需要“比较”预测的掩码和对象特征。**在Mask R-CNN框架内，我们实现了一个名为MaskIoU head的MaskIoU预测网络**。 它将mask head的输出和RoI特征作为输入，并使用简单的回归损失进行训练。 我们将所提出的模型，即具有MaskIoU头的Mask R-CNN命名为Mask Scoring R-CNN（MS R-CNN）。我们已经对我们的MS R-CNN进行了大量实验，结果表明我们的方法提供了一致且显着的性能改善，这归因于掩模质量和得分之间的一致性。

总之，这项工作的主要贡献如下：
1. 我们提出了Mask Scoring R-CNN，这是解决实例分割假设评分问题的第一个框架。 它探讨了提高实例分割模型性能的新方向。考虑到实例掩码的完整性，如果掩码不够好但是实例掩码的分数很高，则可能会受到惩罚。
2. 我们的MaskIoU head非常简单有效。 在具有挑战性的COCO基准测试中的实验结果表明，当使用来自我们的MS R-CNN的掩模得分而不仅仅是分类置信度时，AP在各种骨干网络上始终如一地提高约1.5％。

##### 2.Related Work
###### 2.1 Instance Segmentation

当前实例分割方法可大致分为两类。一种是基于检测的方法，另一种是基于分割的方法。基于检测的方法利用最先进的检测器，如Faster R-CNN， R-FCN [8]，得到每个实例的区域，然后预测每个区域的掩模。Pinheiro等人提出DeepMask以滑动窗口方式对中心对象进行分割和分类。Dai等人提出了实例敏感的FCNs来生成位置敏感的地图并将它们组合起来以获得最终的掩模。 FCIS采用具有内/外分数的位置敏感地图来生成实例分割结果。He等人通过添加实例级语义分段分支，提出了优于Faster R-CNN的Mask R-CNN。基于Mask R-CNN，Chen等人提出MaskLab使用位置敏感分数来获得更好的结果。然而，这些方法的一个潜在缺点是掩模质量只由分类分数来衡量，从而导致了上面讨论的问题。

基于分割的方法首先预测每个像素的类别标签，然后将它们组合在一起以形成实例分割结果。 梁等人。 [24]使用谱聚类来聚类像素。 其他工作，例如[20,21]，在聚类过程中添加边界检测信息。 Bai等人 [1]预测了像素级能量值和用于分组的分水岭算法。最近，有一些工作[30,11,14,10]使用度量学习来学习嵌入。具体而言，这些方法学习每个像素的嵌入，以确保来自相同实例的像素具有类似的嵌入。然后，对学习的嵌入执行聚类以获得最终实例标签。由于这些方法没有明确的分数来测量实例掩码质量，因此他们必须使用平均像素级分类分数作为替代。

上述两种方法都没有考虑掩模评分和掩模质量之间的对齐。由于掩模得分的不可靠性，IoU对ground truth越高的掩模假设，如果掩模得分越低，其优先级越低。在这种情况下，最终的AP因此降级。

###### 2.2 Detection Score Correction

目前有几种针对检测框分类分值进行校正的方法，其目的与我们的方法相似。Tychsen-Smith等人提出了Fitness NMS使用IoU之间检测到的边界框和他们的真实标签。 将box IoU预测作为分类任务。我们的方法与此方法的不同之处在于我们将掩模IoU估计表示为回归任务。Jiang等[19]提出IoU-Net直接对box IoU进行回归，预测的IoU用于NMS和边界框细化。 在[5]中，Cheng等人讨论了假阳性样本并使用分离的网络来校正这些样本的得分。SoftNMS [2]使用两个框之间的重叠来校正低分框。Neumann等[29]提出了Relaxed Softmax来预测标准softmax中的温度比例因子值，用于安全关键行人检测。

与专注于边界框级别检测的这些方法不同，我们的方法设计用于实例分割。在我们的Mask-IoU head中进一步处理实例掩码，以便网络可以知道实例掩码的完整性，并且最终掩码分数可以反映实例分段假设的实际质量。这是提高实例分割性能的一个新方向。

##### 3.Method
###### 3.1 Motivation

在当前的Mask R-CNN框架中，检测（即，实例分割）假设的分数由其分类分数中的最大元素确定。由于背景杂乱，遮挡等问题，分类得分可能高但掩模质量低，如图1所示的例子。为了定量分析这个问题，我们将Mask R-CNN的vanilla mask评分与预测的mask和它的真实标签掩码（MaskIoU）之间的实际IoU进行比较。具体来说，我们使用Mask R-CNN和ResNet-18 FPN在COCO 2017验证数据集上进行实验。然后我们在Soft-NMS之后选择MaskIoU和分类分数大于0.5的检测假设。MaskIoU在分类分数上的分布如图2（a）所示，每个MaskIoU区间的平均分类分数在图2（c）中以蓝色显示。这些图显示分类得分和MaskIoU在Mask R-CNN中没有很好地相关。

<div align=center>
<img src="zh-cn/img/msrcnn/p2.png" />
</div>

*Fig2。比较Mask R-CNN和我们提出的MS R-CNN。(a)显示Mask R-CNN的结果，Mask score与MaskIoU的关系较小。(b)显示MS R-CNN的检测结果，我们用高评分和低MaskIoU来惩罚检测，掩模评分可以更好地与MaskIoU相关。(c)给出量化结果，将每个MaskIoU区间的得分取平均值，可以看出我们的方法与MaskIoU之间有更好的对应关系.*

在大多数实例分割评估中，如COCO，低MaskIoU和高分数的检测假设是有害的。在许多实际应用中，确定检测结果何时可以信任，何时不能信任[29]非常重要。这促使我们根据MaskIoU的每一个检测假设学习一个校准的mask评分。在不失一般性，我们研究了Mask R-CNN框架，并提出了相应的解决方案Mask Scoring R-CNN (MS R-CNN)，加上MaskIoU head 模块Mask R-CNN可以学到对应mask分数更高。我们框架预测到的mask得分显示在图2（b）和图2（c）的橙色直方图中。

###### 3.2 Mask scoring in Mask R-CNN

MS R-CNN在概念上很简单：使用MaskIoU head的 Mask R-CNN，它将实例特征和预测的掩模一起作为输入，并预测输入掩模和真实标签掩模之间的IoU，如图3所示我们框架的详细信息。

<div align=center>
<img src="zh-cn/img/msrcnn/p3.png" />
</div>

*Fig 3. Mask Scoring R-CNN的网络架构。 输入图像feed到骨干网络(backbone)，以通过RPN生成RoIs并且通过RoIAlign生成RoI。 R-CNN head和mask head是Mask R-CNN的标准组件。 为了预测MaskIoU，我们使用预测的mask和RoI特征作为输入。 MaskIoU head有4个卷积层（所有kernal= 3，最后一个使用stride = 2进行下采样）和3个完全连接的层（最后一个输出C类MaskIoU。）*


**Mask R-CNN**：我们首先简要回顾一下Mask R-CNN [15]。Faster R-CNN [33]，Mask R-CNN都由两个阶段组成。第一阶段是候选区域网络（RPN）。无论对象类别如何，它都会提出候选对象边界框。第二阶段称为R-CNN阶段，它使用RoIAlign为每个候选提取特征，并执行候选分类，边界框回归和掩模预测。

**Mask scoring**：我们将$S_{mask}$定义为预测掩膜的得分。理想的$S_{mask}$等于预测掩模与其匹配的真实标签掩模之间的像素级IoU，之前称为MaskIoU。理想的$S_{mask}$也应该只对真实标签类别有正值，对于其他类别应该为零，因为掩模只属于一个类。这需要掩码得分在两个任务上表现良好：将掩码分类到正确的类别，并回归候选的MaskIoU用于前景对象类别。

仅使用单个目标函数很难训练这两个任务。为了简化，我们可以将掩码得分学习任务分解为掩码分类和IoU回归，表示为所有对象类别的$S_{mask} = S_{cls} \times S_{iou}$。$S_{cls}$侧重于对属于哪个类的预测进行分类，$S_{iou}$专注于回归MaskIoU。

至于$S_{cls}$的目标是对属于哪个类的预测进行分类，这已经在R-CNN阶段的分类任务中完成。所以我们可以直接拿相应的分类得分。回归$S_{iou}$是本文的目标，将在下一段中讨论。

**MaskIoU head**: MaskIoU head旨在回归预测掩模与其真实标签掩模之间的IoU。我们使用RoIAlign图层的特征串联和预测的掩模作为MaskIoU head的输入。在concat时，我们使用kernal大小为2且stride为2的最大池化层，以使预测的掩模具有与RoI特征相同的空间大小。我们只选择将MaskIoU回归到真实标签类（用于测试，我们选择预测的类）而不是所有类。我们的MaskIoU head由4个卷积层和3个完全连接的层组成。对于4个卷积层，我们遵循Mask head并将所有卷积层的内核大小和过滤器数分别设置为3和256。对于3个完全连接（FC）层，我们遵循R-CNN head并将前两个FC层的输出设置为1024，将最终FC的输出设置为类的数量。

**Training**: 为了训练MaskIoU head，我们使用RPN预测作为训练样本。 训练样本需要在预测框和匹配的真实标签框之间具有大于0.5的IoU，这与Mask R-CNN的Mask head的训练样本相同。为了生成每个训练样本的回归目标，我们首先获得目标类的预测掩码，并使用阈值0.5对预测掩码进行二值化。

然后我们在二元掩模和它匹配的真实标签之间使用MaskIoU作为MaskIoU目标。我们使用L2损失来回归MaskIoU，并且将损失权重设置为1。预测的MaskIoU head被集成到Mask R-CNN中，并且整个网络是端对端训练的。

**Inference**: 在推断过程中，我们只使用MaskIoU head来校准从R-CNN生成的分类分数。具体地，假设Mask R-CNN的R-CNN级输出N个边界框，并且其中在选择SoftNMS [2]之后的top-k（即k = 100）得分框。然后将前k个box送入Mask head以生成多类掩码。这是标准的Mask R-CNN推理程序。 我们也遵循这个程序，并输入top-k个目标掩码来预测MaskIoU。 将预测的MaskIoU与分类分数相乘，以获得新的校准掩模分数作为最终的掩模置信度。

##### 4.Experiments

所有实验都在COCO数据集[26]上进行，有80个类别。我们遵循COCO 2017设置，使用115k图像train split进行训练，5k validation split进行验证，20k test-dev split进行测试。我们使用COCO评估指标AP（平均超过IoU阈值）来报告结果，包括AP @ 0.5，AP @ 0.75，和$AP_S$，$AP_M$，$AP_L$（不同规模的AP）。AP@0.5（或AP@0.75）表示使用IoU阈值0.5（或0.75）来确定评估中预测的边界框或掩模是否为正。除非另有说明，否则使用掩模IoU评估AP。

###### 4.1 Implementation Details

我们使用我们的reproduced Mask R-CNN进行所有实验。我们使用基于ResNet-18的FPN网络进行消融研究，使用ResNet-18/50/101基于更快的R-CNN / FPN / DCN + FPN [9]，将我们的方法与其他基线结果进行比较。 对于ResNet-18 FPN，输入图像的大小调整为沿短轴为600px，沿长轴最大为1000px，用于训练和测试。与标准FPN [25]不同，我们在ResNet-18中仅使用C4，C5作为RPN预测和特征提取器。对于ResNet-50/101，输入图像的短轴调整为800 px，长轴调整为1333px 用于训练和测试。ResNet-50/101的其余配置遵循Detectron [13]。所有网络我们训练了18个epoch，在14个epoch和17个epoch之后将学习率降低了0.1倍。将具有动量0.9的同步SGD用作优化器。 为了进行测试，我们使用SoftNMS并保留每个图像的前100个分数检测。

###### 4.2 Quantitative Results

我们在不同的骨干网络上报告我们的结果，包括ResNet-18/50/101和不同的框架，包括Faster R-CNN / FPN / DCN + FPN [9]，以证明我们的方法的有效性。结果显示在表1和表2中。我们使用$AP_m$报告实例分割结果，使用$AP_b$报告检测结果。我们报告了我们重现的Mask R-CNN结果和我们的MS R-CNN结果。如表1所示，与Mask R-CNN相比，我们的MS R-CNN对骨干网络不敏感，可以在所有骨干网络上实现稳定的改善：我们的MS R-CNN可以获得显着的改善（约1.5 AP）。特别是对于AP@0.75，我们的方法可以将基线提高约2个点。表2表明我们的MS R-CNN对于不同的框架是稳健的，包括Faster R-CNN / FPN / DCN + FPN。此外，我们的MS R-CNN不会损害边界框检测性能；实际上，它略微提高了边界框检测性能。test-dev的结果在表3中显示，仅报告实例分割结果。

<div align=center>
<img src="zh-cn/img/msrcnn/p4.png" />
</div>

*表1. COCO 2017验证结果。显示了检测和实例分割结果。$AP_m$表示实例分割结果，$AP_b$表示检测结果。 没有✔的结果是Mask R-CNN，而有✔的则是我们的MS R-CNN。结果表明，我们的方法对不同的骨干网络不敏感。*

<div align=center>
<img src="zh-cn/img/msrcnn/p5.png" />
</div>

*表2.COCO 2017验证结果。显示了检测和实例分割结果。$AP_m$表示实例分割结果，$AP_b$表示检测结果。在结果区域中，第1行和第2行使用Faster R-CNN框架；第3行和第4行另外使用FPN框架；第5行和第6行另外使用DCN + FPN。结果表明，提出的MaskIoU head一致得到改进*

<div align=center>
<img src="zh-cn/img/msrcnn/p6.png" />
</div>

*表3.比较COCO 2017 test-dev上的不同实例分割方法。*


###### 4.3 Ablation Study

我们在COCO 2017验证集上全面评估我们的方法。 我们使用ResNet-18 FPN进行所有消融研究实验。

**MaskIoU head输入的设计选择**： 我们首先研究MaskIoU head输入的设计选择，它是掩模头部的预测掩模得分图（`28x28xC`）与RoI特征的融合。有一些设计选择如图4所示，并解释如下：

（a）目标掩码连接RoI特征：获取目标类的分数图，max-pooled并与RoI特征进行concat。

（b）目标掩模乘以RoI特征：获取目标类的分数图，max-p00led并乘以RoI特征。

（c）所有掩码连接RoI特征：所有C类掩模分数图max-pooled并与RoI特征concat。

（d）目标掩模连接高分辨率RoI特征：获取目标类的分数图并与`28 * 28`的RoI特征concat。

结果显示在表4中。我们可以看到MaskIoU head的性能对于融合掩模预测和RoI特征的不同方式是稳健的。在各种设计中都观察到性能提升。由于连接目标分数图和RoI特征可获得最佳结果，因此我们将其用作默认选项。

<div align=center>
<img src="zh-cn/img/msrcnn/p7.png" />
</div>

*Fig 4. 不同的设计选择的MaskIoU head输入。*

<div align=center>
<img src="zh-cn/img/msrcnn/p8.png" />
</div>

*表4. MaskIoU head输入的不同设计选择的结果。*


**训练目标的选择**： 如前所述，我们将掩码分数学习任务分解为掩码分类和MaskIoU回归。 是否可以直接学习掩膜评分？ 此外，RoI可以包含多个类别的对象。我们应该为所有类别学习MaskIoU吗？如何设置MaskIoU head的训练目标仍需要探索。训练目标有很多不同的选择：

1. 学习目标类别的MaskIoU，同时忽略proposal中的其他类别。这也是本文中的默认训练目标，以及本段中所有实验的对照组。
2. 学习所有类别的MaskIoU。如果类别未出现在RoI中，则其目标MaskIoU设置为0。此设置表示仅使用回归来预测MaskIoU，这要求回归量知道不存在不相关的类别。
3. 学习所有正类别的MaskIoU，其中正类别表示该类别出现在RoI区域。并且忽略了预测中的其余类别。此设置用于查看对RoI区域中更多类别执行回归是否更好。

表5显示了上述训练目标的结果。通过比较设置`＃1`和设置`＃2`，我们可以发现所有类别的训练MaskIoU（仅基于回归的Mask-IoU预测）将大大降低性能，这验证了我们对训练的看法 使用单一目标函数进行分类和回归是困难的。

设置`＃3`的性能不如设置`＃1`是合理的，因为对所有正类别回归MaskIoU会增加MaskIoU head的负担。因此，学习目标类别的MaskIoU将用作我们的默认选择。

<div align=center>
<img src="zh-cn/img/msrcnn/p9.png" />
</div>

*表5. 使用不同训练目标的结果。*


**如何选择训练样本**： 由于所提出的MaskIoU head建立在Mask R-CNN框架之上，因此MaskIoU头部的所有训练样本都具有大于0.5的box-level IoU，其GT边界框根据Mask R-CNN中的设置。 但是，他们的MaskIoU不得超过0.5。

给定阈值，我们使用Mask-IoU大于训练MaskIoU head的样本。表6显示了结果。结果表明，使用所有示例的训练获得最佳性能。

<div align=center>
<img src="zh-cn/img/msrcnn/p10.png" />
</div>


*表6. MaskIoU head选择不同训练样本的结果。*


###### 4.4 讨论

在本节中，我们将首先讨论预测的MaskIoU的质量，然后如果MaskIoU的预测是完美的，则研究Mask Scoring R-CNN的上界性能，并最后分析MaskIoU head的计算复杂度。在讨论中，使用弱骨干网络（即ResNet-18 FPN和强骨干网络，即ResNet-101 DCN + FPN）在COCO 2017验证集上获得所有结果。

**预测MaskIoU的质量**： 我们使用真实标签和预测的Mask-IoU之间的相关系数来测量我们的预测质量。回顾我们的测试程序，我们根据分类分数选择SoftNMS之后的前100个评分框，将检测到的框输入到Mask head并获得预测的掩模，然后使用预测的掩模和RoI特征作为MaskIoU head的输入。MaskIoU head的输出和分类得分进一步整合到最终的mask得分。

我们在COCO 2017验证数据集中为每个图像保留100个预测的MaskIoU，从所有5,000张图像中收集500,000个预测。 我们在图5中绘制每个预测及其相应的基本事实。我们可以看到MaskIoU预测与它们的基本事实具有良好的相关性，特别是对于那些具有MaskIoU的预测。对于ResNet-18 FPN和ResNet-101 DCN + FPN骨干网，预测与其基本事实之间的相关系数约为0.74。它表明预测的质量对骨干网的变化不敏感。这个结论也与表1一致。由于之前没有方法可以预测MaskIoU，我们参考之前关于预测边界框IoU的工作[19]。 [19]得到0.617相关系数，不如我们的。

<div align=center>
<img src="zh-cn/img/msrcnn/p11.png" />
</div>

*Fig 5. MaskIoU预测的可视化及其基本事实。（a）ResNet-18 FPN骨干的结果和（b）ResNet-101 DCN + FPN骨干的结果。x轴表示GT MaskIoU，y轴表示所提出的MaskIoU head的预测Mask-IoU。*

**MS R-CNN的上限性能**： 这里我们将讨论我们方法的上限性能。 对于每个预测的掩模，我们可以找到它匹配的真实标签掩模；然后，当真实标签MaskIoU大于0时，我们只使用GT Mask-IoU来替换预测的MaskIoU。结果如表7所示。结果表明，Mask Scoring R-CNN始终优于Mask R-CNN。与MS R-CNN的理想预测相比，仍然存在改进实际MS R-CNN的空间，其对于ResNet-18 FPN骨干为2.2％AP，对于ResNet-101 DCN + FPN骨干为2.6％AP。

<div align=center>
<img src="zh-cn/img/msrcnn/p12.png" />
</div>

*表7. Mask R-CNN，MS R-CNN和使用ResNet-18 FPN和ResNet-101 DCN + FPN作为COCO 2017验证集的主干的MS R-CNN的理想情况的结果。*


**模型大小和运行时间**： 我们的MaskIoU head有大约0.39G FLOP，而Mask head每个预测大约有0.53G LOP。 我们使用一个TITAN V GPU来测试速度（秒/图像）。至于ResNet-18 FPN，Mask R-CNN和MS R-CNN的速度约为0.132。 对于ResNet-101 DCN + FPN，Mask R-CNN和MS R-CNN的速度约为0.202。MS R-CNN中MaskIoU head的计算成本可以忽略不计。


##### 5. 结论

在本文中，我们研究了对实例分割掩码进行评分的问题，并提出了Mask scoring R-CNN。通过在Mask R-CNN中添加MaskIoU head，掩码的分数与MaskIoU对齐，这在大多数实例分割框架中通常被忽略。 所提出的MaskIoU head非常有效且易于实现。在COCO基准测试中，大量结果表明，Mask Scoring R-CNN始终明显优于Mask R-CNN。 它还可以应用于其他实例分割网络以获得更可靠的掩模分数。 我们希望我们简单有效的方法将作为基准，并有助于未来在实例分割任务中的研究。


------


#### YOLACT: Real-time Instance Segmentation

!> 论文地址：https://arxiv.org/abs/1904.02689

!> YOLACT: **Y**ou **O**nly **L**ook **A**t **C**oefficient**T**s

##### 0.Abstract

我们提出了一个简单的全卷积模型，用于实时实例分割，在单个Titan Xp上以33 FPS在MS COCO上实现29.8 mAP，这比以前的任何竞争方法都要快得多。此外，**我们只在一个GPU上训练后获得此结果**。我们通过将实例分割分成两个并行子任务来实现这一点：（1）生成一组原型掩码(prototype mask)和（2）预测每个实例的掩码系数(mask coefficients)。然后我们通过将原型与掩模系数线性组合来生成实例掩模。我们发现，因为这个过程不依赖于repooling，这种方法可以产生非常高质量的掩模，并且可以免费提供时间稳定性。此外，我们分析了原型的最新表现，发现他们学会以他们自有的一个translation variant manner定位实例，尽管是全卷积的。最后，我们还提出了**Fast NMS**，即以12毫秒的速度替代标准NMS，只有轻微的性能损失。


##### 1.Introduction

> “Boxes are stupid anyway though, I’m probably a true believer in masks except I can’t get YOLO to learn them.”

>“无论如何，盒子都是愚蠢的，我可能更相信masks，除非我不能让YOLO学习它们。”  From Joseph Redmon, YOLOv3[34]

创建实时实例分割算法需要什么？在过去的几年中，视觉领域在实例分割方面取得了巨大的进步，部分是借鉴了完善的目标检测领域的强大功能。最先进的实例分割方法，如MASK-RCNN和FCIS直接建立在先进的目标检测如Faster R-CNN和R-FCN。然而，这些方法主要关注性能而不是速度，让实例分割的场景缺乏类似于目标检检测器(如SSD,YOLO)实时性。在这项工作中，我们的目标填补快速，one-stage这一实例分割模型空白，就像SSD和YOLO填充目标检测的差距一样。

然而，实例分割比目标检测困难得多。像SSD和YOLO这样的One-stage目标检测器能够通过简单地移除the second stage 并以其他方式弥补丢失的性能来加速现有的 second stage detectors，如Faster R-CNN。然而，相同的方法不容易扩展到实例分割。最先进的second stage实例分割方法在很大程度上依赖于特征定位来生成掩模。也就是说，“repool”这些方法是以一些边界框区域为特征（例如，通过RoIpool / align），然后将这些现在已定位的特征提供给它们的掩模预测器。这种方法本质上是按顺序进行的，因此很难加速。确实存在像FCIS那样并行执行这些步骤的 One-stage方法，但它们在定位后需要大量的后处理，因此仍然远非实时。

为了解决这些问题，我们提出了YOLACT，这是一个舍弃明确定位步骤的实时实例分割框架。相反，YOLACT将实例分割分解为两个并行任务：（1）在整个图像上生成非局部原型掩模的字典，（2）预测每个实例的一组线性组合系数。然后从这两个组件生成全图像实例分割是很简单的：对于每个实例，使用相应的预测系数线性组合原型，然后使用预测的边界框进行裁剪。我们表明，通过以这种方式进行分割，网络学习如何自己定位实例掩码，其中在视觉上，空间上和语义上相似的实例在原型中看起来是不同的。

此外，由于原型掩模的数量与类别的数量无关（例如，可能存在比原型更多的类别），因此YOLACT学习分布式表示，其中每个实例用跨类别共享的原型的组合来分段。这种分布式表示导致原型空间中有趣的紧急行为：一些原型在空间上对图像进行分区，一些本地化实例，一些检测实例轮廓，一些编码位置敏感的方向图（类似于通过对FCIS中的位置敏感模块进行硬编码而获得的那些），大多数都是这些任务的组合.(见图5)

这种方法也有几个实际的优点。首先，它的速度很快：由于其并行结构和极轻的组装过程，YOLACT仅为one-stage backbone 探测器增加了少量的计算开销，即使使用ResNet-101也很容易达到30 FPS 。其次，masks是高质量的：由于masks使用图像空间的全部范围而不会因repooling而损失任何质量，因此我们用于大型物体的maska质量明显高于其他方法（见图7）。最后，不失一般性：生成原型和掩模系数的想法可以添加到几乎任何现代目标探测器。

作为奖励，以这种方式分解实例分割也与假设在人类视觉中起重要作用的腹侧（“什么”）和背侧（“哪里”）流松散相关。线性系数和相应的检测分支可以被认为是识别单个实例（“什么”），而原型掩模可以被视为在空间中定位实例（“在哪里”）。与two-stage“局部化 - 分割”类型方法相比，这更接近人类视觉，尽管仍然远离人类视觉。（这段说的有点牵强，硬向人类视觉去靠）

我们的主要贡献是第一个实时（> 30 FPS）实例分割算法，在具有挑战性的MS COCO数据集上具有竞争结果（见图1）。此外，我们分析了YOLACT原型的一些性质，并提供了实验来研究使用不同主干架构，原型数量和图像分辨率获得的速度与性能之间的权衡。我们还提供了一种新颖的 Fast NMS方法，比传统NMS快12ms，性能损失可以忽略不计。Pytorch源码已经公开，小编会在接下来的时间测试YOLACT在医疗领域实例分割的效果。

<div align=center>
<img src="zh-cn/img/yolact/p1.png" />
</div>

*图1：COCO上各种实例分割方法的速度 - 性能权衡。据我们所知，我们是第一个实时（超过30 FPS）方法在COCO test-dev上有大约30个 mask mAP。*

YOLACT 主要特点与贡献点：


[1] - 将实例分割划分为两个并行任务：

(1)生成整张图片的非局部原型mask的字典 (generating a dictionary of non-local prototype masks over the entire image.)

(2)预测每个实例的线性组合系数集合(predicting a set of linear combination mask coefficients per instance.)

[2] - 建立 prototype masks 和 mask coefficients 之间的线性组合，以得到实例 masks.

  - 由于不依赖于 repooling 操作，故可以得到高质量的 masks 以及较高的时序稳定性(temporal stability).
  - 根据 prototype mask 和 mask coefficients 得到整张图片的实例分割的处理：对于每个实例，线性组合 prototypes 及对应的预测 coefficients，然后再根据预测的边界框裁剪实例 mask.
  - 这种分割方式，网络自身会学习如何定位实例 masks(the network learns how to localize instance
masks on its own)，视觉上、空间上以及语义上相似的实例可能在 prototypes 是不同的.
  - prototype masks 的数量与物体类别无关(如，类别数量可能多于 prototypes 的数量)，YOLACT 实际上是学习了一种分布表示，每一个实例被分割为被多个类别共享的 prototypes 的组合(each instance is segmented with a combination of prototypes that are shared across categories.) 在 prototype 空间，某些 prototypes 对图片空间分块，某些 prototypes 定位实例，某些 prototypes 检测实例廓形，某些 prototypes 编码位置敏感的方向图(position-sensitive directional maps)，等等，这些 prototypes 的组合构成了最终的分割结果.

[3] - 提出 Fast NMS，在仅很少损失精度，对比标准 NMS 处理，有 12-ms 的速率提升.

YOLACT 的主要优势：

[1] - Fast: 并行化网络任务分支以及及其少量的线性组合处理. 相对于 one-stage backbone 检测器，仅新增了很少的计算量，甚至在采用 ResNet-101 时也取得了 30 FPS.

[2] - High-quality Masks: masks 是采用了全部的图片空间的信息，没有 repooling 的质量损失，对于大目标物体的 masks 明显高于其它方法.

[3] - General: 生成 prototypes 和 mask coefficients 的思想几乎可以用于任何深度学习目标检测器.


##### 2.Related Work

实例分割鉴于其重要性，已经进行了大量研究工作以推动实例分割的准确性。Mask-RCNN 是一种代表性的 two-stage实例分割方法，首先生成候选感兴趣区域（ROI），然后分类并在第二阶段对这些ROI进行实例分割。后续工作试图通过例如丰富FPN特征或解决掩模的置信分数与其定位精度之间的不兼容性来提高其准确性。这些 two-stage方法需要为每个ROI重新汇集功能并使用后续计算对其进行处理，这使得即使在降低图像大小时也无法获得实时速度（30 FPS）（参见表2c）

One-stage实例分割方法生成position sensitive maps，这些maps被组合成具有位置敏感池化的最终掩模，或者组合语义分段logits和方向预测logits 。虽然在概念上比Two-stage方法更快，但它们仍然需要重新计算或其他非平凡的计算（例如，掩模投票）。这严重限制了他们的速度，使他们远离实时。相比之下，我们的组装步骤更轻量级（只是线性组合），并且可以实现为一个GPU加速矩阵矩阵乘法，使我们的方法非常快。

最后，一些方法首先执行语义分割，然后进行边界检测，像素聚类，或者学习嵌入以形成实例掩码。同样，这些方法具有多个阶段和/或涉及成本过大的聚类过程，这限制了它们对于实时应用的可行性。

**实时实例分割**，虽然存在实时目标检测和语义分割方法，但很少有工作集中在实时实例分割上。Straight to Shapes[19] 可以用30 FPS的学习形状编码进行实例分割，但其准确性远远不如现代基线。Box2Pix[40] 依靠极轻量级骨干探测器（GoogLeNet v1 和SSD ）与手工设计算法相结合，在Cityscapes上获得10.9 FPS，在KITTI上获得35 FPS。然而，他们没有报告更具挑战性和语义丰富的COCO数据集的结果，该数据集与KITTI和Cityscapes中的8个相比有80个类别。此外，他们观察到从语义上简单的数据集（KITTI）到更复杂的数据集（城市景观）的相对性能大幅下降，因此更加困难的数据集（COCO）将构成挑战。事实上，Mask R-CNN [16]仍然是语义上具有挑战性的数据集的最快实例分割方法之一（5502像素图像上的13.5 FPS;参见表2c）。

**Prototypes**: 
Learning prototypes（又名词汇或码本）已经在计算机视觉中得到了广泛的探索。经典表征包括textons[21]和visual words[37]，通过稀疏性和位置先验进行。其他人设计了用于物体检测的原型。虽然相关，但这些工作使用Prototypes来表示特征，而我们使用它们来组装masks以进行实例分割。此外，我们学习了特定于每个图像的Prototypes，而不是整个数据集共享的全局Prototypes。

YOLACT Prototypes 与 BoFs(BoW):

在阅读 YOLACT 论文的时候，发现 prototypes 和 coefficients 与计算机视觉中的 BoF(Bag of Feature) 方法有很大的相关性.

+ Prototypes，也可以叫作词汇(vocabulary)、编码本(codebook).

+ BOF(BoW) 是一种图像特征提取方法，其实际上起源于文本领域的 BoW(Bag of Words) 模型. BoW 假定对于一个文本而言，忽略其词序、语法、句法等，仅看作是一个个词语的组合，每个词的出现都是相互独立的，不依赖于其它词的出现.

BOF 假设图像相当于一个文本，图像中的不同局部区域或特征可以看作是构成图像的词汇(codebook). 如图：

<div align=center>
<img src="zh-cn/img/yolact/p2.png" />
</div>

根据得到的图像的词汇，统计每个单词的频次，即可得到图片的特征向量，如图：

<div align=center>
<img src="zh-cn/img/yolact/p3.png" />
</div>

YOLACT 中的 Prototypes 与 BOFs 类图像特征提取方法的区别是：

[1] - BOFs 是特征提取方法，其采用 prototypes 来表示特征；而 YOLACT 将 prototypes 用于生成实例分割的 masks.

[2] - YOLACT 是针对每张图片学习 prototypes，而 BOFs 是对整个数据集所学习的全局共享的 prototypes.


##### 3.YOLACT

类比Mask R-CNN之于Faster R-CNN，YOLACT旨在现有的one-stage型检测器上添加一个mask分支来达到实例分割的目的，但这一过程中不希望引入特征定位步骤。对已有 one-stage 目标检测模型添加一个 mask 分支. 所采用的方式即为，将复杂的实例分割任务分解为两个更简单的、并行化的任务，并可以组合为最终的 masks：

[1] - 并行化分支一：采用 FCN 生成图像大小(image-sized)的 prototype masks 集合，其不依赖于任何一个实例.

[2] - 并行化分支二：在目标检测分支添加一个额外的 head，用于预测每个 anchor 的 mask coefficients 向量，其编码了在 prototype 空间每个实例的表示.

[3] - 最后，对于 NMS 处理后得到的每个实例，线性组合两个分支所输出的 prototypes masks 和 mask coefficients，即可生成实例的 mask 分割结果.

YOLACT 结构如图：

<div align=center>
<img src="zh-cn/img/yolact/p4.png" />
</div>

*图2：YOLACT体系结构蓝色/黄色表示原型中的低/高值，灰色节点表示未训练的功能，本例中为。我们使用ResNet-101 + FPN将此架构基于RetinaNet 。*

**可行性分析**: YOLACT 实例分割所基于的基本原理是，masks是空间语义连贯的(mask are spatially coherent)，如相互接近的像素更可能是相同实例的一部分.

卷积层利用了 masks 的空间语义连贯性，但是全连接层却是无法学习该信息的. 这就遇到了一个问题：由于 one-stage 目标检测器是以全连接层的输出作为每个 anchor 的类别和边界框系数的；two-stage 的方法，如 Mask R-CNN 是通过一个定位处理，如 RoIAlign 层，以保持特征的空间连贯性，并采用卷积层输出实例 masks. 不过，这种方法需要模型里实例分割要等着 first-stage RPN 来生成候选边界框，导致速度明显受影响.

YOLACT 将问题分解为两个并行化分支，使用全连接层来生成语义向量；使用卷积层来生成空间连贯的 masks，以分别得到 mask coefficients 和 prototype masks. 由于 prototypes 和 mask coefficients 可以分别独立进行计算，因此其新增的计算大部分来自于线性组合处理，其实现可以看作是矩阵相乘(matrix multiplication). 这样就实现了同时保持了在特征空间的空间连贯性和 one-stage 的快速.


###### 3.1 Prototype 生成

YOLACT 的 prototype 生成分支(protonet) 会输出整张图片的 k 个 prototype masks 集合.Protonet 是 `FCN` 网络，其输出层包含 `k 个通道(channels)`，每个通道分别表示一个 prototype. 如图3所示. Protonet 类似于语义分割任务，但其不同之处在于，其没有具体的关于 prototypes 的 loss；所有的监督都来自于线性组合后最终的 mask loss.

<div align=center>
<img src="zh-cn/img/yolact/p5.png" />
</div>

*图3：Protonet体系结构，标签表示图像大小为`550X550`的特征尺寸和通道。箭头表示`3X3`的卷积层，最后一层采用了`1X1`的卷积。尺寸的增加是一个上采样(upsample)，然后是卷积。灵感来自[16]中的掩模分支。*

这里包含两个重要的网络设计选择：

[1] - 采用 deeper backbone 特征的 Protonet 能够生成更鲁棒的 masks；

[2] - 更高分辨率的 prototypes 能够得到更高质量的 masks 以及对于小目标物体具有更好的性能.

因此，YOLACT 采用 FPN 网络，因为 FPN 的最大特征层(如图2中的$P_3$) 是最深的网络层. 因此，将$P_3$上采样到输入图片尺寸的四分之一大小，以增强对于小目标物体的效果.

最后，Protonet 网络输出的无界性(unbounded) 也是非常重要的，因为其使得网络能够对于 prototypes 非常置信的地方，生成更大的、更强的激活值(如，明显是背景的区域). 因此，YOLACT 采用 `ReLU`激活函数以得到解释性更好的 prototypes.


*解释*:

橙色的 P3-P7 是 FPN 网络。这个网络的生成是由 C5 经过一个卷积层得到 P5 开始的。接下来，对 P5 进行一次双线性插值将其放大，与经过卷积的 C4 相加得到 P4，同样的方法得到 P3。此外，还对 P5 进行了卷积得到 P6，对 P6 进行卷积得到 P7。

论文中解释了使用 FPN 的原因：作者注意到更深层的特征图能生成更鲁棒的 mask，而更大的 prototype mask 能确保最终的 mask 质量更高且更好地检测到小物体。又想特征深，又想特征图大，那就得使用 FPN 了。

接下来是并行的操作。P3 被送入 Protonet，P3-P7 也被同时送到 Prediction Head 中。 Protonet 的设计是受到了 Mask R-CNN 的启发，它由若干卷积层组成。其输入是 P3，其输出的 mask 维度是 `138*138*32`，即 32 个 prototype mask，每个大小是 `138*138`。

###### 3.2 Mask Coefficients

一般来说，anchor-based 目标检测器包含两个预测输出分支：(1)分支一预测 `c` 个类别置信度；(2)分支二预测 `4`个边界框回归值. 对于 mask coefficients 预测，YOLACT 只需添加一个并行化分支，用于预测 `k` 个 mask coefficients，每个 coefficient 分别对应一个 prototype. 因此,对于每个 anchor 而言，其不是生成`c+4` 个 coefficients, 而是`c+4+k`个，额外k个值即为mask coefficients.

<div align=center>
<img src="zh-cn/img/yolact/p6.png" />
</div>

另外作者认为，为了能够通过线性组合来得到最终想要的mask，能够从最终的mask中减去原型mask是很重要的。换言之就是，mask系数必须有正有负。所以，在mask coefficients预测时使用了`tanh`函数进行非线性激活，因为`tanh`函数的值域是`(-1,1)`.

*解释*：

这个分支的输入是 `P3-P7` 共五个特征图，Prediction Head 也有五个共享参数的预测层与之一一对应。

输入的特征图先生成 anchor。每个像素点生成 3 个 anchor，比例是` 1:1、1:2 和 2:1`。五个特征图的 anchor 基本边长分别是 `24、48、96、192 和 384`。基本边长根据不同比例进行调整，确保 anchor 的面积相等。

为了便于理解，接下来以 `P3` 为例，标记它的维度为 `W3*H3*256`，那么它的 anchor 数就是 `a3 = W3*H3*3`。接下来 Prediction Head 为其生成三类输出：

+ 类别置信。因为 COCO 中共有 81 类（包括背景），所以其维度为 `a3*81`；
+ 位置偏移，维度为 `a3*4`；
+ mask coefficient，维度为 `a3*32`。

对 `P4-P7` 进行的操作是相同的，最后将这些结果拼接起来，标记 `a = a3 + a4 + a5 + a6 + a7`，得到：

+ 全部类别置信。因为 COCO 中共有 81 类（包括背景），所以其维度为 `a*81`；
+ 全部位置偏移，维度为 `a*4`；
+ 全部 mask coefficient，维度为 `a*32`。

###### 3.3 Mask组合(合成)

在得到 mask prototypes 和 mask coefficients 后，采用线性组合处理，并再接 sigmoid 非线性函数，以生成最终的 masks. 该过程可以记为矩阵相乘与 sigmoid 的实现如下：

$$M=\sigma(PC^{T})$$

其中，`P`为`hxwxk`的 prototype masks 矩阵；`C`为`nxk`的 mask coefficients 矩阵；`n`为 NMS 和阈值处理后得到的`n`个实例数.

**Losses**: YOLACT模型训练时，采用了三个losess函数：

[1] - 分类loss$L_{cls}$, 使用smooth L1

[2] - 边界框回归loss$L_{box}$, 使用 smooth L1

[3] - mask loss$L_{mask}$,计算的是组合的mask M与GT mask $M_{gt}$之间的逐像素的二值交叉熵：$L_{mask}=BCE(M,M_{gt})$.


其中 mask loss 在计算时，因为 mask 的大小是 `138*138`，需要先将原图的 mask 数据通过双线性插值缩小到这一尺寸。

除了上面的三个 loss，源代码中还多了 `prototype loss`、`coefficient diversity loss`、`class existence loss` 和 `semantic segmentation loss`。


**Masks裁剪**：

为了改善小目标的分割效果，在inference时会首先根据检测框进行裁剪，再阈值化。而在训练时，会使用GT框来进行裁剪，并通过除以对应GT框面积来平衡loss尺度。

*解释*: Crop 和 Threshold

将 mask coefficient 和 prototype mask 做矩阵乘法，就得到了图像中的一个个 mask。Crop 指的是将边界外的 mask 清零，训练阶段的边界是 ground truth bounding box，评估阶段的边界是预测的 bounding box。

Threshold 指的是在评估阶段，只将置信度大于一个阈值的结果输出，源代码中用的是 0.3


###### 3.4 Emergent Behavior

YOLACT取得的效果可能有点出人意料，因为围绕实例分割任务的一个共识是：因为FCNs是平移不变的，所以需要在模型中添加转移方差。因此，在Mask R-CNN和FCIS中，通过显式方法添加了转移方差：方向图、位置存档，或是把mask预测分支放在第二个stage，都使得它们不需要再处理定位问题。

在YOLACT，唯一算是添加转移方差的地方是使用预测框裁剪feature map时。但其实这只是为了改善对小目标的分割效果，作者发现对大中型目标，不裁剪效果就很好了。所以，YOLACT似乎通过其原型的不同激活学习到了如何定位目标。

怎么理解YOLACT隐式学习到了转移方差？

<div align=center>
<img src="zh-cn/img/yolact/p7.png" />
</div>

*图5. 不同图片的相同的六种 prototypes 的激活情况例示. Prototype 1,4和5 是在图片 a 中具有清晰便捷定义的分割图；prototype 2 是 bottom-left 方向图；prototype 3 分割出背景，并给出了实例的轮廓；prototype 6 分割出了地面 (ground).*

在上图5中，红色方块图像(图像 a)的 prototype 激活值实际上是不能以 FCN without padding 操作来得到. 因为卷积输出单个像素，如果卷积的输入中图像中的各区域都是相同的，则，卷积的输出结果也应该是相同的. 另一方面，FCNs 网络中，如 ResNet，padding 操作使得网络能够知道某个像素距离图像边缘的距离. 从概念上来讲，其可以通过序列化的多个网络层，从边缘向中心逐步 padding 0 操作(如 kernel 为[1, 0]). 以ResNet 为例，其具有平移变化(inherently translation variant)，YOLACT 充分利用了该特点(如Fig 5 的图像 b 和图像 c 表现处理很明显的平移变化.)

许多原型mask只在图像的某些部分上激活，即它们只激活位于隐式学习边界一侧的对象。例如，上图中原型6学习的是背景信息。通过对这些原型进行组合，网络可以区分同一语义的不同（甚至重叠）的实例，比如在图d中，原型4减去原型5，可以区分开红色伞和绿色伞。


此外，原型学习是可压缩的。也就是说，如果protonet将多个原型的功能合并成一个，那么mask系数分支就会去对应学习相应的组合方法。例如，上图中，原型4具有分割的能力，但同时它又对左下角部分图像具有较高响应，原型5也类似，但它就对右下角部分响应更大。这也就解释了为什么可以根据实际情况调整原型数量（即protonet的输出通道数，默认为32），而又不会带来模型性能的下降。

##### 4.Backbone Detector

对于 backbone 检测器，YOLACT 同时关注速度与特征的信息丰富度，因为 prototypes 和 coefficients 的预测是比较困难的任务，需要利用较好特征. 因此，采用了类似于 RetinaNet 的网络.


**YOLACT 检测器**: 采用 ResNet-101 with FPN 作为默认的特征 backbone 网络，输入图像尺寸为 `550x550`. 但没有保持长宽比(aspect ratio).

类似于RetinaNet，YOLACT对FPN进行修改，不采用$P_2,P_6,P_7$的输入，从$P_5$开始采用连续的`3X3`，步长为2的卷积层，并分别放置长宽比为`[1,1/2,2]`的3个anchors, $P_3$的anchors的面积为24pixels方形，且其后的$P_4,P_5,P_6,P_7$的尺寸依次翻倍，即`[24,48,96,192,384]`.对于接每个$P_i$的预测head网络，采用被所有的3个分支共享的`3X3`卷积操作，然后每个分支并行的分别采用对应的`3X3`卷积操作。 对比于 RetinaNet，YOLACT 的 head 网络设计(图4所示) 更加轻量和快速。 YOLACT 采用 `smooth-L1 loss` 来训练 `box` 回归器，并采用与 SSD 相同的方式编码 box 回归坐标值。 YOLACT 采用 `softmax 交叉熵 loss` 来对 `c` 个类别标签(positive labels)和 `1`个背景类别(background label)进行处理. 采用 OHEM 选择训练样本,`neg:pos=3:1`。 与 RetinaNet 不同的是，YOLACT 未采用 focal loss。


##### 5.其他改进：Fast NMS与语义分割loss

**Fast NMS**: 在得到每个 anchor 的边界框回归系数和类别置信度后，类似于目标检测，YOLACT 也采用 NMS 去除重复的检测结果.

通常情况下，NMS 是序列的处理的，即对于数据集中的`c`个类别中的每一个类别，首先根据置信度，对检测到的边界框降序排列；然后，对于每个检测结果，移除所有比其置信度小的以及与其IoU 的重叠程度大于设定阈值的检测结果, [传统NMS算法参考](https://www.aiuai.cn/aifarm189.html)。

YOLACT 中 Fast NMS 可以并行地对每个实例进行判读是保留还是丢弃，主要思想是将传统NMS计算方法转为矩阵运算，从而受益于一些快速矢量运算库。

算法流程：

1）对每一个类别ci，取top-n个候选目标，并按得分降序排列；

2）计算一个c×n×n的IOU矩阵，其中每个n×n矩阵表示对该类n个候选框，两两之间的IOU；

3）因为自己与自己的IOU=1，IOU(A,B)=IOU(B,A)，所以对上一步得到的IOU矩阵进行一次处理。具体做法是将每一个通道，的对角线元素和下三角部分置为0；

4）去除与得分高的候选框重叠比例较大的框，具体做法是对上一步得到的矩阵，按列取最大值，然后对取完最大值的矩阵按阈值划分，只留下小于指定阈值的部分，即为Fast NMS过滤的结果。

为了便于理解，下面依旧举例说明:

假设我们有 5 个 RoI，对于 person 这一类，按照置信度由高到低分别是 `b1、b2、b3、b4 和 b5`。接下来通过矩阵运算得出它们彼此之间的 IoU，假设结果如下：

<div align=center>
<img src="zh-cn/img/yolact/p8.png" />
</div>

接下来将这个矩阵的下三角和对角线元素删去，得到下面的结果：

<div align=center>
<img src="zh-cn/img/yolact/p9.png" />
</div>

这其中的每一个元素都满足行号小于列号。接下来对每一列取最大值，得到 `[-, 0.8, 0.6, 0.6, 0.4]`。假设阈值为 `0.5`，即 IoU 超过 `0.5` 的两个 RoI 需要舍弃掉置信度低的那一个。根据最大值，`b2、b3 和 b4` 对应的列都超出了阈值，所以这三个 RoI 会在这一步舍去。

这样做的原因是，由于每一个元素都是行号小于列号，而序号又是按照置信度从高到低降序排列的，因此任一元素大于阈值，代表着这一列对应的 RoI 与一个比它置信度高的 RoI 过于重叠了，需要将它舍去。

这里需要注意的是，b3 虽然和 b2 过于重叠（IoU 为 0.6），但 b3 与 b1 的 IoU 只有 0.1，而 b2 与 b1 的 IoU 为 0.8。按照传统 NMS 算法，b2 会在第一轮循环中被舍去，这样 b3 将会被保留。这也是 Fast NMS 与 NMS 不同的地方，即原文所述：

> ..., we simply allow already-removed detections to suppress other detections, which is not possible in traditional NMS.


**语义分割loss**: 在训练阶段添加辅助loss。具体做法是在最大的feature map后面接1×1卷积，输出一个通道数为`c`的feature map。它对应的ground truth从实例分割的标注中获得，因此没有强制要求每个像素只能属于单一类别。所以此处使用的loss计算方法是在`c`个通道上分别运行`sigmoid`，类似YOLO-v3多标签分类的loss设计。


##### 6.Results

###### 6.1实例分割

在 COCO test-dev 数据集，基于单张 Titan Xp 显卡，不同方法的对比结果如下：

<div align=center>
<img src="zh-cn/img/yolact/p10.png" />
</div>

除了采用的 `550x550` 分辨率的输入，即 YOLACT-550 模型，还采用了 `400x400` 和 `700x700` 的输入图片尺寸分别训练了 YOLACT-400 和 YOLACT-700 模型，其中，anchor scales 采用等比例的调整($s_x=s_{550}/500x$).

降低输入图片的尺寸会导致模型精度出现较大的衰退，说明实例分割需要较大尺寸的图片. 增大图片尺寸会导致速度明显降低，但可以增加模型精度.

除了 ResNet-101 backbone 网络，还测试 ResNet-50 和 DarkNet-53，可以得到更快的速率. 如果需要更快的检测速度，建议采用 ResNet-50 或 DarkNet-53，而不是降低输入图片尺寸.

<div align=center>
<img src="zh-cn/img/yolact/p11.png" />
</div>

<div align=center>
<img src="zh-cn/img/yolact/p12.png" />
</div>

###### 6.2Mask Quality

由于 YOLACT 最终输出的 mask 尺寸为`138*138` ，且直接根据原始特征来生成的 masks(无需 repooling 变换以及没有特征 mis-align 问题)，因此，YOLACT 对于大目标的检测结果的 mask 质量是高于 Mask R-NN 和 FCIS 的(虽然mAP 略低). 如图7.

<div align=center>
<img src="zh-cn/img/yolact/p13.png" />
</div>


###### 6.3时间稳定性

作者实验发现，YOLACT在连续帧（视频）上的分割效果更加稳定，一方面因为mask质量本身就比较高，另一方面，作者认为Two-stage方法过多地依赖于第一阶段的区域提议，对YOLACT来说，虽然检测框可能会有波动，但因为原型mask的预测比较稳定，所以产生的分割效果也会更稳定。


##### 7.讨论

通过实验分析，作者发现YOLACT的错误大多是由检测器引起的，比如错误分类或bounding box定位不准。

**定位失败**
当在画面某个位置存在多个重叠的实例时，则网络可能无法通过自身学习到的原型mask对其进行定位。在这种情况下，会输出更接近前景mask的内容，而不是某些实例的分割。如下图所示，红色飞机下面的两辆卡车没有被正确分开。

<div align=center>
<img src="zh-cn/img/yolact/p14.png" />
</div>

**泄露**
因为YOLACT默认mask是在合成以后裁剪得到的，所以没有抑制框外部噪声的功能。如果检测框定位不准，那么就会导致mask泄露现象。另外，当多个同类实例相隔较远但大小又很大的时候，也可能发生这种现象。因为网络可能认为这几个实例已经离得很远了，自身不需要去分开定位它们，裁剪分支会负责处理这种情况。如下图所示，就属于这种情况。

<div align=center>
<img src="zh-cn/img/yolact/p15.png" />
</div>


------

### 3.标注工具

常用的图像分割标注工具有：

- VGG Image Annotator（VIA）：http://www.robots.ox.ac.uk/~vgg/software/via/

- labelme：https://github.com/wkentaro/labelme

- RectLabel：https://rectlabel.com/（但是仅限Mac）

- LabelBox：https://www.labelbox.io/

- COCO UI：https://github.com/tylin/coco-ui

这些标注工具使用简单，这里仅简单介绍labelme的使用。

### labelme

官方教程：https://github.com/wkentaro/labelme#anaconda

**1.安装**

```python
# 需要python2.7的环境或虚拟环境
conda create --name=labelme python=2.7（这一步python=*选择自己的Python版本）
activate labelme
conda install pyqt
pip install labelme

# 启动
activate labelme
labelme

```

<div align=center>
<img src="zh-cn/img/chapter10/labels/p1.png" />
</div>

**2.标注**

点击open dir，选择标注文件所在的文件夹，然后开始标注。注意标注的时候，假如你要标注的对象为人和狗，在画掩码过程中，一幅图像中如果有多个person、dog，命名规则为person1、person2…… dog1、dog2……。因为labelme生成的标签为一个label.png文件，这个文件只有一通道，在你标注时同一标签mask会被给予一个标签位，而mask要求不同的实例要放在不同的层中。最终训练所要得到的输入为一个`w*h*n`的ndarray，其中n为该图片中实例的个数。（如果是做语义分割，则没必要如此区分）,标注完成后，会生成一个json文件

<div align=center>
<img src="zh-cn/img/chapter10/labels/p2.png" />
</div>

**3.文件转换**

进入json文件所在的目录，在终端执行以下命令
```python
labelme_json_to_dataset <文件名>.json
```
可得到一个文件夹，里边有五个文件，分别是
```shell
*.png 
info.yaml 
label.png 
label_names.txt 
label_viz.png
```

其中 label.png 和 info.yaml 是我们需要用到的！ 标注已经完成！


