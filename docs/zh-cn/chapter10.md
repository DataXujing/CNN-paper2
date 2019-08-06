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

主要有：FCIS、DeepMask、Mask R-CNNC, Mask Scoring R-CNN 和 PANet,YOLACT 等网络，我们这里着重介绍**Mask R-CNN,Mask Scoring R-CNN和YOLACT**

#### Mask R-CNN

!> 请移驾到 [Mask R-CNN章节](/zh-cn/mask-r-cnn)


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


