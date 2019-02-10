## Fast R-CNN

------

### 0.摘要
<div align=center>
<img src="zh-cn/img/fast-R-CNN/p1.png" />
</div>

### 1.简介

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p2.png" />
</div>

#### 1.1 R-CNN与SPPnet

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p3.png" />
</div>

#### 1.2 贡献

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p4.png" />
</div>


### 2.Fast R-CNN架构与训练

Fast R-CNN的架构如下图（图1）所示：
<div align=center>
<img src="zh-cn/img/fast-R-CNN/p5.png" />
</div>

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p6.png" />
</div>

#### 2.1 RoI池化层

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p7.png" />
</div>

#### 2.2 从预训练网络初始化

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p8.png" />
</div>

#### 2.3 微调

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p9.png" />
</div>

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p10.png" />
</div>

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p11.png" />
</div>

#### 2.4 尺度不变性

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p12.png" />
</div>

### 3. Fast R-CNN检测

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p13.png" />
</div>


#### 3.1 使用截断的SVD来进行更快的检测

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p14.png" />
</div>

① 物体分类和窗口回归都是通过全连接层实现的，假设全连接层输入数据为x，输出数据为y，全连接层参数为W，尺寸为u×v，那么该层全连接计算为: y=Wx(计算复杂度为u×v)

② 若将W进行SVD分解，并用前t个特征值近似代替，即:W=U∑VT≈U(u,1:t)⋅∑(1:t,1:t)⋅V(v,1:t)T

那么原来的前向传播分解成两步: y=Wx=U⋅(∑⋅VT)⋅x=U⋅z 计算复杂度为u×t+v×t，若t<min(u,v)，则这种分解会大大减少计算量；

在实现时，相当于把一个全连接层拆分为两个全连接层，第一个全连接层不含偏置，第二个全连接层含偏置；实验表明，SVD分解全连接层能使mAP只下降0.3%的情况下提升30%的速度，同时该方法也不必再执行额外的微调操作。

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p26.png" />
</div>

### 4.主要结果

三个主要结果支持本文的贡献：

+ VOC07，2010和2012的最高的mAP。
+ 相比R-CNN，SPPnet，快速训练和测试。
+ 在VGG16中微调卷积层改善了mAP。

#### 4.1 实验配置

我们的实验使用了三个经过预训练的ImageNet网络模型，这些模型可以在线获得(https://github.com/BVLC/caffe/wiki/Model-Zoo)。第一个是来自R-CNN3的CaffeNet（实质上是AlexNet1）。 我们将这个CaffeNet称为模型S，即小模型。第二网络是来自14的VGG_CNN_M_1024，其具有与S相同的深度，但是更宽。 我们把这个网络模型称为M，即中等模型。最后一个网络是来自15的非常深的VGG16模型。由于这个模型是最大的，我们称之为L。在本节中，所有实验都使用单尺度训练和测试（s=600，详见尺度不变性：暴力或精细？）。

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p15.png" />
</div>

#### 4.2 多任务训练有用吗？

论文中的实验表明：多任务训练是方便的，因为它避免管理顺序训练任务的流水线，同时 多任务训练改进了分段训练的mAP。

#### 4.3 尺度不变性：暴力或精细？

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p16.png" />
</div>

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p17.png" />
</div>

#### 4.4 我们需要更过训练数据吗？

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p18.png" />
</div>

#### 4.5 SVM分类是否优于Softmax？

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p19.png" />
</div>

#### 4.6 更多的候选区域更好吗？

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p20.png" />
</div>

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p21.png" />
</div>

### 5.结论


<div align=center>
<img src="zh-cn/img/fast-R-CNN/p22.png" />
</div>

### 6.其他说明

重述一下Fast R-CNN的过程:

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p23.png" />
</div>

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p24.png" />
</div>

首先是读入一张图像，这里有两个分支，一路送入FCN（全卷机网络），输出 feature maps，另一路通过selective search提取region proposals（注意，Fast R-CNN论文中并没有明确说明使用selective search提取region proposals，但是Fast R-CNN是基于R-CNN的，姑且默认采用selective search提取region proposals吧。）提取的每个region proposal 都有一个对应的Ground-truth Bounding Box和Ground-truth class label。其中每个region proposals用四元数组进行定义，即(r, c, h, w)，即窗口的左上行列坐标与高和宽。值得注意的是，这里的坐标均是对应原图像的，而不是输出的feature maps。因此，还需要把原图像的坐标系映射到feature maps上。这一点也很简单，比如采用的是pre-trained 网络模型为VGG16的话，RoIPooling替换掉最后一个max pooling层的话，则原图像要经过4个max pooling层，输出的feature maps是原图像的1/16，因此，将原图像对应的四元数组转换到feature maps上就是每个值都除以16，并量化到最接近的整数。那么将region proposal的四元组坐标映射到feature maps上之后接下干什么呢？接下来就是把region proposal窗口框起来的那部分feature maps输入到RoIPooling（R-CNN是将其缩放到224x224，然后送入经过Fine-tuning的网络模型），得到固定大小的输出maps。

那么现在就谈一下RoIPooling层是怎样得到输出的，如下图所示：

<div align=center>
<img src="zh-cn/img/fast-R-CNN/p25.png" />
</div>




### Reference 

A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012. ↩ ↩2 ↩3 ↩4

Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard, W. Hubbard, and L. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Comp., 1989. ↩

R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014. ↩ ↩2 ↩3 ↩4 ↩5 ↩6 ↩7 ↩8 ↩9

P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. In ICLR, 2014. ↩ ↩2 ↩3

K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV, 2014. ↩ ↩2 ↩3 ↩4 ↩5 ↩6 ↩7 ↩8 ↩9 ↩10 ↩11 ↩12 ↩13 ↩14 ↩15 ↩16 ↩17 ↩18 ↩19 ↩20 ↩21

Y. Zhu, R. Urtasun, R. Salakhutdinov, and S. Fidler. segDeepM: Exploiting segmentation and context in deep neural networks for object detection. In CVPR, 2015. ↩ ↩2

S. Lazebnik, C. Schmid, and J. Ponce. Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories. In CVPR, 2006. ↩

Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. In Proc. of the ACM International Conf. on Multimedia, 2014. ↩

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009. ↩

D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable object detection using deep neural networks. In CVPR, 2014. ↩ ↩2

P. Felzenszwalb, R. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained part based models. TPAMI, 2010. ↩ ↩2 ↩3

E. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. In NIPS, 2014. ↩

J. Xue, J. Li, and Y. Gong. Restructuring of deep neural network acoustic models with singular value decomposition. In Interspeech, 2013. ↩

K. Chatfield, K. Simonyan, A. Vedaldi, and A. Zisserman. Return of the devil in the details: Delving deep into convolutional nets. In BMVC, 2014. ↩

K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. ↩

M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR, 2014. ↩ ↩2 ↩3

J. Carreira, R. Caseiro, J. Batista, and C. Sminchisescu. Semantic segmentation with second-order pooling. In ECCV, 2012. ↩

R. Caruana. Multitask learning. Machine learning, 28(1), 1997. ↩

R. Girshick, J. Donahue, T. Darrell, and J. Malik. Region-based convolutional networks for accurate object detection and segmentation. TPAMI, 2015. ↩

X. Zhu, C. Vondrick, D. Ramanan, and C. Fowlkes. Do we need more training data or better models for object detection? In BMVC, 2012. ↩

J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. IJCV, 2013. ↩ ↩2

P. Viola and M. Jones. Rapid object detection using a boosted cascade of simple features. In CVPR, 2001. ↩

J. H. Hosang, R. Benenson, P. Dollár, and B. Schiele. What makes for effective detection proposals? arXiv preprint arXiv:1502.05082, 2015. ↩

T. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft COCO: common objects in context. arXiv e-prints, arXiv:1405.0312 [cs.CV], 2014. ↩

https://alvinzhu.xyz/2017/10/10/fast-r-cnn/#fn:9
