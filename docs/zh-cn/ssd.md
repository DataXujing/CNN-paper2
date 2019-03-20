## SSD

------

### 1.概述

SSD算法，其英文全名是Single Shot MultiBox Detector，名字取得不错，Single shot指明了SSD算法属于one-stage方法，MultiBox指明了SSD是多框预测。之前讲了Yolo算法，从下图也可以看到，SSD算法在准确度和速度（除了SSD512）上都比Yolo要好很多。下图同时给出了不同算法的基本框架图，对于Faster R-CNN，其先通过CNN得到候选框，然后再进行分类与回归，而Yolo与SSD可以一步到位完成检测。相比Yolo，SSD采用CNN来直接进行检测，而不是像Yolo那样在全连接层之后做检测。其实采用卷积直接做检测只是SSD相比Yolo的其中一个不同点，另外还有两个重要的改变，一是SSD提取了不同尺度的特征图来做检测，大尺度特征图（较靠前的特征图）可以用来检测小物体，而小尺度特征图（较靠后的特征图）用来检测大物体；二是SSD采用了不同尺度和长宽比的先验框（Prior boxes, Default boxes，在Faster R-CNN中叫做锚，Anchors）。Yolo算法缺点是难以检测小目标，而且定位不准，但是这几点重要改进使得SSD在一定程度上克服这些缺点。下面我们详细讲解SDD算法的原理。


<div align=center>
<img src="zh-cn/img/ssd/ssd/p1.png" />
</div>


<div align=center>
<img src="zh-cn/img/ssd/ssd/p2.png" />
</div>

+ SSD，这是一种针对多个类别的单次检测器，比先前的先进的单次检测器（YOLO）更快，并且准确得多，事实上，与执行显式区域提出和池化的更慢的技术具有相同的精度（包括Faster R-CNN）。

+ SSD的核心是预测固定的一系列默认边界框的类别分数和边界框偏移，使用更小的卷积滤波器应用到特征映射上。

+ 为了实现高检测精度，我们根据不同尺度的特征映射生成不同尺度的预测，并通过纵横比明确分开预测。

+ 这些设计功能使得即使在低分辨率输入图像上也能实现简单的端到端训练和高精度，从而进一步提高速度与精度之间的权衡。

+ 实验包括在PASCAL VOC，COCO和ILSVRC上评估具有不同输入大小的模型的时间和精度分析，并与最近的一系列最新方法进行比较。


### 2.SSD模型

**算法概述**：SSD算法是一种直接预测目标类别和bounding box的多目标检测算法。与faster rcnn相比，该算法没有生成 proposal 的过程，这就极大提高了检测速度。针对不同大小的目标检测，传统的做法是先将图像转换成不同大小（图像金字塔），然后分别检测，最后将结果综合起来（NMS）。而SSD算法则利用不同卷积层的 feature map 进行综合也能达到同样的效果。算法的主网络结构是VGG16，将最后两个全连接层改成卷积层，并随后增加了4个卷积层来构造网络结构。对其中5种不同的卷积层的输出（feature map）分别用两个不同的 3×3 的卷积核进行卷积，一个输出分类用的confidence，每个default box 生成21个类别confidence；一个输出回归用的 localization，每个 default box 生成4个坐标值（x, y, w, h）。此外，这5个feature map还经过 PriorBox 层生成 prior box（生成的是坐标）。上述5个feature map中每一层的default box的数量是给定的(8732个)。最后将前面三个计算结果分别合并然后传给loss层。

**Default box**：文章的核心之一是作者同时采用lower和upper的feature map做检测。如下图所示，这里假定有8×8和4×4两种不同的feature map。第一个概念是feature map cell，feature map cell 是指feature map中每一个小格子，如图中分别有64和16个cell。另外有一个概念：default box，是指在feature map的每个小格(cell)上都有一系列固定大小的box，如下图有4个（下图中的虚线框，仔细看格子的中间有比格子还小的一个box）。假设每个feature map cell有k个default box，那么对于每个default box都需要预测c个类别score和4个offset，那么如果一个feature map的大小是m×n，也就是有mxn个feature map cell，那么这个feature map就一共有（c+4）xk x mxn 个输出。这些输出个数的含义是：采用3×3的卷积核对该层的feature map卷积时卷积核的个数，包含两部分（实际code是分别用不同数量的3*3卷积核对该层feature map进行卷积）：数量cxkxmxn是confidence输出，表示每个default box的confidence，也就是类别的概率；数量4xkxmxn是localization输出，表示每个default box回归后的坐标）。训练中还有一个东西：prior box，是指实际中选择的default box（每一个feature map cell 不是k个default box都取）。也就是说default box是一种概念，prior box则是实际的选取。训练中一张完整的图片送进网络获得各个feature map，对于正样本训练来说，需要先将prior box与ground truth box做匹配，匹配成功说明这个prior box所包含的是个目标，但离完整目标的ground truth box还有段距离，训练的目的是保证default box的分类confidence的同时将prior box尽可能回归到ground truth box。 举个列子：假设一个训练样本中有2个ground truth box，所有的feature map中获取的prior box一共有8732个。那个可能分别有10、20个prior box能分别与这2个ground truth box匹配上。训练的损失包含定位损失和回归损失两部分。

<div align=center>
<img src="zh-cn/img/ssd/ssd/p3.png" />
</div>

这里用到的 default box 和Faster RCNN中的 anchor 很像，在Faster RCNN中 anchor 只用在最后一个卷积层，但是在本文中，default box 是应用在多个不同层的feature map上。

<div align=center>
<img src="zh-cn/img/ssd/ssd/p4.png" />
</div>


**模型结构**

<div align=center>
<img src="zh-cn/img/ssd/ssd/p5.png" />
</div>

SSD的结构在VGG16网络的基础上进行修改，训练时同样为conv1_1，conv1_2，conv2_1，conv2_2，conv3_1，conv3_2，conv3_3，conv4_1，conv4_2，conv4_3，conv5_1，conv5_2，conv5_3（512），(conv6)fc6经过3x3x1024的卷积（原来VGG16中的fc6是全连接层，这里变成卷积层，下面的fc7层同理），(conv7)fc7经过1x1x1024的卷积，然后又增加了4个额外的卷积层：conv8_2,conv9_2,conv10_2,conv11_2，loss。然后一方面：针对conv4_3（4），fc7（6），conv8_2（6），conv9_2（6），conv10_2（4），conv11_2（4）（括号里数字是每一层选取的default box种类）中的每一个再分别采用两个3x3大小的卷积核进行卷积，这两个卷积核是并列的（括号里的数字代表prior box的数量，可以参考Caffe代码，所以上图中SSD结构的倒数第二列的数字8732表示的是所有prior box的数量，是这么来的38x38x4+19x19x6+10x10x6+5x5x6+3x3x4+1x1x4=8732），这两个3*3的卷积核一个是用来做localization的（回归用，如果prior box是6个，那么就有6x4=24个这样的卷积核，卷积后map的大小和卷积前一样，因为pad=1，下同），另一个是用来做confidence的（分类用，如果prior box是6个，VOC的object类别有20个，那么就有6x（20+1）=126个这样的卷积核）。如下图是conv6_2的localizaiton的3x3卷积核操作，卷积核个数是24（6x4=24，由于pad=1，所以卷积结果的map大小不变，下同）：这里的permute层就是交换的作用，比如你卷积后的维度是32×24×19×19，那么经过交换层后就变成32×19×19×24，顺序变了而已。而flatten层的作用就是将32×19×19×24变成32x8664，32是batchsize的大小。另一方面结合conv4_3（4），fc7（6），conv8_2（6），conv9_2（6），conv10_2（4），conv11_2（4）中的每一个和数据层（ground truth boxes）经过priorBox层生成prior box。

经过上述两个操作后，对每一层feature的处理就结束了。对前面所列的5个卷积层输出都执行上述的操作后，就将得到的结果合并：采用Concat，类似googleNet的Inception操作，是通道合并而不是数值相加。 


<div align=center>
<img src="zh-cn/img/ssd/ssd/p6.png" />
</div>

采用VGG16做基础模型，首先VGG16是在ILSVRC CLS-LOC数据集预训练。然后借鉴了DeepLab-LargeFOV，分别将VGG16的全连接层fc6和fc7转换成 3×3 卷积层 conv6和 1×1 卷积层conv7，同时将池化层pool5由原来的stride=2的 2×2 变成stride=1的 3×3（猜想是不想reduce特征图大小），为了配合这种变化，采用了一种Atrous Algorithm，其实就是conv6采用扩展卷积或带孔卷积（Dilation Conv），其在不增加参数与模型复杂度的条件下指数级扩大卷积的视野，其使用扩张率(dilation rate)参数，来表示扩张的大小，如下图6所示，(a)是普通的 3×3 3\times33×3 卷积，其视野就是 3×3 3\times33×3 ，(b)是扩张率为1，此时视野变成 7×7， （c） 扩张率为3时，视野扩大为 15×15，但是视野的特征更稀疏了。Conv6采用 3×3大小但dilation rate=6的扩展卷积。

<div align=center>
<img src="zh-cn/img/ssd/ssd/p7.png" />
</div>

然后移除dropout层和fc8层，并新增一系列卷积层，在检测数据集上做finetuing。

其中VGG16中的Conv4_3层将作为用于检测的第一个特征图。conv4_3层特征图大小是 38×38，但是该层比较靠前，其norm较大，所以在其后面增加了一个L2 Normalization层，以保证和后面的检测层差异不是很大，这个和Batch Normalization层不太一样，其仅仅是对每个像素点在channel维度做归一化，而Batch Normalization层是在[batch_size, width, height]三个维度上做归一化。归一化后一般设置一个可训练的放缩变量gamma，使用TF可以这样简单实现：


```python
# l2norm (not bacth norm, spatial normalization)
def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma
```

<div align=center>
<img src="zh-cn/img/ssd/ssd/p8.png" />
</div>


<div align=center>
<img src="zh-cn/img/ssd/ssd/p9.jpg" />
</div>


<div align=center>
<img src="zh-cn/img/ssd/ssd/p10.png" />
</div>


### 3.训练

**（1）先验框匹配**：

在训练过程中，首先要确定训练图片中的ground truth（真实目标）与哪个先验框来进行匹配，与之匹配的先验框所对应的边界框将负责预测它。在Yolo中，ground truth的中心落在哪个单元格，该单元格中与其IOU最大的边界框负责预测它。但是在SSD中却完全不一样，SSD的先验框与ground truth的匹配原则主要有两点。首先，对于图片中每个ground truth，找到与其IOU最大的先验框，该先验框与其匹配，这样，可以保证每个ground truth一定与某个先验框匹配。通常称与ground truth匹配的先验框为正样本（其实应该是先验框对应的预测box，不过由于是一一对应的就这样称呼了），反之，若一个先验框没有与任何ground truth进行匹配，那么该先验框只能与背景匹配，就是负样本。一个图片中ground truth是非常少的， 而先验框却很多，如果仅按第一个原则匹配，很多先验框会是负样本，正负样本极其不平衡，所以需要第二个原则。第二个原则是：对于剩余的未匹配先验框，若某个ground truth的 IOU 大于某个阈值（一般是0.5），那么该先验框也与这个ground truth进行匹配。这意味着某个ground truth可能与多个先验框匹配，这是可以的。但是反过来却不可以，因为一个先验框只能匹配一个ground truth，如果多个ground truth与某个先验框 IOU 大于阈值，那么先验框只与IOU最大的那个先验框进行匹配。第二个原则一定在第一个原则之后进行，仔细考虑一下这种情况，如果某个ground truth所对应最大IOU 小于阈值，并且所匹配的先验框却与另外一个ground truth的 IOU 大于阈值，那么该先验框应该匹配谁，答案应该是前者，首先要确保某个ground truth一定有一个先验框与之匹配。但是，这种情况我觉得基本上是不存在的。由于先验框很多，某个ground truth的最大 IOU 肯定大于阈值，所以可能只实施第二个原则既可以了，图8为一个匹配示意图，其中绿色的GT是ground truth，红色为先验框，FP表示负样本，TP表示正样本。

<div align=center>
<img src="zh-cn/img/ssd/ssd/p11.png" />
</div>


尽管一个ground truth可以与多个先验框匹配，但是ground truth相对先验框还是太少了，所以负样本相对正样本会很多。为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排列，选取误差的较大的top-k作为训练的负样本，以保证正负样本比例接近1:3。


**（2）损失函数**:

训练样本确定了，然后就是损失函数了。损失函数定义为位置误差（locatization loss， loc）与置信度误差（confidence loss, conf）的加权和：

<div align=center>
<img src="zh-cn/img/ssd/ssd/p12.png" />
</div>

<div align=center>
<img src="zh-cn/img/ssd/ssd/p13.png" />
</div>

<div align=center>
<img src="zh-cn/img/ssd/ssd/p14.png" />
</div>

<div align=center>
<img src="zh-cn/img/ssd/ssd/p16.png" />
</div>

**（3）难例挖掘**：

在匹配步骤之后，大多数默认边界框为负例，尤其是当可能的默认边界框数量较多时。这在正的训练实例和负的训练实例之间引入了显著的不平衡。我们不使用所有负例，而是使用每个默认边界框的最高置信度损失来排序它们，并挑选最高的置信度，以便负例和正例之间的比例至多为3:1。我们发现这会导致更快的优化和更稳定的训练。


**（4）数据增强**：

为了使模型对各种输入目标大小和形状更鲁棒，每张训练图像都是通过以下选项之一进行随机采样的：

+ 使用整个原始输入图像。
+ 采样一个图像块，使得与目标之间的最小Jaccard重叠为0.1，0.3，0.5，0.7或0.9。
+ 随机采样一个图像块。

每个采样图像块的大小是原始图像大小的[0.1，1]，长宽比在12和2之间。如果实际边界框的中心在采用的图像块中，我们保留实际边界框与采样图像块的重叠部分。在上述采样步骤之后，除了应用类似于文献[14]中描述的一些光度变形之外，将每个采样图像块调整到固定尺寸并以0.5的概率进行水平翻转

<div align=center>
<img src="zh-cn/img/ssd/ssd/p15.png" />
</div>


**预测过程**

预测过程比较简单，对于每个预测框，首先根据类别置信度确定其类别（置信度最大者）与置信度值，并过滤掉属于背景的预测框。然后根据置信度阈值（如0.5）过滤掉阈值较低的预测框。对于留下的预测框进行解码，根据先验框得到其真实的位置参数（解码后一般还需要做clip，防止预测框位置超出图片）。解码之后，一般需要根据置信度进行降序排列，然后仅保留top-k（如400）个预测框。最后就是进行NMS算法，过滤掉那些重叠度较大的预测框。最后剩余的预测框就是检测结果了。

### 4.实验

SSD paper中对各种公开数据集进行实验，在模型分析中强调：

+ 数据增强是至关重要的
+ 更多默认边界框形状会更好
+ 使用空洞卷积（Atrous）速度更快（VGG的空洞版本）
+ 多个不同分辨率的输出层会更好

### 5.相关工作

在图像中有两种建立的用于目标检测的方法，一种基于滑动窗口，另一种基于区域提出分类。在卷积神经网络出现之前，这两种方法的最新技术——可变形部件模型（DPM）[26]和选择性搜索[1]——具有相当的性能。然而，在R-CNN[22]结合选择性搜索区域提出和基于后分类的卷积网络带来的显著改进后，区域提出目标检测方法变得流行。

最初的R-CNN方法已经以各种方式进行了改进。第一套方法提高了后分类的质量和速度，因为它需要对成千上万的裁剪图像进行分类，这是昂贵和耗时的。SPPnet[9]显著加快了原有的R-CNN方法。它引入了一个空间金字塔池化层，该层对区域大小和尺度更鲁棒，并允许分类层重用多个图像分辨率下生成的特征映射上计算的特征。Fast R-CNN[6]扩展了SPPnet，使得它可以通过最小化置信度和边界框回归的损失来对所有层进行端到端的微调，最初在MultiBox[7]中引入用于学习目标。

第二套方法使用深度神经网络提高了提出生成的质量。在最近的工作MultiBox[7,8]中，基于低级图像特征的选择性搜索区域提出直接被单独的深度神经网络生成的提出所取代。这进一步提高了检测精度，但是导致了一些复杂的设置，需要训练两个具有依赖关系的神经网络。Faster R-CNN[2]将选择性搜索提出替换为区域提出网络（RPN）学习到的区域提出，并引入了一种方法，通过交替两个网络之间的微调共享卷积层和预测层将RPN和Fast R-CNN结合在一起。通过这种方式，使用区域提出池化中级特征，并且最后的分类步骤比较便宜。我们的SSD与Faster R-CNN中的区域提出网络（RPN）非常相似，因为我们也使用一组固定的（默认）边界框进行预测，类似于RPN中的锚边界框。但是，我们不是使用这些来池化特征并评估另一个分类器，而是为每个目标类别在每个边界框中同时生成一个分数。因此，我们的方法避免了将RPN与Fast R-CNN合并的复杂性，并且更容易训练，更快且更直接地集成到其它任务中。

与我们的方法直接相关的另一组方法，完全跳过提出步骤，直接预测多个类别的边界框和置信度。OverFeat[4]是滑动窗口方法的深度版本，在知道了底层目标类别的置信度之后，直接从最顶层的特征映射的每个位置预测边界框。YOLO[5]使用整个最顶层的特征映射来预测多个类别和边界框（这些类别共享）的置信度。我们的SSD方法属于这一类，因为我们没有提出步骤，但使用默认边界框。然而，我们的方法比现有方法更灵活，因为我们可以在不同尺度的多个特征映射的每个特征位置上使用不同长宽比的默认边界框。如果我们只从最顶层的特征映射的每个位置使用一个默认框，我们的SSD将具有与OverFeat[4]相似的架构；如果我们使用整个最顶层的特征映射，并添加一个全连接层进行预测来代替我们的卷积预测器，并且没有明确地考虑多个长宽比，我们可以近似地再现YOLO[5]。

### 6.结论

本文介绍了SSD，一种快速的单次多类别目标检测器。我们模型的一个关键特性是使用网络顶部多个特征映射的多尺度卷积边界框输出。这种表示使我们能够高效地建模可能的边界框形状空间。我们通过实验验证，在给定合适训练策略的情况下，大量仔细选择的默认边界框会提高性能。我们构建的SSD模型比现有的方法至少要多一个数量级的边界框预测采样位置，尺度和长宽比[5,7]。我们证明了给定相同的VGG-16基础架构，SSD在准确性和速度方面与其对应的最先进的目标检测器相比毫不逊色。在PASCAL VOC和COCO上，我们的SSD512模型的性能明显优于最先进的Faster R-CNN[2]，而速度提高了3倍。我们的实时SSD300模型运行速度为59FPS，比目前的实时YOLO[5]更快，同时显著提高了检测精度。

速度比YOLO V1快，效果比Faster R-CNN好！

### Reference

Uijlings, J.R., van de Sande, K.E., Gevers, T., Smeulders, A.W.: Selective search for object recognition. IJCV (2013)

Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks. In: NIPS. (2015)

He, K., Zhang, X., Ren, S., Sun, J.:Deep residual learning for image recognition. In:CVPR. (2016)

Sermanet, P., Eigen, D., Zhang, X., Mathieu, M., Fergus, R., LeCun, Y.: Overfeat:Integrated recognition, localization and detection using convolutional networks. In: ICLR. (2014)

Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: CVPR. (2016)

Girshick, R.: Fast R-CNN. In: ICCV. (2015)

Erhan, D., Szegedy, C., Toshev, A., Anguelov, D.: Scalable object detection using deep neural networks. In: CVPR. (2014)

Szegedy, C., Reed, S., Erhan, D., Anguelov, D.: Scalable, high-quality object detection. arXiv preprint arXiv:1412.1441 v3 (2015)

He, K., Zhang, X., Ren, S., Sun, J.: Spatial pyramid pooling in deep convolutional networks for visual recognition. In: ECCV. (2014)

Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: CVPR. (2015)

Hariharan, B., Arbeláez, P., Girshick, R., Malik, J.: Hypercolumns for object segmentation and fine-grained localization. In: CVPR. (2015)

Liu, W., Rabinovich, A., Berg, A.C.: ParseNet: Looking wider to see better.In:ILCR.(2016)

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Object detector semerge in deep scene cnns. In: ICLR. (2015)

Howard, A.G.: Some improvements on deep convolutional neural network based image classification. arXiv preprint arXiv:1312.5402 (2013)

Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: NIPS. (2015)

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: Imagenet large scale visual recognition challenge. IJCV (2015)

Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L.: Semantic image segmentation with deep convolutional nets and fully connected crfs. In: ICLR. (2015)

Holschneider, M., Kronland-Martinet, R., Morlet, J., Tchamitchian, P.: Areal-time algorithm for signal analysis with the help of the wavelet transform. In: Wavelets. Springer (1990) 286–297

Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S., Darrell, T.: Caffe: Convolutional architecture for fast feature embedding. In: MM. (2014)

Glorot, X., Bengio, Y.: Understanding the difficulty of training deep feedforward neural networks. In: AISTATS. (2010)

Hoiem, D., Chodpathumwan, Y., Dai, Q.: Diagnosing error in object detectors. In: ECCV 2012. (2012)

Girshick, R., Donahue, J., Darrell, T., Malik, J.: Rich feature hierarchies for accurate object detection and semantic segmentation. In: CVPR. (2014)

Zhang, L., Lin, L., Liang, X., He, K.: Is faster r-cnn doing well for pedestrian detection. In: ECCV. (2016)

Bell, S., Zitnick, C.L., Bala, K., Girshick, R.: Inside-outside net:Detecting objects in context with skip pooling and recurrent neural networks. In: CVPR. (2016)

COCO: Common Objects in Context. http://mscoco.org/dataset/#detections-leaderboard (2016) [Online; accessed 25-July-2016].

Felzenszwalb, P., McAllester, D., Ramanan, D.: A discriminatively trained, multiscale, deformable part model. In: CVPR. (2008)

------


## DSSD

------


## FSSD

------

## ESSD

------