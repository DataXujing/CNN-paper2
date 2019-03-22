## YOLO 系列目标检测方法

------

### 1.YOLO V1： YOU Only Look Once: Unified,Real-Time,Object Detection

------

#### 0.摘要

我们提出了YOLO，一种新的目标检测方法。以前的目标检测工作重新利用分类器来执行检测。相反，我们将目标检测框架看作回归问题从空间上分割边界框和相关的类别概率。单个神经网络在一次评估中直接从完整图像上预测边界框和类别概率。由于整个检测流水线是单一网络，因此可以直接对检测性能进行端到端的优化。

我们的统一架构非常快。我们的基础YOLO模型以45帧/秒的速度实时处理图像。网络的一个较小版本，快速YOLO，每秒能处理惊人的155帧，同时实现其它实时检测器两倍的mAP。与最先进的检测系统相比，YOLO产生了更多的定位误差，但不太可能在背景上的预测假阳性。最后，YOLO学习目标非常通用的表示。当从自然图像到艺术品等其它领域泛化时，它都优于其它检测方法，包括DPM和R-CNN。

#### 1.引言

人们瞥一眼图像，立即知道图像中的物体是什么，它们在哪里以及它们如何相互作用。人类的视觉系统是快速和准确的，使我们能够执行复杂的任务，如驾驶时没有多少有意识的想法。快速，准确的目标检测算法可以让计算机在没有专门传感器的情况下驾驶汽车，使辅助设备能够向人类用户传达实时的场景信息，并表现出对一般用途和响应机器人系统的潜力。

<div align=center>
<img src="zh-cn/img/yolov1/p1.png" />
</div>

*计算机视觉任务（来源: cs231n）*

目前的检测系统重用分类器来执行检测。为了检测目标，这些系统为该目标提供一个分类器，并在不同的位置对其进行评估，并在测试图像中进行缩放。像可变形部件模型（DPM）这样的系统使用滑动窗口方法，其分类器在整个图像的均匀间隔的位置上运行[10]。

最近的方法，如R-CNN使用区域提出方法首先在图像中生成潜在的边界框，然后在这些提出的框上运行分类器。在分类之后，后处理用于细化边界框，消除重复的检测，并根据场景中的其它目标重新定位边界框[13]。这些复杂的流程很慢，很难优化，因为每个单独的组件都必须单独进行训练。这类算法基于Region Proposal的R-CNN系算法（R-CNN，Fast R-CNN, Faster R-CNN），它们是two-stage的，需要先使用启发式方法（selective search）或者CNN网络（RPN）产生Region Proposal，然后再在Region Proposal上做分类与回归。。而另一类是Yolo，SSD这类one-stage算法，其仅仅使用一个CNN网络直接预测不同目标的类别与位置。第一类方法是准确度高一些，但是速度慢，但是第二类算法是速度快，但是准确性要低一些。这可以在图2中看到。本文介绍的是Yolo算法。

<div align=center>
<img src="zh-cn/img/yolov1/p2.jpg" />
</div>

*目标检测算法进展与对比*

**滑动窗口与CNN**: 在介绍Yolo算法之前，首先先介绍一下滑动窗口技术，这对我们理解Yolo算法是有帮助的。采用滑动窗口的目标检测算法思路非常简单，它将检测问题转化为了图像分类问题。其基本原理就是采用不同大小和比例（宽高比）的窗口在整张图片上以一定的步长进行滑动，然后对这些窗口对应的区域做图像分类，这样就可以实现对整张图片的检测了，如下图3所示，如DPM就是采用这种思路。但是这个方法有致命的缺点，就是你并不知道要检测的目标大小是什么规模，所以你要设置不同大小和比例的窗口去滑动，而且还要选取合适的步长。但是这样会产生很多的子区域，并且都要经过分类器去做预测，这需要很大的计算量，所以你的分类器不能太复杂，因为要保证速度。解决思路之一就是减少要分类的子区域，这就是R-CNN的一个改进策略，其采用了selective search方法来找到最有可能包含目标的子区域（Region Proposal），其实可以看成采用启发式方法过滤掉很多子区域，这会提升效率。

<div align=center>
<img src="zh-cn/img/yolov1/p3.png" />
</div>

*采用滑动窗口进行目标检测（来源：deeplearning.ai）*

如果你使用的是CNN分类器，那么滑动窗口是非常耗时的。但是结合卷积运算的特点，我们可以使用CNN实现更高效的滑动窗口方法。这里要介绍的是一种全卷积的方法，简单来说就是网络中用卷积层代替了全连接层，如图4所示。输入图片大小是16x16，经过一系列卷积操作，提取了2x2的特征图，但是这个2x2的图上每个元素都是和原图是一一对应的，如图上蓝色的格子对应蓝色的区域，这不就是相当于在原图上做大小为14x14的窗口滑动，且步长为2，共产生4个字区域。最终输出的通道数为4，可以看成4个类别的预测概率值，这样一次CNN计算就可以实现窗口滑动的所有子区域的分类预测。这其实是overfeat算法的思路。之所可以CNN可以实现这样的效果是因为卷积操作的特性，就是图片的空间位置信息的不变性，尽管卷积过程中图片大小减少，但是位置对应关系还是保存的。说点题外话，这个思路也被R-CNN借鉴，从而诞生了Fast R-cNN算法。

<div align=center>
<img src="zh-cn/img/yolov1/p4.png" />
</div>

*滑动窗口的CNN实现（来源：deeplearning.ai）*

上面尽管可以减少滑动窗口的计算量，但是只是针对一个固定大小与步长的窗口，这是远远不够的。Yolo算法很好的解决了这个问题，它不再是窗口滑动了，而是直接将原始图片分割成互不重合的小方块，然后通过卷积最后生产这样大小的特征图，基于上面的分析，可以认为特征图的每个元素也是对应原始图片的一个小方块，然后用每个元素来可以预测那些中心点在该小方格内的目标，这就是Yolo算法的朴素思想。

YOLO很简单：参见下图。单个卷积网络同时预测这些盒子的多个边界框和类概率。YOLO在全图像上训练并直接优化检测性能。这种统一的模型比传统的目标检测方法有一些好处。

<div align=center>
<img src="zh-cn/img/yolov1/p5.png" />
</div>

*YOLO检测系统。用YOLO处理图像简单直接。我们的系统（1）将输入图像调整为448×448，（2）在图像上运行单个卷积网络，以及（3）由模型的置信度对所得到的检测进行阈值处理。*


如图所示，使用YOLO来检测物体，其流程是非常简单明了的： 

1、将图像resize到448 * 448作为神经网络的输入 

2、运行神经网络，得到一些bounding box坐标、box中包含物体的置信度和class probabilities 

3、进行非极大值抑制，筛选Boxes

下图是各物体检测系统的检测流程对比：

<div align=center>
<img src="zh-cn/img/yolov1/p6.png" />
</div>


YOLO模型相对于之前的物体检测方法有多个优点：

1、YOLO检测物体非常快。 

因为没有复杂的检测流程，只需要将图像输入到神经网络就可以得到检测结果，YOLO可以非常快的完成物体检测任务。标准版本的YOLO在Titan X 的 GPU 上能达到45 FPS。更快的Fast YOLO检测速度可以达到155 FPS。而且，YOLO的mAP是之前其他实时物体检测系统的两倍以上。关于我们的系统在网络摄像头上实时运行的演示，请参阅我们的项目网页：<http://pjreddie.com/yolo/>。

2、YOLO可以很好的避免背景错误，产生false positives。

不像其他物体检测系统使用了滑窗或region proposal，分类器只能得到图像的局部信息。YOLO在训练和测试时都能够看到一整张图像的信息，因此YOLO在检测物体时能很好的利用上下文信息，从而不容易在背景上预测出错误的物体信息。和Fast-R-CNN相比，YOLO的背景错误不到Fast-R-CNN的一半。

3、YOLO可以学到物体的泛化特征。 

当YOLO在自然图像上做训练，在艺术作品上做测试时，YOLO表现的性能比DPM、R-CNN等之前的物体检测系统要好很多。因为YOLO可以学习到高度泛化的特征，从而迁移到其他领域。

尽管YOLO有这些优点，它也有一些缺点：

1、YOLO的物体检测精度低于其他state-of-the-art的物体检测系统。 

2、YOLO容易产生物体的定位错误。 

3、YOLO对小物体的检测效果不好（尤其是密集的小物体，因为一个栅格只能预测2个物体）。


#### 2.统一检测(Unified Detection)

我们将目标检测的单独组件集成到单个神经网络中。我们的网络使用整个图像的特征来预测每个边界框。它还可以同时预测一张图像中的所有类别的所有边界框。这意味着我们的网络全面地推理整张图像和图像中的所有目标。YOLO设计可实现端到端训练和实时的速度，同时保持较高的平均精度。

我们的系统将输入图像分成S×S的网格。如果一个目标的中心落入一个网格单元中，该网格单元负责检测该目标。

<div align=center>
<img src="zh-cn/img/yolov1/p7.png" />
</div>

每一个栅格预测B个bounding boxes，以及这些bounding boxes的confidence scores。 
这个 confidence scores反映了模型对于这个栅格的预测：该栅格是否含有物体，以及这个box的坐标预测的有多准。
公式定义如下： 

<div align=center>
<img src="zh-cn/img/yolov1/p8.jpg" />
</div>

如果这个栅格中不存在一个 object，则confidence score应该为0；否则的话，confidence score则为 predicted bounding box与 ground truth box之间的 IOU（intersection over union）

YOLO对每个bounding box有5个predictions：`x, y, w, h,confidence`。 
坐标x,y代表了预测的bounding box的中心与栅格边界的相对值。 
坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例。 
confidence就是预测的bounding box和ground truth box的IOU值。 

<div align=center>
<img src="zh-cn/img/yolov1/p9.gif" />
</div>

每一个栅格还要预测C个 conditional class probability（条件类别概率）：Pr(Classi|Object)。即在一个栅格包含一个Object的前提下，它属于某个类的概率。 我们只为每个栅格预测一组（C个）类概率，而不考虑框B的数量。

<div align=center>
<img src="zh-cn/img/yolov1/p10.jpg" />
</div>

注意： 
+ conditional class probability信息是针对每个网格的。 
+ confidence信息是针对每个bounding box的。

在测试阶段，将每个栅格的conditional class probabilities与每个 bounding box的 confidence相乘： 

<div align=center>
<img src="zh-cn/img/yolov1/p11.png" />
</div>

它为我们提供了每个框特定类别的置信度分数。这些分数编码了该类出现在框中的概率以及预测框拟合目标的程度。这乘积既包含了bounding box中预测的class的 probability信息，也反映了bounding box是否含有Object和bounding box坐标的准确度。

<div align=center>
<img src="zh-cn/img/yolov1/p12.gif" />
</div>

将YOLO用于PASCAL VOC数据集时： 
论文使用的 `S=7`，即将一张图像分为`7×7=49`个栅格每一个栅格预测`B=2`个boxes（每个box有 `x,y,w,h,confidence`，5个预测值），同时`C=20`（PASCAL数据集中有20个类别）。 
因此，最后的prediction是`7×7×30` `{ 即S * S * ( B * 5 + C) }`的Tensor。

<div align=center>
<img src="zh-cn/img/yolov1/p13.jpg" />
</div>

<div align=center>
<img src="zh-cn/img/yolov1/p14.gif" />
</div>

<div align=center>
<img src="zh-cn/img/yolov1/p15.png" />
</div>


##### 2.1网络设计

我们将此模型作为卷积神经网络来实现，并在Pascal VOC检测数据集[9]上进行评估。网络的初始卷积层从图像中提取特征，而全连接层预测输出概率和坐标。

我们的网络架构受到GoogLeNet图像分类模型的启发[34]。我们的网络有24个卷积层，后面是2个全连接层。我们只使用1×1降维层，后面是3×3卷积层，这与Lin等人[22]类似，而不是GoogLeNet使用的Inception模块。完整的网络如下图所示。

<div align=center>
<img src="zh-cn/img/yolov1/p16.png" />
</div>

*架构。我们的检测网络有24个卷积层，其次是2个全连接层。交替1×1卷积层减少了前面层的特征空间。我们在ImageNet分类任务上以一半的分辨率（224×224的输入图像）预训练卷积层，然后将分辨率加倍来进行检测。*

<div align=center>
<img src="zh-cn/img/yolov1/p17.png" />
</div>

##### 2.2训练

我们在ImageNet 1000类竞赛数据集[30]上预训练我们的卷积图层。对于预训练，我们使用图3中的前20个卷积层，接着是平均池化层和全连接层。我们对这个网络进行了大约一周的训练，并且在ImageNet 2012验证集上获得了单一裁剪图像88%的top-5准确率，与Caffe模型池中的GoogLeNet模型相当。我们使用Darknet框架进行所有的训练和推断[26]。关于Darknet介绍参考<https://cloud.tencent.com/developer/news/76803>。

然后我们转换模型来执行检测。Ren等人表明，预训练网络中增加卷积层和连接层可以提高性能[29]。按照他们的例子，我们添加了四个卷积层和两个全连接层，并且具有随机初始化的权重。检测通常需要细粒度的视觉信息，因此我们将网络的输入分辨率从224×224变为448×448。

我们的最后一层预测类概率和边界框坐标。我们通过图像宽度和高度来规范边界框的宽度和高度，使它们落在0和1之间。我们将边界框x和y坐标参数化为特定网格单元位置的偏移量，所以它们边界也在0和1之间。

我们对最后一层使用线性激活函数，所有其它层使用下面的漏泄修正线性激活（Leaky Relu）为了防止过拟合，在第一个全连接层后面接了一个 ratio=0.5 的 Dropout 层。 
为了提高精度，对原始图像做数据提升。：

<div align=center>
<img src="zh-cn/img/yolov1/p18.png" />
</div>

**损失函数**

损失函数的设计目标就是让坐标（x,y,w,h），confidence，classification 这个三个方面达到很好的平衡。 
简单的全部采用了sum-squared error loss来做这件事会有以下不足： 

a) 8维的localization error和20维的classification error同等重要显然是不合理的。 

b) 如果一些栅格中没有object（一幅图中这种栅格很多），那么就会将这些栅格中的bounding box的confidence 置为0，相比于较少的有object的栅格，这些不包含物体的栅格对梯度更新的贡献会远大于包含物体的栅格对梯度更新的贡献，这会导致网络不稳定甚至发散。

<div align=center>
<img src="zh-cn/img/yolov1/p19.png" />
</div>

解决方案如下： 

更重视8维的坐标预测，给这些损失前面赋予更大的loss weight, 记为 `λcoord `,在pascal VOC训练中取5。（上图蓝色框） 
对没有object的bbox的confidence loss，赋予小的loss weight，记为 `λnoobj` ，在pascal VOC训练中取0.5。（上图橙色框） 
有object的bbox的confidence loss (上图红色框) 和类别的loss （上图紫色框）的loss weight正常取1。

对不同大小的bbox预测中，相比于大bbox预测偏一点，小box预测偏相同的尺寸对IOU的影响更大。而sum-square error loss中对同样的偏移loss是一样。 
为了缓和这个问题，作者用了一个巧妙的办法，就是将box的width和height取平方根代替原本的height和width。 如下图：small bbox的横轴值较小，发生偏移时，反应到y轴上的loss（下图绿色）比big box(下图红色)要大。

<div align=center>
<img src="zh-cn/img/yolov1/p20.png" />
</div>

在 YOLO中，每个栅格预测多个bounding box，但在网络模型的训练中，希望每一个物体最后由一个bounding box predictor来负责预测。 
因此，当前哪一个predictor预测的bounding box与ground truth box的IOU最大，这个 predictor就负责 predict object。 
这会使得每个predictor可以专门的负责特定的物体检测。随着训练的进行，每一个 predictor对特定的物体尺寸、长宽比的物体的类别的预测会越来越好。

##### 2.3神经网络输出后的检测流程

在说明Yolo算法的预测过程之前，这里先介绍一下非极大值抑制算法（non maximum suppression, NMS），这个算法不单单是针对Yolo算法的，而是所有的检测算法中都会用到。NMS算法主要解决的是一个目标被多次检测的问题，如图11中人脸检测，可以看到人脸被多次检测，但是其实我们希望最后仅仅输出其中一个最好的预测框，比如对于美女，只想要红色那个检测结果。那么可以采用NMS算法来实现这样的效果：首先从所有的检测框中找到置信度最大的那个框，然后挨个计算其与剩余框的IOU，如果其值大于一定阈值（重合度过高），那么就将该框剔除；然后对剩余的检测框重复上述过程，直到处理完所有的检测框。Yolo预测过程也需要用到NMS算法。

<div align=center>
<img src="zh-cn/img/yolov1/p22.png" />
</div>

*NMS应用在人脸检测*

<div align=center>
<img src="zh-cn/img/yolov1/p23.gif" />
</div>

<div align=center>
<img src="zh-cn/img/yolov1/p24.gif" />
</div>

下面就来分析Yolo的预测过程，这里我们不考虑batch，认为只是预测一张输入图片。根据前面的分析，最终的网络输出是个边界框。

所有的准备数据已经得到了，那么我们先说第一种策略来得到检测框的结果，我认为这是最正常与自然的处理。首先，对于每个预测框根据类别置信度选取置信度最大的那个类别作为其预测标签，经过这层处理我们得到各个预测框的预测类别及对应的置信度值，其大小都是。一般情况下，会设置置信度阈值，就是将置信度小于该阈值的box过滤掉，所以经过这层处理，剩余的是置信度比较高的预测框。最后再对这些预测框使用NMS算法，最后留下来的就是检测结果。一个值得注意的点是NMS是对所有预测框一视同仁，还是区分每个类别，分别使用NMS。Ng在deeplearning.ai中讲应该区分每个类别分别使用NMS，但是看了很多实现，其实还是同等对待所有的框，我觉得可能是不同类别的目标出现在相同位置这种概率很低吧。

上面的预测方法应该非常简单明了，但是对于Yolo算法，其却采用了另外一个不同的处理思路（至少从C源码看是这样的），其区别就是先使用NMS，然后再确定各个box的类别。其基本过程如图12所示。对于98个boxes，首先将小于置信度阈值的值归0，然后分类别地对置信度值采用NMS，这里NMS处理结果不是剔除，而是将其置信度值归为0。最后才是确定各个box的类别，当其置信度值不为0时才做出检测结果输出。这个策略不是很直接，但是貌似Yolo源码就是这样做的。Yolo论文里面说NMS算法对Yolo的性能是影响很大的，所以可能这种策略对Yolo更好。但是我测试了普通的图片检测，两种策略结果是一样的

<div align=center>
<img src="zh-cn/img/yolov1/p25.gif" />
</div>

##### 2.4YOLO的限制

YOLO对边界框预测强加空间约束，因为每个网格单元只预测两个盒子，只能有一个类别。这个空间约束限制了我们的模型可以预测的邻近目标的数量。我们的模型与群组中出现的小物体（比如鸟群）进行斗争。

由于我们的模型学习从数据中预测边界框，因此它很难泛化到新的、不常见的方向比或配置的目标。我们的模型也使用相对较粗糙的特征来预测边界框，因为我们的架构具有来自输入图像的多个下采样层。

最后，当我们训练一个近似检测性能的损失函数时，我们的损失函数会同样的对待小边界框与大边界框的误差。大边界框的小误差通常是良性的，但小边界框的小误差对IOU的影响要大得多。我们的主要错误来源是不正确的定位。

#### 3.与其它检测系统的比较

目标检测是计算机视觉中的核心问题。检测流程通常从输入图像上（Haar [25]，SIFT [23]，HOG [4]，卷积特征[6]）提取一组鲁棒特征开始。然后，分类器[36,21,13,10]或定位器[1,32]被用来识别特征空间中的目标。这些分类器或定位器在整个图像上或在图像中的一些子区域上以滑动窗口的方式运行[35,15,39]。我们将YOLO检测系统与几种顶级检测框架进行比较，突出了关键的相似性和差异性。

**可变形部件模型**。可变形零件模型（DPM）使用滑动窗口方法进行目标检测[10]。DPM使用不相交的流程来提取静态特征，对区域进行分类，预测高评分区域的边界框等。我们的系统用单个卷积神经网络替换所有这些不同的部分。网络同时进行特征提取，边界框预测，非极大值抑制和上下文推理。网络内嵌训练特征而不是静态特征，并为检测任务优化它们。我们的统一架构导致了比DPM更快，更准确的模型。

**R-CNN**。R-CNN及其变种使用区域提出而不是滑动窗口来查找图像中的目标。选择性搜索[35]产生潜在的边界框，卷积网络提取特征，SVM对边界框进行评分，线性模型调整边界框，非极大值抑制消除重复检测。这个复杂流程的每个阶段都必须独立地进行精确调整，所得到的系统非常慢，测试时每张图像需要超过40秒[14]。

YOLO与R-CNN有一些相似之处。每个网格单元提出潜在的边界框并使用卷积特征对这些框进行评分。但是，我们的系统对网格单元提出进行了空间限制，这有助于缓解对同一目标的多次检测。我们的系统还提出了更少的边界框，每张图像只有98个，而选择性搜索则只有2000个左右。最后，我们的系统将这些单独的组件组合成一个单一的，共同优化的模型。

**其它快速检测器**。Fast R-CNN和Faster R-CNN通过共享计算和使用神经网络替代选择性搜索来提出区域加速R-CNN框架[14]，[28]。虽然它们提供了比R-CNN更快的速度和更高的准确度，但两者仍然不能达到实时性能。

许多研究工作集中在加快DPM流程上[31] [38] [5]。它们加速HOG计算，使用级联，并将计算推动到GPU上。但是，实际上只有30Hz的DPM [31]可以实时运行。

YOLO不是试图优化大型检测流程的单个组件，而是完全抛弃流程，被设计为快速检测。
像人脸或行人等单类别的检测器可以高度优化，因为他们必须处理更少的变化[37]。YOLO是一种通用的检测器，可以学习同时检测多个目标。

**Deep MultiBox**。与R-CNN不同，Szegedy等人训练了一个卷积神经网络来预测感兴趣区域[8]，而不是使用选择性搜索。MultiBox还可以通过用单类预测替换置信度预测来执行单目标检测。然而，MultiBox无法执行通用的目标检测，并且仍然只是一个较大的检测流程中的一部分，需要进一步的图像块分类。YOLO和MultiBox都使用卷积网络来预测图像中的边界框，但是YOLO是一个完整的检测系统。

**OverFeat**。Sermanet等人训练了一个卷积神经网络来执行定位，并使该定位器进行检测[32]。OverFeat高效地执行滑动窗口检测，但它仍然是一个不相交的系统。OverFeat优化了定位，而不是检测性能。像DPM一样，定位器在进行预测时只能看到局部信息。OverFeat不能推断全局上下文，因此需要大量的后处理来产生连贯的检测。

**MultiGrasp**。我们的工作在设计上类似于Redmon等[27]的抓取检测。我们对边界框预测的网格方法是基于MultiGrasp系统抓取的回归分析。然而，抓取检测比目标检测任务要简单得多。MultiGrasp只需要为包含一个目标的图像预测一个可以抓取的区域。不必估计目标的大小，位置或目标边界或预测目标的类别，只找到适合抓取的区域。YOLO预测图像中多个类别的多个目标的边界框和类别概率。



#### 4.实验

这里看一下Yolo算法在PASCAL VOC 2007数据集上的性能，这里Yolo与其它检测算法做了对比，包括DPM，R-CNN，Fast R-CNN以及Faster R-CNN。其对比结果如表1所示。与实时性检测方法DPM对比，可以看到Yolo算法可以在较高的mAP上达到较快的检测速度，其中Fast Yolo算法比快速DPM还快，而且mAP是远高于DPM。但是相比Faster R-CNN，Yolo的mAP稍低，但是速度更快。所以。Yolo算法算是在速度与准确度上做了折中。

<div align=center>
<img src="zh-cn/img/yolov1/p26.png" />
</div>

为了进一步分析Yolo算法，文章还做了误差分析，将预测结果按照分类与定位准确性分成以下5类：

+ Correct：类别正确，IOU>0.5；（准确度）
+ Localization：类别正确，0.1 < IOU<0.5（定位不准）；
+ Similar：类别相似，IOU>0.1；
+ Other：类别错误，IOU>0.1；
+ Background：对任何目标其IOU<0.1。（误把背景当物体）

Yolo与Fast R-CNN的误差对比分析如下图所示：

<div align=center>
<img src="zh-cn/img/yolov1/p27.png" />
</div>

*Yolo与Fast R-CNN的误差对比分析*

可以看到，Yolo的Correct的是低于Fast R-CNN。另外Yolo的Localization误差偏高，即定位不是很准确。但是Yolo的Background误差很低，说明其对背景的误判率较低。Yolo的那篇文章中还有更多性能对比，感兴趣可以看看。

现在来总结一下Yolo的优缺点:

首先是优点，Yolo采用一个CNN网络来实现检测，是单管道策略，其训练与预测都是end-to-end，所以Yolo算法比较简洁且速度快。

第二点由于Yolo是对整张图片做卷积，所以其在检测目标有更大的视野，它不容易对背景误判。其实我觉得全连接层也是对这个有贡献的，因为全连接起到了attention的作用。另外，Yolo的泛化能力强，在做迁移时，模型鲁棒性高。

最后不得不谈一下Yolo的缺点:首先Yolo各个单元格仅仅预测两个边界框，而且属于一个类别。对于小物体，Yolo的表现会不如人意。这方面的改进可以看SSD，其采用多尺度单元格。也可以看Faster R-CNN，其采用了anchor boxes。Yolo对于在物体的宽高比方面泛化率低，就是无法定位不寻常比例的物体。当然Yolo的定位不准确也是很大的问题。

#### 5.现实环境下的实时检测

YOLO是一种快速，精确的目标检测器，非常适合计算机视觉应用。我们将YOLO连接到网络摄像头，并验证它是否能保持实时性能，包括从摄像头获取图像并显示检测结果的时间。

由此产生的系统是交互式和参与式的。虽然YOLO单独处理图像，但当连接到网络摄像头时，其功能类似于跟踪系统，可在目标移动和外观变化时检测目标。系统演示和源代码可以在我们的项目网站上找到：http://pjreddie.com/yolo/。


#### 6.结论


我们介绍了YOLO，一种统一的目标检测模型。我们的模型构建简单，可以直接在整张图像上进行训练。与基于分类器的方法不同，YOLO直接在对应检测性能的损失函数上训练，并且整个模型联合训练。

快速YOLO是文献中最快的通用目的的目标检测器，YOLO推动了实时目标检测的最新技术。YOLO还很好地泛化到新领域，使其成为依赖快速，强大的目标检测应用的理想选择。


#### Reference

[0] <https://pjreddie.com/darknet/yolo/>,

<https://github.com/gliese581gg/YOLO_tensorflow>,

<https://blog.csdn.net/m0_37192554/article/details/81092514>,

<https://blog.csdn.net/hrsstudy/article/details/70305791>

[1] M. B. Blaschko and C. H. Lampert. Learning to localize objects with structured output regression. In Computer Vision–ECCV 2008, pages 2–15. Springer, 2008. 4

[2] L. Bourdev and J. Malik. Poselets: Body part detectors trained using 3d human pose annotations. In International Conference on Computer Vision (ICCV), 2009. 8

[3] H. Cai, Q. Wu, T. Corradi, and P. Hall. The cross-depiction problem: Computer vision algorithms for recognising objects in artwork and in photographs. arXiv preprint arXiv:1505.00110, 2015. 7

[4] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on, volume 1, pages 886–893. IEEE, 2005. 4, 8

[5] T. Dean, M. Ruzon, M. Segal, J. Shlens, S. Vijaya-narasimhan, J. Yagnik, et al. Fast, accurate detection of 100,000 object classes on a single machine. In Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on, pages 1814–1821. IEEE, 2013. 5

[6] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. arXiv preprint arXiv:1310.1531, 2013. 4

[7] J. Dong, Q. Chen, S. Yan, and A. Yuille. Towards unified object detection and semantic segmentation. In Computer Vision–ECCV 2014, pages 299–314. Springer, 2014. 7

[8] D.Erhan, C.Szegedy, A.Toshev, and D.Anguelov. Scalable object detection using deep neural networks. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 2155–2162. IEEE, 2014. 5, 6

[9] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes challenge: A retrospective. International Journal of Computer Vision, 111(1):98–136, Jan. 2015. 2

[10] P.F.Felzenszwalb, R.B.Girshick, D.McAllester, and D.Ramanan. Object detection with discriminatively trained part based models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(9):1627–1645, 2010. 1, 4

[11] S. Gidaris and N. Komodakis. Object detection via a multi-region & semantic segmentation-aware CNN model. CoRR, abs/1505.01749, 2015. 7

[12] S. Ginosar, D. Haas, T. Brown, and J. Malik. Detecting people in cubist art. In Computer Vision-ECCV 2014 Workshops, pages 101–116. Springer, 2014. 7

[13] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 580–587. IEEE, 2014. 1, 4, 7

[14] R. B. Girshick. Fast R-CNN. CoRR, abs/1504.08083, 2015. 2, 5, 6, 7

[15] S. Gould, T. Gao, and D. Koller. Region-based segmentation and object detection. In Advances in neural information processing systems, pages 655–663, 2009. 4

[16] B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Simultaneous detection and segmentation. In Computer Vision–ECCV 2014, pages 297–312. Springer, 2014. 7

[17] K.He, X.Zhang, S.Ren, and J.Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. arXiv preprint arXiv:1406.4729, 2014. 5

[18] G.E.Hinton, N.Srivastava, A.Krizhevsky, I.Sutskever, and R. R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012. 4

[19] D.Hoiem, Y.Chodpathumwan, and Q.Dai. Diagnosing error in object detectors. In Computer Vision–ECCV 2012, pages 340–353. Springer, 2012. 6

[20] K. Lenc and A. Vedaldi. R-cnn minus r. arXiv preprint arXiv:1506.06981, 2015. 5, 6

[21] R. Lienhart and J. Maydt. An extended set of haar-like features for rapid object detection. In Image Processing. 2002. Proceedings. 2002
International Conference on, volume 1, pages I–900. IEEE, 2002. 4

[22] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013. 2

[23] D. G. Lowe. Object recognition from local scale-invariant features. In Computer vision, 1999. The proceedings of the seventh IEEE international conference on, volume 2, pages 1150–1157. Ieee, 1999. 4

[24] D. Mishkin. Models accuracy on imagenet 2012 val. https://github.com/BVLC/caffe/wiki/ Models-accuracy-on-ImageNet-2012-val. Accessed: 2015-10-2. 3

[25] C. P. Papageorgiou, M. Oren, and T. Poggio. A general framework for object detection. In Computer vision, 1998. sixth international conference on, pages 555–562. IEEE, 1998. 4

[26] J. Redmon. Darknet: Open source neural networks in c. http://pjreddie.com/darknet/, 2013–2016. 3

[27] J.Redmon and A.Angelova. Real-time grasp detection using convolutional neural networks. CoRR, abs/1412.3128, 2014. 5

[28] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. arXiv preprint arXiv:1506.01497, 2015. 5, 6, 7

[29] S. Ren, K. He, R. B. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. CoRR, abs/1504.06066, 2015. 3, 7

[30] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 2015. 3

[31] M. A. Sadeghi and D. Forsyth. 30hz object detection with dpm v5. In Computer Vision–ECCV 2014, pages 65–79. Springer, 2014. 5, 6

[32] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. CoRR, abs/1312.6229, 2013. 4, 5

[33] Z.Shen and X.Xue. Do more dropouts in pool5 feature maps for better object detection. arXiv preprint arXiv:1409.6911, 2014. 7

[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014. 2

[35] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders. Selective search for object recognition. International journal of computer vision, 104(2):154–171, 2013. 4, 5

[36] P. Viola and M. Jones. Robust real-time object detection. International Journal of Computer Vision, 4:34–47, 2001. 4

[37] P. Viola and M. J. Jones. Robust real-time face detection. International journal of computer vision, 57(2):137–154, 2004. 5

[38] J. Yan, Z. Lei, L. Wen, and S. Z. Li. The fastest deformable part model for object detection. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 2497–2504. IEEE, 2014. 5, 6

[39] C.L.Zitnick and P.Dollár.Edgeboxes:Locating object proposals from edges. In Computer Vision–ECCV 2014, pages 391–405. Springer, 2014. 4

------

### 2.YOLO V2: YOLO9000:Better,Faster,Stronger

#### 0.摘要

我们引入了一个先进的实时目标检测系统YOLO9000，可以检测超过9000个目标类别。首先，我们提出了对YOLO检测方法的各种改进，既有新颖性，也有前期的工作。改进后的模型YOLOv2在PASCAL VOC和COCO等标准检测任务上是最先进的。使用一种新颖的，多尺度训练方法，同样的YOLOv2模型可以以不同的尺寸运行，从而在速度和准确性之间提供了一个简单的折衷。在67FPS时，YOLOv2在VOC 2007上获得了76.8 mAP。在40FPS时，YOLOv2获得了78.6 mAP，比使用ResNet的Faster R-CNN和SSD等先进方法表现更出色，同时仍然运行速度显著更快。最后我们提出了一种联合训练目标检测与分类的方法。使用这种方法，我们在COCO检测数据集和ImageNet分类数据集上同时训练YOLO9000。我们的联合训练允许YOLO9000预测未标注的检测数据目标类别的检测结果。我们在ImageNet检测任务上验证了我们的方法。YOLO9000在ImageNet检测验证集上获得19.7 mAP，尽管200个类别中只有44个具有检测数据。在没有COCO的156个类别上，YOLO9000获得16.0 mAP。但YOLO可以检测到200多个类别；它预测超过9000个不同目标类别的检测结果。并且它仍然能实时运行。

#### 1.引言

通用目的的目标检测应该快速，准确，并且能够识别各种各样的目标。自从引入神经网络以来，检测框架变得越来越快速和准确。但是，大多数检测方法仍然受限于一小部分目标。

与分类和标记等其他任务的数据集相比，目前目标检测数据集是有限的。最常见的检测数据集包含成千上万到数十万张具有成百上千个标签的图像[3][10][2]。分类数据集有数以百万计的图像，数十或数十万个类别[20][2]。

我们希望检测能够扩展到目标分类的级别。但是，标注检测图像要比标注分类或贴标签要昂贵得多（标签通常是用户免费提供的）。因此，我们不太可能在近期内看到与分类数据集相同规模的检测数据集。

我们提出了一种新的方法来利用我们已经拥有的大量分类数据，并用它来扩大当前检测系统的范围。我们的方法使用目标分类的分层视图，允许我们将不同的数据集组合在一起。

使用这种方法我们训练YOLO9000，一个实时的目标检测器，可以检测超过9000种不同的目标类别。首先，我们改进YOLO基础检测系统，产生最先进的实时检测器YOLOv2。然后利用我们的数据集组合方法和联合训练算法对来自ImageNet的9000多个类别以及COCO的检测数据训练了一个模型。



<div align=center>
<img src="zh-cn/img/yolov2/p1.jpg" />
</div>

*图1：YOLO9000。YOLO9000可以实时检测许多目标类别。*

所有代码和预训练模型都可在线获得：http://pjreddie.com/yolo9000/。

#### 2.更好（Better）

与最先进的检测系统相比，YOLO有许多缺点。YOLO与Fast R-CNN相比的误差分析表明，**YOLO造成了大量的定位误差**。此外，与基于区域提出的方法相比，**YOLO召回率相对较低(TP/(TP+FN))**。因此，我们主要侧重于提高召回率和改进定位，同时保持分类准确性。

计算机视觉一般趋向于更大，更深的网络[6][18][17]。更好的性能通常取决于训练更大的网络或将多个模型组合在一起。但是，**在YOLOv2中，我们需要一个更精确的检测器**，它仍然很快。我们不是扩大我们的网络，而是简化网络，然后让表示更容易学习。我们将过去的工作与我们自己的新概念汇集起来，以提高YOLO的性能。表2列出了结果总结。

<div align=center>
<img src="zh-cn/img/yolov2/p2.png" />
</div>

*表2：从YOLO到YOLOv2的路径。列出的大部分设计决定都会导致mAP的显著增加。有两个例外是切换到具有Anchor Boxes的一个全卷积网络和使用新网络。切换到Anchor Boxes风格的方法增加了召回，而不改变mAP，而使用新网络会削减33%的计算量*

**Trick1: 批标准化**。批标准化导致收敛性的显著改善，同时消除了对其他形式正则化的需求[7]。通过在YOLO的所有卷积层上添加批标准化，我们在mAP中获得了超过2%的改进。批标准化也有助于模型正则化。通过批标准化，我们可以从模型中删除丢弃而不会过拟合。


**Trick2: 高分辨率分类器**(High Resolution Classifier)。所有最先进的检测方法都使用在ImageNet[16]上预训练的分类器。从AlexNet开始，大多数分类器对小于256×256[8]的输入图像进行操作。原来的YOLO以224×224的分辨率训练分类器网络，并将分辨率提高到448进行检测。这意味着网络必须同时切换到学习目标检测和调整到新的输入分辨率。

对于YOLOv2，我们首先ImageNet上以448×448的分辨率对分类网络进行10个迭代周期的微调。这给了网络时间来调整其滤波器以便更好地处理更高分辨率的输入。然后，我们在检测上微调得到的网络。这个高分辨率分类网络使我们增加了近4%的mAP。


**Trick3: 具有Anchor Boxex的卷积**。YOLO直接使用卷积特征提取器顶部的全连接层来预测边界框的坐标。Faster R-CNN使用手动选择的先验来预测边界框而不是直接预测坐标[15]。Faster R-CNN中的区域提出网络（RPN）仅使用卷积层来预测锚盒(Anchor Boxes)的偏移和置信度。由于预测层是卷积的，所以RPN在特征映射的每个位置上预测这些偏移(offset)。预测偏移而不是坐标简化了问题，并且使网络更容易学习。


我们从YOLO中移除全连接层，并使用锚盒来预测边界框。首先，我们消除了一个池化层，使网络卷积层输出具有更高的分辨率。我们还缩小了网络，操作416×416的输入图像而不是448×448。我们这样做是因为我们要在我们的特征映射中有奇数个位置，所以只有一个中心单元。目标，特别是大目标，往往占据图像的中心，所以在中心有一个单独的位置来预测这些目标，而不是四个都在附近的位置是很好的。YOLO的卷积层将图像下采样32倍，所以通过使用416的输入图像，我们得到了13×13的输出特征映射。

当我们移动到锚盒时，我们也将类预测机制与空间位置分离，预测每个锚盒的类别和目标。在YOLO之后，目标预测仍然预测了实际值和提出的边界框的IOU，并且类别预测了当存在目标时该类别的条件概率。

使用锚盒，我们在精度上得到了一个小下降。YOLO每张图像只预测98个边界框，但是使用锚盒我们的模型预测超过一千。如果没有锚盒，我们的中间模型将获得69.5的mAP，召回率为81%。具有锚盒我们的模型得到了69.2 mAP，召回率为88%。尽管mAP下降，但召回率的上升意味着我们的模型有更大的提升空间。

**Trick4: 维度聚类**。当锚盒与YOLO一起使用时，我们遇到了两个问题。首先是边界框尺寸是手工挑选的。网络可以学习适当调整边界框，但如果我们为网络选择更好的先验，我们可以使网络更容易学习它以便预测好的检测。

我们不用手工选择先验，而是在训练集边界框上运行`k-means聚类`，自动找到好的先验。如果我们使用具有欧几里得距离的标准k-means，那么较大的边界框比较小的边界框产生更多的误差。然而，我们真正想要的是导致好的IOU分数的先验，这是独立于边界框大小的。因此，对于我们的距离度量，我们使用：
<div align="center">
d(box,centroid)=1−IOU(box,centroid)
</div>

我们运行各种k值的k-means，并画出平均IOU与最接近的几何中心，见图2。我们选择`k=5`作为模型复杂性和高召回率之间的良好折衷。聚类中心与手工挑选的锚盒明显不同。有更短更宽的边界框和更高更细的边界框。

<div align=center>
<img src="zh-cn/img/yolov2/p3.png" />
</div>

*图2：VOC和COCO的聚类边界框尺寸。我们对边界框的维度进行k-means聚类，以获得我们模型的良好先验。左图显示了我们通过对k的各种选择得到的平均IOU。我们发现k=5给出了一个很好的召回率与模型复杂度的权衡。右图显示了VOC和COCO的相对中心。这两种先验都赞成更薄更高的边界框，而COCO比VOC在尺寸上有更大的变化。*

在表1中我们将平均IOU与我们聚类策略中最接近的先验以及手工选取的锚盒进行了比较。仅有5个先验中心的平均IOU为61.0，其性能类似于9个锚盒的60.9。如果我们使用9个中心，我们会看到更高的平均IOU。这表明使用k-means来生成我们的边界框会以更好的表示开始训练模型，并使得任务更容易学习。


<div align=center>
<img src="zh-cn/img/yolov2/p4.png" />
</div>

*表1：VOC 2007上最接近先验的边界框平均IOU。VOC 2007上目标的平均IOU与其最接近的，使用不同生成方法之前未经修改的平均值。聚类结果比使用手工选择的先验结果要更好。*


**Trick5:直接位置预测**: 当YOLO使用锚盒时，我们会遇到第二个问题：模型不稳定，特别是在早期的迭代过程中。大部分

<div align=center>
<img src="zh-cn/img/yolov2/p5.png" />
</div>

<div align=center>
<img src="zh-cn/img/yolov2/p6.png" />
</div>

这个公式是不受限制的，所以任何锚盒都可以在图像任一点结束，而不管在哪个位置预测该边界框。随机初始化模型需要很长时间才能稳定以预测合理的偏移量。

我们没有预测偏移量，而是按照YOLO的方法预测相对于网格单元位置的位置坐标。这限制了落到0和1之间的真实值。我们使用逻辑激活来限制网络的预测落在这个范围内。

<div align=center>
<img src="zh-cn/img/yolov2/p7.png" />
</div>

<div align=center>
<img src="zh-cn/img/yolov2/p8.png" />
</div>

*图3：具有维度先验和位置预测的边界框。我们预测边界框的宽度和高度作为聚类中心的偏移量。我们使用sigmoid函数预测边界框相对于滤波器应用位置的中心坐标。*

由于我们限制位置预测参数化更容易学习，使网络更稳定。使用维度聚类以及直接预测边界框中心位置的方式比使用锚盒的版本将YOLO提高了近5%。

**Trick6: 细粒度功能**。这个修改后的YOLO在13×13特征映射上预测检测结果。虽然这对于大型目标来说已经足够了，但它可以从用于定位较小目标的更细粒度的特征中受益。Faster R-CNN和SSD都在网络的各种特征映射上运行他们提出的网络，以获得一系列的分辨率。我们采用不同的方法，仅仅添加一个通道层，从26x26分辨率的更早层中提取特征。

上述网络上的修改使YOLO最终在13x13的特征图上进行预测，虽然这足以胜任大尺度物体的检测，但是用上细粒度特征的话，这可能对小尺度的物体检测有帮助。Faser R-CNN和SSD都在不同层次的特征图上产生区域建议（SSD直接就可看得出来这一点），获得了多尺度的适应性。这里使用了一种不同的方法，简单添加了一个转移层（ passthrough layer），这一层要把浅层特征图（分辨率为26x26，是底层分辨率4倍）连接到深层特征图。

关于passthrough layer，具体来说就是特征重排（不涉及到参数学习），前面26x26x512的特征图使用按行和按列隔行采样的方法，就可以得到4个新的特征图，维度都是13x13x512，然后做concat操作，得到13x13x2048的特征图，将其拼接到后面的层，相当于做了一次特征融合，有利于检测小目标。


**Trick7: 多尺度训练**。原来的YOLO使用448×448的输入分辨率。通过添加锚盒，我们将分辨率更改为416×416。但是，由于我们的模型只使用卷积层和池化层，因此它可以实时调整大小。我们希望YOLOv2能够鲁棒的运行在不同大小的图像上，因此我们可以将其训练到模型中。

我们没有固定的输入图像大小，每隔几次迭代就改变网络。每隔10个批次我们的网络会随机选择一个新的图像尺寸大小。由于我们的模型缩减了32倍，我们从下面的32的倍数中选择：{320,352，…，608}。因此最小的选项是320×320，最大的是608×608。我们调整网络的尺寸并继续训练。

这个制度迫使网络学习如何在各种输入维度上做好预测。这意味着相同的网络可以预测不同分辨率下的检测结果。在更小尺寸上网络运行速度更快，因此YOLOv2在速度和准确性之间提供了一个简单的折衷。

在低分辨率YOLOv2作为一个便宜，相当准确的检测器。在288×288时，其运行速度超过90FPS，mAP与Fast R-CNN差不多。这使其成为小型GPU，高帧率视频或多视频流的理想选择。

在高分辨率下，YOLOv2是VOC 2007上最先进的检测器，达到了78.6 mAP，同时仍保持运行在实时速度之上。请参阅表3，了解YOLOv2与VOC 2007其他框架的比较。图4

<div align=center>
<img src="zh-cn/img/yolov2/p9.png" />
</div>

*表3：PASCAL VOC 2007的检测框架。YOLOv2比先前的检测方法更快，更准确。它也可以以不同的分辨率运行，以便在速度和准确性之间进行简单折衷。每个YOLOv2条目实际上是具有相同权重的相同训练模型，只是以不同的大小进行评估。所有的时间信息都是在Geforce GTX Titan X（原始的，而不是Pascal模型）上测得的。*

<div align=center>
<img src="zh-cn/img/yolov2/p10.png" />
</div>

*图4：VOC 2007上的准确性与速度。*


#### 3.更快(Faster)

希望检测准确,快速。大多数检测应用（如机器人或自动驾驶机车）依赖于低延迟预测。为了最大限度提高性能，从头开始设计YOLOv2。大多数检测框架依赖于VGG-16作为的基本特征提取器[17]。VGG-16是一个强大的，准确的分类网络，但它是不必要的复杂。在单张图像224×224分辨率的情况下VGG-16的卷积层运行一次传递需要306.90亿次浮点运算。

YOLO框架使用基于Googlenet架构[19]的自定义网络。这个网络比VGG-16更快，一次前馈传播只有85.2亿次的操作。然而，它的准确性比VGG-16略差。在ImageNet上，对于单张裁剪图像，224×224分辨率下的top-5准确率，YOLO的自定义模型获得了88.0%，而VGG-16则为90.0%。

**Darknet-19**,一个新的分类模型作为YOLOv2的基础。模型建立在网络设计先前工作以及该领域常识的基础上。与VGG模型类似，大多使用3×3滤波器，并在每个池化步骤之后使通道数量加倍[17]。按照Network in Network（NIN），使用全局平均池化做预测以及1×1滤波器来压缩3×3卷积之间的特征表示[9]。使用批标准化来稳定训练，加速收敛，并正则化模型[7]。

最终模型叫做Darknet-19，它有19个卷积层和5个最大池化层。完整描述请看表6。Darknet-19只需要55.8亿次运算来处理图像，但在ImageNet上却达到了72.9%的top-1准确率和91.2%的top-5准确率。


<div align=center>
<img src="zh-cn/img/yolov2/p11.png" />
</div>

*表6：Darknet-19*

如上所述，对224×224的图像进行初始训练之后，对网络在更大的尺寸448上进行了微调。对于这种微调，使用上述参数进行训练，但是只有10个迭代周期，并且以10−3的学习率开始。在这种更高的分辨率下，网络达到了76.5%的top-1准确率和93.3%的top-5准确率。

**检测训练**。修改这个网络进行检测，删除了最后一个卷积层，加上了三个具有1024个滤波器的3×3卷积层，其后是最后的1×1卷积层与我们检测需要的输出数量。对于VOC，我们预测5个边界框，每个边界框有5个坐标和20个类别，所以有125个滤波器。我们还添加了从最后的3×3×512层到倒数第二层卷积层的直通层，以便我们的模型可以使用细粒度特征。

训练网络160个迭代周期，初始学习率为10−3，在60个和90个迭代周期时将学习率除以10。使用0.0005的权重衰减和0.9的动量。对YOLO和SSD进行类似的数据增强，随机裁剪，色彩偏移等。对COCO和VOC使用相同的训练策略。


#### 4.更强（stronger） YOLO90000

提出了一个联合训练分类和检测数据的机制。使用标记为检测的图像来学习边界框坐标预测和目标之类的特定检测信息以及如何对常见目标进行分类。它使用仅具有类别标签的图像来扩展可检测类别的数量。

在训练期间，混合来自检测和分类数据集的图像。当网络看到标记为检测的图像时，可以基于完整的YOLOv2损失函数进行反向传播。当它看到一个分类图像时，只能从该架构的分类特定部分反向传播损失。

这种方法提出了一些挑战。检测数据集只有通用目标和通用标签，如“狗”或“船”。分类数据集具有更广更深的标签范围。ImageNet有超过一百种品种的狗，包括Norfolk terrier，Yorkshire terrier和Bedlington terrier。如果想在两个数据集上训练，需要一个连贯的方式来合并这些标签。

大多数分类方法使用跨所有可能类别的softmax层来计算最终的概率分布。使用softmax假定这些类是相互排斥的。这给数据集的组合带来了问题，例如你不想用这个模型来组合ImageNet和COCO，因为类Norfolk terrier和dog不是相互排斥的。

我们可以改为使用多标签模型来组合不假定互斥的数据集。这种方法忽略了我们已知的关于数据的所有结构，例如，所有的COCO类是互斥的。

**分层分类**。ImageNet标签是从WordNet中提取的，这是一个构建概念及其相互关系的语言数据库[12]。在WordNet中，Norfolk terrier和Yorkshire terrier都是terrier的下义词，terrier是一种hunting dog，hunting dog是dog，dog是canine等。分类的大多数方法为标签假设一个扁平结构，但是对于组合数据集，结构正是我们所需要的。

WordNet的结构是有向图，而不是树，因为语言是复杂的。例如，dog既是一种canine，也是一种domestic animal，它们都是WordNet中的同义词。我们不是使用完整的图结构，而是通过从ImageNet的概念中构建分层树来简化问题。

为了构建这棵树，我们检查了ImageNet中的视觉名词，并查看它们通过WordNet图到根节点的路径，在这种情况下是“物理对象”。许多同义词通过图只有一条路径，所以首先我们将所有这些路径添加到我们的树中。然后我们反复检查我们留下的概念，并尽可能少地添加生长树的路径。所以如果一个概念有两条路径到一个根，一条路径会给我们的树增加三条边，另一条只增加一条边，我们选择更短的路径。

<div align=center>
<img src="zh-cn/img/yolov2/p12.png" />
</div>

为了验证这种方法，我们在使用1000类ImageNet构建的WordTree上训练Darknet-19模型。为了构建WordTree1k，我们添加了所有将标签空间从1000扩展到1369的中间节点。在训练过程中，我们将真实标签向树上面传播，以便如果图像被标记为Norfolk terrier，则它也被标记为dog和mammal等。为了计算条件概率，我们的模型预测了具有1369个值的向量，并且我们计算了相同概念的下义词在所有同义词集上的softmax，见图5。

<div align=center>
<img src="zh-cn/img/yolov2/p13.png" />
</div>

*图5：在ImageNet与WordTree上的预测。大多数ImageNet模型使用一个较大的softmax来预测概率分布。使用WordTree，我们可以在共同的下义词上执行多次softmax操作。*

<div align=center>
<img src="zh-cn/img/yolov2/p14.png" />
</div>

使用与以前相同的训练参数，我们的分级Darknet-19达到71.9%的top-1准确率和90.4%的top-5准确率。尽管增加了369个额外的概念，而且我们的网络预测了一个树状结构，但我们的准确率仅下降了一点点。以这种方式进行分类也有一些好处。在新的或未知的目标类别上性能会优雅地降低。例如，如果网络看到一只狗的照片，但不确定它是什么类型的狗，它仍然会高度自信地预测“狗”，但是在下义位扩展之间有更低的置信度。

这个构想也适用于检测。现在，我们不是假定每张图像都有一个目标，而是使用YOLOv2的目标预测器给我们Pr(physical object)的值。检测器预测边界框和概率树。我们遍历树，在每个分割中采用最高的置信度路径，直到达到某个阈值，然后我们预测目标类

**联合分类和检测**。现在我们可以使用WordTree组合数据集，我们可以在分类和检测上训练联合模型。我们想要训练一个非常大规模的检测器，所以我们使用COCO检测数据集和完整的ImageNet版本中的前9000个类来创建我们的组合数据集。我们还需要评估我们的方法，以便从ImageNet检测挑战中添加任何尚未包含的类。该数据集的相应WordTree有9418个类别。ImageNet是一个更大的数据集，所以我们通过对COCO进行过采样来平衡数据集，使得ImageNet仅仅大于4:1的比例。

使用这种联合训练，YOLO9000学习使用COCO中的检测数据来查找图像中的目标，并学习使用来自ImageNet的数据对各种目标进行分类。

我们在ImageNet检测任务上评估YOLO9000。ImageNet的检测任务与COCO共享44个目标类别，这意味着YOLO9000只能看到大多数测试图像的分类数据，而不是检测数据。YOLO9000在从未见过任何标记的检测数据的情况下，整体上获得了19.7 mAP，在不相交的156个目标类别中获得了16.0 mAP。这个mAP高于DPM的结果，但是YOLO9000在不同的数据集上训练，只有部分监督[4]。它也同时检测9000个其他目标类别，所有的都是实时的。

当我们分析YOLO9000在ImageNet上的表现时，我们发现它很好地学习了新的动物种类，但是却在像服装和设备这样的学习类别中挣扎。新动物更容易学习，因为目标预测可以从COCO中的动物泛化的很好。相反，COCO没有任何类型的衣服的边界框标签，只针对人，因此YOLO9000正在努力建模“墨镜”或“泳裤”等类别。

#### 5.结论

YOLOv2和YOLO9000，两个实时检测系统。YOLOv2在各种检测数据集上都是最先进的，也比其他检测系统更快。此外，它可以运行在各种图像大小，以提供速度和准确性之间的平滑折衷。

YOLO9000是一个通过联合优化检测和分类来检测9000多个目标类别的实时框架。使用`WordTree`将各种来源的数据和联合优化技术相结合，在ImageNet和COCO上同时进行训练。YOLO9000是在检测和分类之间缩小数据集大小差距的重要一步。

许多技术都可以泛化到目标检测之外。对ImageNet的`WordTree`表示为图像分类提供了更丰富，更详细的输出空间。使用分层分类的数据集组合在分类和分割领域将是有用的。像多尺度训练这样的训练技术可以为各种视觉任务提供益处。

对于未来的工作，希望使用类似的技术来进行弱监督的图像分割。还计划使用更强大的匹配策略来改善检测结果，以在训练期间将弱标签分配给分类数据。计算机视觉受到大量标记数据的束缚。继续寻找方法，将不同来源和数据结构的数据整合起来，形成更强大的视觉世界模型。



#### Reference

[1] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. arXiv preprint arXiv:1512.04143, 2015. 6

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei- Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 248–255. IEEE, 2009. 1

[3] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303– 338, 2010. 1

[4] P. F. Felzenszwalb, R. B. Girshick, and D. McAllester. Discriminatively trained deformable part models, release 4. http://people.cs.uchicago.edu/pff/latent-release4/. 8

[5] R. B. Girshick. Fast R-CNN. CoRR, abs/1504.08083, 2015. 4, 5, 6

[6] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015. 2, 4, 5

[7] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015. 2, 5

[8] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. 2

[9] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv preprint arXiv:1312.4400, 2013. 5

[10] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014. 1, 6

[11] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. E. Reed. SSD: single shot multibox detector. CoRR, abs/1512.02325, 2015. 4, 5, 6

[12] G. A. Miller, R. Beckwith, C. Fellbaum, D. Gross, and K. J. Miller. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3(4):235–244, 1990. 6

[13] J. Redmon. Darknet: Open source neural networks in c. http://pjreddie.com/darknet/, 2013–2016. 5

[14] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. arXiv preprint arXiv:1506.02640, 2015. 4, 5

[15] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal net- works. arXiv preprint arXiv:1506.01497, 2015. 2, 3, 4, 5, 6

[16] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 2015. 2

[17] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 2, 5

[18] C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. CoRR, abs/1602.07261, 2016. 2

[19] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014. 5

[20] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L.-J. Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016. 1


------

### 3.YOLO V3： An Incremental Improvement