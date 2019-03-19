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

### 2.YOLO V2


------
### 3.YOLO V3