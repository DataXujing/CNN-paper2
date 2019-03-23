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

**一篇‘不正经’的技术报告**

#### 0.摘要

YOLO3为YOLO提供了一系列更新！它包含一堆小设计，可以使系统的性能得到更新；也包含一个新训练的、非常棒的神经网络，虽然比上一版YOLO V2更大一些，但精度也提高了。不用担心，虽然体量大了点，它的速度还是有保障的。在输入320×320的图片后，YOLOv3能在22毫秒内完成处理，并取得28.2mAP的成绩。它的精度和SSD相当，但速度要快上3倍。和旧版数据相比，v3版进步明显。在Titan X环境下，YOLOv3的检测精度为57.9 AP50，用时51ms；而RetinaNet的精度只有57.5 AP50，但却需要198ms，相当于YOLOv3的3.8倍。

代码地址：https://pjreddie.com/darknet/yolo/

#### 1.引言

有时候啊，其实有些人一年的时间就那么蹉跎了，你懂的。所以去年我也没做什么研究，就是在玩Twitter上花了点时间，在GAN上花了点时间，然后回头一看，好像自己还空了一点精力出来[10][1]，于是就想，要不更新下YOLO算了。说实在的，这只是一系列很小的、让YOLO变得更完善的更新，也不是多么有趣的事。另外，我还帮别人做了一点研究。

事实上这也是我今天要讲的内容。我们有一篇论文快截稿了，但还缺一篇关于YOLO更新内容的文章作为引用来源，我们没写，所以以下就是我们的技术报告！

技术报告的伟大之处在于它不需要“引言”，你们都知道我们写这个的目的对吧，所以这段“引言”可以作为你阅读的一个指引。首先我们会告诉你YOLOv3的更新情况，其次我们会展示更新的方法以及一些失败的尝试，最后就是对这轮更新的意义的总结。

#### 2.更新

谈到YOLOv3的更新情况，其实大多数时候我们就是直接把别人的好点子拿来用了。我们还训练了一个全新的、比其他网络更好的分类网络。为了方便你理解，让我们从头开始慢慢介绍。

<div align=center>
<img src="zh-cn/img/yolov3/p3.png" />
</div>


**2.1边界框预测**

bounding box预测手段是YOLO v3论文中提到的又一个亮点。先回忆一下YOLO v2的bounding box预测：想借鉴Faster R-CNN RPN中的anchor机制，但不屑于手动设定anchor prior(模板框)，于是用维度聚类的方法来确定anchor box prior(模板框)，最后发现聚类之后确定的prior在k=5也能够有不错的表现，于是就选用k=5。后来呢，YOLO v2又嫌弃anchor机制线性回归的不稳定性(因为回归的offset可以使box偏移到图片的任何地方)，所以YOLO v2最后选用了自己的方法：直接预测相对位置。预测出bounding box中心点相对于网格单元左上角的相对坐标。


<div align=center>
<img src="zh-cn/img/yolov3/p4.jpg" />
</div>

*公式1**

<div align=center>
<img src="zh-cn/img/yolov3/p5.png" />
</div>

<div align=center>
<img src="zh-cn/img/yolov3/p6.png" />
</div>


```shell
10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
```
<div align=center>
<img src="zh-cn/img/yolov3/p7.png" />
</div>

如果模板框不是最佳的即使它超过我们设定的阈值，我们还是不会对它进行predict。
不同于faster R-CNN的是，yolo_v3只会对1个prior进行操作，也就是那个最佳prior。而logistic回归就是用来从9个anchor priors中找到objectness score(目标存在可能性得分)最高的那一个。logistic回归就是用曲线对prior相对于 objectness score映射关系的线性建模。


注：

+ 第一点， 9个anchor会被三个输出张量平分的。根据大中小三种size各自取自己的anchor。

+ 第二点，每个输出y在每个自己的网格都会输出3个预测框，这3个框是9除以3得到的，这是作者设置
的，我们可以从输出张量的维度来看，13x13x255。255是怎么来的呢，3*(5+80)。80表示80个种类，5表
示位置信息和置信度，3表示要输出3个prediction。在代码上来看，3*(5+80)中的3是直接由
num_anchors//3得到的。

+ 第三点，作者使用了logistic回归来对每个anchor包围的内容进行了一个目标性评分(objectness score)。
根据目标性评分来选择anchor prior进行predict，而不是所有anchor prior都会有输出。


**2.2分类预测**

每个边界框都会使用多标记分类来预测框中可能包含的类。我们不用softmax，而是用单独的逻辑分类器，因为我们发现前者对于提升网络性能没什么用。在训练过程中，我们用二元交叉熵损失来预测类别。

这个选择有助于我们把YOLO用于更复杂的领域，如开放的图像数据集[5]。这个数据集中包含了大量重叠的标签（如女性和人）。如果我们用的是softmax，它会强加一个假设，使得每个框只包含一个类别。但通常情况下这样做是不妥的，相比之下，多标记的分类方法能更好地模拟数据。


**2.3跨尺寸预测**

YOLOv3提供了3种尺寸不一的边界框。我们的系统用相似的概念提取这些尺寸的特征，以形成金字塔形网络[6]。我们在基本特征提取器中增加了几个卷积层，并用最后的卷积层预测一个三维张量编码：边界框、框中目标和分类预测。在COCO数据集实验中，我们的神经网络分别为每种尺寸各预测了3个边界框，所以得到的张量是N ×N ×[3∗(4+ 1+ 80)]，其中包含4个边界框offset、1个目标预测以及80种分类预测。

接着，我们从前两个图层中得到特征图，并对它进行2次上采样。再从网络更早的图层中获得特征图，用element-wise把高低两种分辨率的特征图连接到一起。这样做能使我们找到早期特征映射中的上采样特征和细粒度特征，并获得更有意义的语义信息。之后，我们添加几个卷积层来处理这个特征映射组合，并最终预测出一个相似的、大小是原先两倍的张量。

我们用同样的网络设计来预测边界框的最终尺寸，这个过程其实也有助于分类预测，因为我们可以从早期图像中筛选出更精细的特征。

和上一版一样，YOLOv3使用的聚类方法还是K-Means，它能用来确定边界框的先验。在实验中，我们选择了9个聚类和3个尺寸，然后在不同尺寸的边界框上均匀分割维度聚类。在COCO数据集上，这9个聚类分别是：(10×13)、(16×30)、(33×23)、(30×61)、(62×45)、(59×119)、(116 × 90)、(156 × 198)、(373 × 326)。

**2.4特征提取器**


这次我们用了一个新的网络来提取特征，它融合了YOLOv2、Darknet-19以及其他新型残差网络，由连续的3×3和1×1卷积层组合而成，当然，其中也添加了一些shortcut connection，整体体量也更大。因为一共有53个卷积层，所以我们称它为……别急哦…… **Darknet-53**！

<div align=center>
<img src="zh-cn/img/yolov3/p8.jpg" />
</div>

YOLO系列里面，作者只在v1的论文里给出了结构图，而v2和v3的论文里都没有结构图，这使得读者对后两代YOLO结构的理解变得比较难。对于YOLO学习者来说，脑子里没有一个清晰的结构图，就别说自己懂UOLO了。上图是[某大神](https://blog.csdn.net/leviopku/article/details/82660381)根据官方代码和官方论文以及模型结构可视化工具等经过好几个小时画出来的，修订过几个版本。所以，上图的准确性是可以保证的。

这里推荐的模型结构可视化工具是：[Netron](https://blog.csdn.net/leviopku/article/details/81980249)
Netron方便好用，可以直观看到YOLO v3的实际计算结构，精细到卷积层。要进一步在人性化的角度分析v3的结构图，还需要结合论文和代码。

上图表示了YOLO v3整个YOLO_body的结构，没有包括把输出解析整理成咱要的[box, class, score]。对于把输出张量包装成[box, class, score]那种形式，还需要写一些脚本，但这已经在神经网络结构之外了(我后面会详细解释这波操作)。

为了让YOLO v3结构图更好理解，对上图做一些补充解释：

+ **DBL**: 如上图左下角所示，也就是代码中的Darknetconv2d_BN_Leaky，是YOLO v3的基本组件。就是卷积+BN+Leaky relu。对于v3来说，BN和leaky relu已经是和卷积层不可分离的部分了(最后一层卷积除外)，共同构成了最小组件。

+ **resn**：n代表数字，有res1，res2, … ,res8等等，表示这个res_block里含有多少个res_unit。这是YOLO v3的大组件，YOLO v3开始借鉴了ResNet的残差结构，使用这种结构可以让网络结构更深(从v2的darknet-19上升到v3的darknet-53，前者没有残差结构)。对于res_block的解释，可以在上图的右下角直观看到，其基本组件也是DBL。
concat：张量拼接。将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。

我们可以借鉴Netron来分析网络层，整个yolo_v3_body包含252层，组成如下：

<div align=center>
<img src="zh-cn/img/yolov3/p9.jpg" />
</div>

根据上图可以得出，对于代码层面的layers数量一共有252层，包括add层23层(主要用于res_block的构成，每个res_unit需要一个add层，一共有1+2+8+8+4=23层)。除此之外，BN层和LeakyReLU层数量完全一样(72层)，在网络结构中的表现为：每一层BN后面都会接一层LeakyReLU。卷积层一共有75层，其中有72层后面都会接BN+LeakyReLU的组合构成基本组件DBL。看结构图，可以发现上采样和concat都有2次，和表格分析中对应上。每个res_block都会用上一个零填充，一共有5个res_block。

**1.backbone**

整个v3结构里面，是没有池化层和全连接层的。前向传播过程中，张量的尺寸变换是通过改变卷积核的步长来实现的，比如stride=(2, 2)，这就等于将图像边长缩小了一半(即面积缩小到原来的1/4)。在yolo_v2中，要经历5次缩小，会将特征图缩小到原输入尺寸的1/25 1/2^51/2 
5
 ，即1/32。输入为416x416，则输出为13x13(416/32=13)。
yolo_v3也和v2一样，backbone都会将输出特征图缩小到输入的1/32。所以，通常都要求输入图片是32的倍数。可以对比v2和v3的backbone看看：（DarkNet-19 与 DarkNet-53）

<div align=center>
<img src="zh-cn/img/yolov3/p10.jpg" />
</div>

<div align=center>
<img src="zh-cn/img/yolov3/p11.png" />
</div>


YOLO v2中对于前向过程中张量尺寸变换，都是通过最大池化来进行，一共有5次。而v3是通过卷积核增大步长来进行，也是5次。(darknet-53最后面有一个全局平均池化，在YOLO v3里面没有这一层，所以张量维度变化只考虑前面那5次)。
这也是416x416输入得到13x13输出的原因。从上图可以看出，darknet-19是不存在残差结构(resblock，从resnet上借鉴过来)的，和VGG是同类型的backbone(属于上一代CNN结构)，而darknet-53是可以和resnet-152正面刚的backbone，看下表：

<div align=center>
<img src="zh-cn/img/yolov3/p12.png" />
</div>

从上表也可以看出，darknet-19在速度上仍然占据很大的优势。其实在其他细节也可以看出(比如bounding box prior采用k=9)，YOLO v3并没有那么追求速度，而是在保证实时性(fps>60)的基础上追求performance。不过前面也说了，你要想更快，还有一个tiny-darknet作为backbone可以替代darknet-53，在官方代码里用一行代码就可以实现切换backbone。搭用tiny-darknet的yolo，也就是tiny-yolo在轻量和高速两个特点上，显然是state of the art级别，tiny-darknet是和squeezeNet正面刚的网络，详情可以看下表： 

<div align=center>
<img src="zh-cn/img/yolov3/p13.png" />
</div>

所以，有了YOLO v3，就真的用不着YOLO v2了，更用不着YOLO v1了。这也是[YOLO官方网站](https://pjreddie.com/darknet/)，在v3出来以后，就没提供v1和v2代码下载链接的原因了。


**2.Output**

对于上图而言，更值得关注的是输出张量：

<div align=center>
<img src="zh-cn/img/yolov3/p14.png" />
</div>


YOLO v3输出了3个不同尺度的feature map，如上图所示的y1, y2, y3。这也是v3论文中提到的为数不多的改进点：predictions across scales
这个借鉴了FPN(feature pyramid networks)，采用多尺度来对不同size的目标进行检测，越精细的grid cell就可以检测出越精细的物体。
y1,y2和y3的深度都是255，边长的规律是`13:26:52`
对于COCO类别而言，有80个种类，所以每个box应该对每个种类都输出一个概率。

YOLO v3设定的是每个网格单元预测3个box，所以每个box需要有`(x, y, w, h, confidence)`五个基本参数，然后还要有80个类别的概率。所以3*(5 + 80) = 255。这个255就是这么来的。（还记得yolo v1的输出张量吗？ 
7x7x30，只能识别20类物体，而且每个cell只能预测2个box，和v3比起来就像老人机和iphoneX一样）
v3用上采样的方法来实现这种多尺度的feature map，可以结合上面的一些图，图中concat连接的两个张量是具有一样尺度的(两处拼接分别是26x26尺度拼接和52x52尺度拼接，通过(2, 2)上采样来保证concat拼接的张量尺度相同)。作者并没有像SSD那样直接采用backbone中间层的处理结果作为feature map的输出，而是和后面网络层的上采样结果进行一个拼接之后的处理结果作为feature map。为什么这么做呢？ 我感觉是有点玄学在里面，一方面避免和其他算法做法重合，另一方面这也许是试验之后并且结果证明更好的选择，再者有可能就是因为这么做比较节省模型size的。这点的数学原理不用去管，知道作者是这么做的就对了。


**2.5训练**

我们只是输入完整的图像，并没有做其他处理。实验过程中涉及的多尺寸训练、大量数据增强和batch normalization等操作均符合标准。模型训练和测试的框架是Darknet神经网络。

** Loss Function**

对掌握YOLO来讲，loss function不可谓不重要。在v3的论文里没有明确提所用的损失函数，确切地说，YOLO系列论文里面只有YOLO v1明确提了损失函数的公式。对于YOLO这样一种讨喜的目标检测算法，就连损失函数都非常讨喜。在v1中使用了一种叫sum-square error的损失计算方法，就是简单的差方相加而已。想详细了解的可以看YOLO v1部分。我们知道，在目标检测任务里，有几个关键信息是需要确定的:

<div align="center">
(x,y),(w,h),class,confidence
</div> 

根据关键信息的特点可以分为上述四类，损失函数应该由各自特点确定。最后加到一起就可以组成最终的loss_function了，也就是一个loss_function搞定端到端的训练。可以从代码分析出v3的损失函数，同样也是对以上四类，不过相比于v1中简单的总方误差，还是有一些调整的：

```python
xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

xy_loss = K.sum(xy_loss) / mf
wh_loss = K.sum(wh_loss) / mf
confidence_loss = K.sum(confidence_loss) / mf
class_loss = K.sum(class_loss) / mf
loss += xy_loss + wh_loss + confidence_loss + class_loss

```

以上是一段keras框架描述的YOLO v3 的loss_function代码。忽略恒定系数不看，可以从上述代码看出：除了w, h的损失函数依然采用总方误差之外，其他部分的损失函数用的是二值交叉熵。最后加到一起。那么这个binary_crossentropy又是个什么玩意儿呢？就是一个最简单的交叉熵而已，一般用于二分类，这里的两种二分类类别可以理解为"对和不对"这两种。关于binary_crossentropy的公式详情可参考博文:[常见的损失函数](https://blog.csdn.net/legalhighhigh/article/details/81409551)。


#### 3.我们做了什么

YOLOv3的表现非常好！请参见表3，就COCO奇怪的平均mAP成绩而言，它与SSD变体相当，但速度提高了3倍。尽管如此，它仍然比像RetinaNet这样的模型要差一点。

<div align=center>
<img src="zh-cn/img/yolov3/p15.png" />
</div>

如果仔细看这个表，我们可以发现在IOU=.5（即表中的AP50）时，YOLO v3非常强大。它几乎与RetinaNet相当，并且远高于SSD变体。这就证明了它其实是一款非常灵活的检测器，擅长为检测对象生成合适的边界框。然而，随着IOU阈值增加，YOLO v3的性能开始同步下降，这时它预测的边界框就不能做到完美对齐了。

在过去，YOLO一直被用于小型对象检测。但现在我们可以预见其中的演变趋势，随着新的多尺寸预测功能上线，YOLO v3将具备更高的APS性能。但是它目前在中等尺寸或大尺寸物体上的表现还相对较差，仍需进一步的完善。

当我们基于AP50指标绘制精度和速度时，我们发现YOLO v3与其他检测系统相比具有显着优势。也就是说，它的速度正在越来越快。


#### 4.失败的尝试

我们在研究YOLO v3时尝试了很多东西，以下是我们还记得的一些失败案例。

**Anchor box坐标的偏移预测**。我们尝试了常规的Anchor box预测方法，比如利用线性激活将坐标x、y的偏移程度预测为边界框宽度或高度的倍数。但我们发现这种做法降低了模型的稳定性，且效果不佳。

**用线性方法预测x,y，而不是使用logistic方法**。我们尝试使用线性激活来直接预测x，y的offset，而不是sigmoid激活。这降低了mAP成绩。

**Focal loss**。我们尝试使用Focal loss，但它使我们的mAP降低了2点。 对于Focal loss函数试图解决的问题，YOLO v3从理论上来说已经很强大了，因为它具有单独的对象预测和条件类别预测。因此，对于大多数例子来说，类别预测没有损失？或者其他的东西？我们并不完全确定。(关于Focal Loss详见本章附录部分)

**双IOU阈值和真值分配**。在训练期间，Faster RCNN用了两个IOU阈值，如果预测的边框与.7的ground truth重合，那它是个正面的结果；如果在[.3—.7]之间，则忽略；如果和.3的ground truth重合，那它就是个负面的结果。我们尝试了这种思路，但效果并不好。

我们对现在的更新状况很满意，它看起来已经是最佳状态。有些技术可能会产生更好的结果，但我们还需要对它们做一些调整来稳定训练。

#### 5.更新的意义

YOLO v3是一个很好的检测器，它速度快，精度又高。虽然基于.3和.95的新指标，它在COCO上的成绩不如人意，但对于旧的检测指标.5 IOU，它还是非常不错的。

所以为什么我们要改变指标呢？最初的COCO论文里只有这样一句含糊其词的话：一旦评估完成，就会生成评估指标结果。Russakovsky等人曾经有一份报告，说人类很难区分.3与.5的IOU：“训练人们用肉眼区别IOU值为0.3的边界框和0.5的边界框是一件非常困难的事”。[16]如果人类都难以区分这种差异，那这个指标有多重要？

但也许更好的一个问题是：现在我们有了这些检测器，我们能用它们来干嘛？很多从事这方面研究的人都受雇于Google和Facebook，我想至少我们知道如果这项技术发展得完善，那他们绝对不会把它用来收集你的个人信息然后卖给……等等，你把事实说出来了！！哦哦。

那么其他巨资资助计算机视觉研究的人还有军方，他们从来没有做过任何可怕的事情，比如用新技术杀死很多人……呸呸呸

我有很多希望！我希望大多数人会把计算机视觉技术用于快乐的、幸福的事情上，比如计算国家公园里斑马的数量[11]，或者追踪小区附近到底有多少猫[17]。但是计算机视觉技术的应用已经步入歧途了，作为研究人员，我们有责任思考自己的工作可能带给社会的危害，并考虑怎么减轻这种危害。我们非常珍惜这个世界。

最后，不要在Twitter上@我，我已经弃坑了！！！！


**创新点**

+ 使用金字塔网络
+ 用逻辑回归替代softmax作为分类器
+ Darknet-53

**不足**

+ 速度确实快了，但mAP没有明显提升，特别是IOU > 0.5时。

**彩蛋**

偷偷告诉你哦，那个一作Joseph Redmon，他的简历长这样~

<div align=center>
<img src="zh-cn/img/yolov3/p1.jpg" />
</div>

但他本人实际上长这样哦~

<div align=center>
<img src="zh-cn/img/yolov3/p2.jpg" />
</div>

**代码实现汇总**

+ 汇总

	- https://github.com/amusi/YOLO-Reproduce-Summary

+ Tensorflow

	- https://github.com/YunYang1994/tensorflow-yolov3 支持训练(514 star)

	- https://github.com/mystic123/tensorflow-yolo-v3 不支持训练(446 star)

	- https://github.com/maiminh1996/YOLOv3-tensorflow 支持训练(246 star)

+ PyTorch

	- https://github.com/ayooshkathuria/pytorch-yolo-v3 不支持训练(1.5k star)

	- https://github.com/eriklindernoren/PyTorch-YOLOv3 支持训练(1.3k star)

	- https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch 不支持训练(1k star)

	- https://github.com/TencentYoutuResearch/ObjectDetectionOneStageDet/tree/master/yolo 支持训练(976 star)

	- https://github.com/BobLiu20/YOLOv3_PyTorch 支持训练(341 star)

	- https://github.com/DeNA/PyTorch_YOLOv3 支持训练(179 star)

	- https://github.com/ultralytics/yolov3 支持训练(971 star)

+ Keras

	- https://github.com/qqwweee/keras-yolo3 支持训练(2.8k star)

	- https://github.com/xiaochus/YOLOv3 不支持训练(418 star)

	- https://github.com/Adamdad/keras-YOLOv3-mobilenet 支持训练(191 star)

+ Caffe

	- https://github.com/eric612/MobileNet-YOLO 支持训练(301 star)

	- https://github.com/ChenYingpeng/caffe-yolov3 不支持训练(150 star)

	- https://github.com/eric612/Caffe-YOLOv3-Windows 支持训练(97 star)

+ MXNet

	- https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo/yolo 支持训练(2072 star)



#### Reference

[1] Analogy. Wikipedia, Mar 2018. 1

[2] C.-Y. Fu, W. Liu, A. Ranga, A. Tyagi, and A. C. Berg. Dssd: Deconvolutional single shot detector

[3] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition.

[4] J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara, A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, et al. Speed/accuracy trade-offs for modern convolutional object

[5] I. Krasin, T. Duerig, N. Alldrin, V. Ferrari, S. Abu-El-Haija, A. Kuznetsova, H. Rom, J. Uijlings, S. Popov, A. Veit, S. Belongie, V. Gomes, A. Gupta, C. Sun, G. Chechik, D. Cai, Z. Feng, D. Narayanan, and K. Murphy. Openimages: A public dataset for large-scale multi-label and multi-class image classification.

[6] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2117–2125, 2017. 2, 3

[7] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar. ´Focal loss for dense object detection.

[8] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[9] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.Y. Fu, and A. C. Berg. Ssd: Single shot multibox detector. In European conference on computer vision, pages 21–37. Springer, 2016.

[10] I. Newton. Philosophiae naturalis principia mathematica. William Dawson & Sons Ltd., London, 1687.

[11] J. Parham, J. Crall, C. Stewart, T. Berger-Wolf, and D. Rubenstein. Animal population censusing at scale with citizen science and photographic identification. 2017.

[12] J. Redmon. Darknet: Open source neural networks in c

[13] J. Redmon and A. Farhadi. Yolo9000: Better, faster, stronger. In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on, pages 6517–6525. IEEE, 2017. 1, 2, 3

[14] J. Redmon and A. Farhadi. Yolov3: An incremental improvement. arXiv, 2018.

[15] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks.

[16] O. Russakovsky, L.-J. Li, and L. Fei-Fei. Best of both worlds: human-machine collaboration for object annotation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2121–2131, 2015.

[17] M. Scott. Smart camera gimbal bot scanlime:027, Dec 2017.

[18] A. Shrivastava, R. Sukthankar, J. Malik, and A. Gupta. Beyond skip connections: Top-down modulation for object detection.

[19] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. 2017.


### 附录： Focal Loss

Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。

**损失函数形式**

Focal Loss是在交叉熵损失函数基础上进行的修改，首先回顾二分类交叉上损失：

<div align=center>
<img src="zh-cn/img/yolov3/p16.png" />
</div>

`y'`是经过激活函数的输出，所以在0-1之间。可见普通的交叉熵对于正样本而言，输出概率越大损失越小。对于负样本而言，输出概率越小则损失越小。此时的损失函数在大量简单样本的迭代过程中比较缓慢且可能无法优化至最优。那么Focal loss是怎么改进的呢？

<div align=center>
<img src="zh-cn/img/yolov3/p17.png" />
</div>


<div align=center>
<img src="zh-cn/img/yolov3/p18.png" />
</div>

首先在原有的基础上加了一个因子，其中gamma>0使得减少易分类样本的损失。使得更关注于困难的、错分的样本。

例如gamma为2，对于正类样本而言，预测结果为0.95肯定是简单样本，所以（1-0.95）的gamma次方就会很小，这时损失函数值就变得更小。而预测概率为0.3的样本其损失相对很大。对于负类样本而言同样，预测0.1的结果应当远比预测0.7的样本损失值要小得多。对于预测概率为0.5时，损失只减少了0.25倍，所以更加关注于这种难以区分的样本。这样减少了简单样本的影响，大量预测概率很小的样本叠加起来后的效应才可能比较有效。

此外，加入平衡因子alpha，用来平衡正负样本本身的比例不均：

<div align=center>
<img src="zh-cn/img/yolov3/p19.png" />
</div>

只添加alpha虽然可以平衡正负样本的重要性，但是无法解决简单与困难样本的问题。

lambda调节简单样本权重降低的速率，当lambda为0时即为交叉熵损失函数，当lambda增加时，调整因子的影响也在增加。实验发现lambda为2是最优。

作者认为one-stage和two-stage的表现差异主要原因是大量前景背景类别不平衡导致。作者设计了一个简单密集型网络RetinaNet来训练在保证速度的同时达到了精度最优。在双阶段算法中，在候选框阶段，通过得分和nms筛选过滤掉了大量的负样本，然后在分类回归阶段又固定了正负样本比例，或者通过OHEM在线困难挖掘使得前景和背景相对平衡。而one-stage阶段需要产生约100k的候选位置，虽然有类似的采样，但是训练仍然被大量负样本所主导。

**详细的我们将在RetinaNet中讲解**


**Reference**

[Focal Loss理解](https://www.cnblogs.com/king-lps/p/9497836.html)

[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)