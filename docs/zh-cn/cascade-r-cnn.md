## Cascade R-CNN

!> 论文地址：https://arxiv.org/abs/1712.00726

### 0.摘要

在目标检测中，需要一个交并比(IOU)阈值来定义物体正负标签。使用低IOU阈值(例如0.5)训练的目标检测器通常会产生噪声检测。然而，随着IOU阈值的增加，检测性能趋于下降。影响这一结果的主要因素有两个：1)训练过程中由于正样本呈指数级消失而导致的过度拟合；2)检测器为最优的IOU与输入假设的IOU之间的推断时间不匹配。针对这些问题，提出了一种多级目标检测体系结构-级联R-CNN（Cascade R-CNN）.**它由一系列随着IOU阈值的提高而训练的目标检测器组成，以便对接近的假阳性有更多的选择性。检测器是分阶段训练的，利用观察到的检测器输出是训练下一个高质量检测器的良好分布。逐步改进的假设的重采样保证了所有检测器都有一组等效尺寸的正的例子，从而减少了过拟合问题。同样的级联程序应用于推断，使假设与每个阶段的检测器质量之间能够更紧密地匹配。** Cascade R-CNN的一个简单实现显示，在具有挑战性的COCO数据集上，它超过了所有的单模型对象检测器。实验还表明，Cascade R-CNN在检测器体系结构中具有广泛的适用性，独立于基线检测器强度获得了一致的增益。Tensorflow代码将在 https://github.com/DetectionTeamUCAS/Cascade-RCNN_Tensorflow 上提供。

### 1.Introduction

目标检测是一个复杂的问题，需要完成两个主要任务.首先，检测器必须解决识别问题，区分前景对象和背景，并为它们分配合适的对象类别标签。第二，检测器必须解决定位问题，为不同的对象分配精确的边界框(b-box)。这两种方法都是特别困难的，因为检测器面临许多“相似的”错误，对应于“相似但不正确”的边界框。检测器必须在消除这些相似假阳性的同时找到真阳性。

最近提出的许多目标检测器是基于two-stage R-CNN框架[12，11，27，21]，其中检测是一个结合分类和边界框回归的多任务学习问题。与目标识别不同的是，需要一个交并比(IOU)阈值来定义正/负。然而，通常使用的阈值u(通常u=0.5)对正项的要求相当宽松。产生的检测器经常产生噪声边界框（FP），如图1(a)所示。假设大多数人会经常考虑相似假阳性，通过IOU≥0.5测试。虽然在u=0.5准则下汇集的例子丰富多样，但它们使训练能够有效地拒绝相似假阳性的检测器变得困难。

<div align=center>
<img src="zh-cn/img/casecadercnn/p1.png" /> 
</div>

*Fig1: 提高IOU阈值u的目标检测器的检测输出、定位和检测性能*

上图 (c)为location performance，横轴是测试的 Input IOU，纵轴是输出的 IOU，从(c)可以看出使用0.5训练出来的模型在Input IOU在`[0.55,0.6]`表现最好，0.6得到的模型在`[0.65,075]`表现最好，0.7得到的模型在[0.8：]的效果最好，而且这三个模型输出的IOU都高于baseline。

上图 d 为detection performance，横轴是测试时设置的IOU阈值，高于阈值的视为正例，之后对置信度进行排序，得到AP值。可以看出随着阈值的增大，AP值逐渐降低，同时0.5阈值训练的到的网络在低IOU阈值得到的结果好于0.6训练出来的结果，0.6阈值训练出来的在高阈值得到的结果好，由此可以得知，一个detector只对单个IOU level 表现较好。

单纯的提高阈值结果并不好，例如上图的 `u=0.7` 的结果，这是因为正例个数的指数减少，很容易造成过拟合，另一个原因是因为mismatch，如上图，高质量的detector只对 high quality hypotheses(测试的IoU比较大的)是最优的，对于其他的level不是最优的。

因为一个detector的输出的IOU总是优于他的输入的IOU，上图(c)三个都优于baseline，所以我们自然而然的使用级联的方式，**使用上一层的输出当做下一层的输入**。这样的过程类似于重新采样，但不同于 mine hard negatives，我们通过调整 bounding box 来找到更适合的close false positive 来进行下一个stage的训练。通过IOU的逐步提高可以克服过拟合的问题，同时在Inference使用相同的结构。

在本文中，我们提出了一种新的目标测器结构，Cascade R-CNN，以解决这些问题. 这是R-CNN的多阶段扩展，检测器的级联阶段越深，对相似假阳性就有更多的选择性。**R-CNN的级联是按顺序训练的，使用一个阶段的输出来训练下一个阶段**。这是因为观察到回归器的输出IOU几乎总是优于输入IOU。这个观察可以在图1(c)中进行，其中所有的线都在灰色线之上。结果表明，用一定的IOU阈值训练的检测器的输出是训练下一次较高IOU阈值检测器的良好分布。这类似于[31，8], 主要的区别在于，Cascade R-CNN的重采样程序并不是为了挖掘难负样例。相反，**通过调整边界框，每一阶段的目标是找到一组好的相似假阳性来训练下一阶段**。当以这种方式操作时，适应于越来越高的IoU的一系列检测器可以克服过度拟合的问题，从而得到有效的训练。在推断时，采用相同的级联过程。逐步改进的假设在每个阶段都能更好地与不断提高的检测测器IOU相匹配。如图1(c)和(d)所示，这使检测精度更高。

### 2.Related Work

One-stage detector中，SSD 网络和 RPN 相似，但是使用了多层 feature map 特征。RetinaNet 网络解决了dense object detection的类极度不平衡问题，优于最好的 two-stage detector。

还有许多multi-satge object detection。有使用Fast RCNN循环产生bounding-box的， 有将cascade 嵌入到检测网络中的，有迭代检测和分割任务的。

### 3.Object Detection

在本文中，我们扩展了Faster R-CNN[27，21]的两阶段体系结构，如图3(a)所示。第一阶段是RPN(H0)，应用于整个图像，产生初步的候选区域, 在第二阶段，Detection Head(H1)处理，每个ROI都有一个最终的分类分数(“C”)和一个边框回归偏移(“B”)。我们专注于多阶段检测子网络的建模，并采用但不限于RPN[27]来进行候选区域检测。

<div align=center>
<img src="zh-cn/img/casecadercnn/p2.png" /> 
</div>

*Fig3: 不同框架的结构。"I"是输入图像，“conv”是骨干卷积，“pool”是区域特征提取，“H”是网络头部，“B”是边界框，“C"是分类，”$B_0$”是所有结构的候选区域。*


!> **Fig3 重要注解:**

Fig3是关于几种网络结构的示意图。（a）是Faster RCNN，因为two-stage类型的object detection算法基本上都基于Faster RCNN，所以这里也以该算法为基础算法。（b）是迭代式的bbox回归，从图也非常容易看出思想，就是前一个检测模型回归得到的bbox坐标初始化下一个检测模型的bbox，然后继续回归，这样迭代三次后得到结果。Iterative BBox为了定位准确，采用了级联结构来对Box进行回归，使用的是完全相同的级联结构。但是这样以来，第一个问题：单一阈值0.5是无法对所有proposal取得良好效果的，如第1部分的图所示，proposal经过0.5阈值的detector后IoU都在0.75以上，再使用这一阈值并不明智；第二个，detector会改变样本的分布，这时候再使用同一个结构效果也不好，看下图:

<div align=center>
<img src="zh-cn/img/casecadercnn/p9.png" /> 
</div>

第一行横纵轴分别是回归目标中的box的x方向和y方向偏移量；第二行横纵轴分别是回归目标中的box的宽、高偏差量，由于比较基础这里不贴公式了。我们可以看到，从1st stage到2nd stage，proposal的分布其实已经发生很大变化了，因为很多噪声经过box reg实际上也提高了IoU，2nd和3rd中的那些红色点已经属于outliers，如果不提高阈值来去掉它们，就会引入大量噪声干扰，对结果很不利。从这里也可以看出，阈值的重新选取本质上是一个resample的过程，它保证了样本的质量。

（c）是Integral Loss，表示对输出bbox的标签界定采取不同的IOU阈值，因为当IOU较高时，虽然预测得到bbox很准确，但是也会丢失一些bbox。Iterative Loss实际上没有级联结构，从c图可以看出来，它只是使用了不同的阈值来进行分类，然后融合他们的结果进行分类推理，并没有同时进行Box reg。作者认为，从图4中的第一个图可以看出来，当IoU提高的时候，proposal的比重下降非常迅速，这种方法没有从根本上克服overfit问题；另外，这种结构使用了多个高阈值的分类器，训练阈值却只能有一个，必然会导致mismatch问题而影响性能。

（d）就是本文提出的cascade-R-CNN。cascade-R-CNN看起来和（b）这种迭代式的bbox回归以及（c）这种Integral Loss很像，和（b）最大的不同点在于cascade-R-CNN中的检测模型是基于前面一个阶段的输出进行训练，而不是像（b）一样3个检测模型都是基于最初始的数据进行训练，而且（b）是在验证阶段采用的方式，而cascade-R-CNN是在训练和验证阶段采用的方式。和（c）的差别也比较明显，cascade R-CNN中每个stage的输入bbox是前一个stage的bbox输出，而（c）其实没有这种refine的思想，仅仅是检测模型基于不同的IOU阈值训练得到而已。


#### 3.1 Bounding Box Regression

Bounding Box $b=(b_x,b_y,b_w,b_h)$, Bounding Box回归的目的是使用回归器$f(x,b)$将候选的边界框b回归到GT,表示为g。 这是从训练样本$\{g_i,b_i\}$中学习到的，以便将边界框风险降到最低
$$R_{loc}[f] = \sum_{i=1}^{N}L_{loc}(f(x_i,b_i),g_i)$$
其中，$L_{loc}$是R-CNN种的$L_2$损失函数，但是在Fast R-CNN中被更新为smoothed $L_1$损失函数。$L_{loc}$中的距离向量定义为$(\delta_x,\delta_y,\delta_w,\delta_h)$

<div align=center>
<img src="zh-cn/img/casecadercnn/p3.png" /> 
</div>

由于边界框回归通常对b进行较小的调整，所以上式的数值可能很小。因此，边界框回归的风险通常比分类风险小得多。为了提高多任务学习的有效性，∆通常用均值和方差进行归一化，即用$\delta_{x^{‘}}=(\delta_x−\mu_x)/\\sigma_x$代替$\delta_x$，这在文献[27，1，4，21，14]中得到了广泛的应用。

[9，10，16]认为，f的一个回归步骤不足以精确定位。相反，f被迭代地应用，作为后处理步骤来改善边界框b

<div align=center>
<img src="zh-cn/img/casecadercnn/p4.png" /> 
</div>

但这仍然有很多缺陷.

#### 3.2 Classification

一个 `M + 1` 维的预测类别后验概率的分支，使用经典的 cross-entropy loss进行优化


#### 3.3 Detection Quality 

由于边界框通常包括一个物体和一定数量的背景，因此很难确定检测是正的还是负的。这通常由IOU度量来解决。如果IOU高于阈值u，则该部分被视为类的一个示例。因此，假设x的类标号是u的函数，

<div align=center>
<img src="zh-cn/img/casecadercnn/p5.png" /> 
</div>

其中$g_y$是GT的类别标签，IoU的阈值u定义为目标检测器的质量。
训练时u值较大，阳性值包含的背景较少，很难收集正的样本；u值较小，会获得更丰富和多样的正样本，但是检测器对假阳性进行误判。一般来说，要求单个检测器在所有IOU级别上都表现良好是非常困难的。

Naive(朴素)的解决方案是开发一组分类器，其结构如图3(c)所示，并以针对不同质量级别的损失为目标进行优化，

<div align=center>
<img src="zh-cn/img/casecadercnn/p6.png" /> 
</div>

其中U是一组IoU阈值。这与[34]的积分损失密切相关，其中$U={0.5,0.55,...,0.75}$，旨在适应COCO挑战的评价标准。根据定义，分类器需要在推理时集合起来。这一解决方案未能解决上式的不同损失在不同数量的正数上工作的问题。随着u的增加，正样本集合迅速减少。这是特别有问题的，因为高质量的分类器容易过度拟合。此外，这些高质量的分类器被要求在推断时处理压倒性低质量的候选区域，但它们并不是最优的。由于所有这些，上式集成在大多数质量级别上都无法达到更高的精度，而且该体系结构与图3(a)相比几乎没有什么好处。


### 4.Cascade R-CNN

在本节中，我们将介绍图3(d)中提出的Cascade R-CNN对象检测体系结构。

#### 4.1 Cascaded Bounding Box Regression

如图1(c)所示，很难要求单个回归器在所有质量级别上完美地一致执行。在cascade pose regression[6]和人脸对齐[2，32]的启发下，将困难的回归任务分解为一系列简单的步骤。在Cascade R-CNN中，它被描述为一个级联回归问题，其结构如图3(d)所示。这依赖于一系列specialized的回归器

<div align=center>
<img src="zh-cn/img/casecadercnn/p7.png" /> 
</div>

其中T是级联的总数。请注意，对于到达相应阶段的样本分布$\{b^t\}$，对级联中的每个回归器$f_t$进行了优化，而不是对$\{b^1\}$的初始分布进行优化。

它在几个方面不同于图3(b)的iterative BBOX体系结构。首先, iterative  BBOX是用于改进边界框的后处理过程，而级联回归是一种重采样过程，它改变了不同阶段要处理的假设的分布。第二，由于它既用于训练又用于推断，因此训练和推断分布之间没有差异。第三，针对不同阶段的重采样分布，对多个特殊回归器$\{f_T,f_{T-1},...,f_1\}$ 进行了优化。这与(b)的单个$f$相反，后者仅对初始分布是最优的。与iterative BBOX相比，这些差异使定位更加精确，没有进一步的人类工程。

正如在第3.1节中所讨论的，$(\delta_x,\delta_y,\delta_w,\delta_h)$在(2)中需要通过其均值和方差进行规范化，才能有效地进行多任务学习。在每个回归阶段之后，这些统计数据将按顺序演变，在训练时，相应的统计信息将用于在每个阶段规范∆。

#### 4.2Cascaded Detection

如图4左图所示，最初假设(如RPN提案)的分布严重倾向于低质量。这不可避免地导致了对高质量分类器的无效学习。Cascade R-CNN依靠级联回归作为重采样机制来解决这个问题.这是因为在图1(c)中，所有曲线都在对角灰色线之上，即为某个确切u值训练的边界框回归器倾向于产生较高的IOU边界框。因此，从一组示例$(x_i,b_i)$开始，级联回归成功重采样出一个有高IoU的分布$(x_i^{'},b_i^{'})$这样，即使提高了探测器质量(IOU阈值)，也有可能使连续各阶段的一组正例保持在大致恒定的大小。这在图4中得到了说明，在每个重采样步骤之后，分布更倾向于高质量的示例。随后产生了两个后果。首先，没有overfitting的情况，因为在所有level上都有大量的例子。其次，针对较高的IOU阈值，对较深阶段的检测器进行了优化。请注意，一些异常值是通过增加IOU阈值来依次删除的，如图Fig2所示，从而实现了经过更好训练的专用检测器序列。

<div align=center>
<img src="zh-cn/img/casecadercnn/p8.png" /> 
</div>

在每个阶段t,R-CNN包括一个分类器$h_t$和一个针对IoU阈值$u_t$优化的回归器$f_t$,其中$u_t>u_{t-1}$,这是通过最小化损失来保证的。


<div align=center>
<img src="zh-cn/img/casecadercnn/p10.png" /> 
</div>

其中$b^t=f_{t-1}(x^{t-1},b^{t-1})$, $g$是$x^t$对应的真值, $[.]$表示示性函数, $y^t$是$x^t$的label。

### 5.实验结果

文中对比了Cascade R-CNN based on FPN+和ResNet-101结构和一些优秀算法的对比，从AP指标上来看Cascade R-CNN对检测精度的提升很大；

<div align=center>
<img src="zh-cn/img/casecadercnn/p11.png" /> 
</div>

同时还比对了Cascade R-CNN的参数量和速度，从表中可以发现参数量增加得比较多，但是inference和training的速度影响不大：


<div align=center>
<img src="zh-cn/img/casecadercnn/p12.png" /> 
</div>


### 6.总结

+ 本文提出了一个级联目标检测器Cascade R-CNN，给高质量目标检测器的设计提出一个好的方向；
+ Cascade R-CNN在提升IOU的同时避免了training阶段的过拟合以及inference阶段的检测质量mismatch；

文中对比实验很充分：
+ 在proposal中添加GT bbox验证检测质量mismatch问题；
+ 和iterative bbox和integral loss两种方法进行对比；
+ 通过调整stages的数目来分析合适级联数，3stages级联综合表现最好；
+ 在现有two stage算法框架上加入cascade思想和原始实现进行对比；
+ Cascade R-CNN带来的检测结果还是很优秀的；
检测问题不同于分类问题，分类问题中样本的label是离散的可以很好区分正负样本，但是检测问题中bbox的信息不是离散的只能通过IOU阈值来判定正负样本，并且合适IOU阈值没法在训练中通过学习来调整优化，所以Cascade R-CNN核心思想主要集中在IOU的优化上，算是在高质量目标检测器设计思路上一次优秀的尝试；