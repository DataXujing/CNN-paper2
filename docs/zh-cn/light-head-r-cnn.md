## Light-Head R-CNN

!> 论文地址：https://arxiv.org/abs/1711.07264

### 0.摘要

在本文中，我们首先研究了为什么典型的two-stage方法不如YOLO[26,27]和SSD[22]等one-stage检测器快。我们发现 Faster RCNN[28]和R-FCN[17]在RoI warping前后执行密集的计算。Faster R-CNN涉及两个全连接的层用于RoI识别，而R-FCN生成一个大的score map。因此，这些网络的速度是缓慢的，因为沉重的头部设计的架构(heavy-head design)。即使我们大大降低了基本模型，计算成本也不能相应地大幅度降低。

针对现有two-stage检测方法的不足，提出了一种新的two-stage检测方法——`Light-Head R-CNN`。在我们的设计中，我们使用一个小的特征图和一个轻量的R-CNN子网络(池化和单一全连接层)，使网络的头部尽可能轻。我们基于ResNet-101的Light-Head R-CNN在保持时间效率的同时，在COCO上的性能超过了最先进的目标检测器。更重要的是，只需用一个很小的网络(比如 Xception)，我们的Light-Head R-CNN在COCO上以每秒`102`帧的速度获得30.7 mmAP，在速度和精度上都明显优于YOLO[26,27]和SSD[22]等one-stage检测器。TF代码：https://github.com/zengarden/light_head_rcnn。


### 1.Introduction

近年来基于CNN的目标检测器可分为one-stage目标检测器[26,27,22,20,4]和two-stage目标检测器[5,28,19,7]。one-stage探测器通常以一个非常快的速度和相当好的精度为目标。two-stage检测器将任务划分为两个步骤:第一步(body)生成多个proposals，第二步(head)重点识别proposasls。通常，为了达到最好的精度，头部的设计是沉重的。two-stage探测器通常具有(相对)较慢的速度和非常高的精度的优点。

two-stage探测器在效率和准确度上能否超过one-stage探测器? 我们发现，典型的two-stage目标检测器，如Faster R-CNN[5]和R-FCN[17]具有相似的特征:一个沉重的头部连接到主干网络。例如，对于每个RoIs的识别和回归，Faster R-CNN使用两个大的全连接层或ResNet 5-th阶段的所有卷积层[28,29]。就每一区域的预测而言，这是费时的，如果使用大量的建议框，情况甚至会变得更糟。此外，ROI池化后的特征通道数量较大，使得第一次全连接占用较大内存，可能会影响计算速度。不同于Fast/Faster R-CNN多次应用每个区域的子网，基于区域的全卷积网络(R-FCN)[17]试图在所有RoIs中共享计算。然而，R-FCN需要使用`# classes * p*p `通道生成一个非常大的附加score map(p是下面的池化大小)，这也会占用内存和时间。Faster R-CNN或R-FCN的重头部设计使得如果我们使用一个小的骨干网络，two-stage的方法竞争更少。

在这篇论文中，我们提出了一个light head设计来建造一个高效而精确的two-stage目标检测器。具体来说，我们使用lage-kernal可分卷积来生成小通道数的“瘦”特征图(实验中使用的是`α *p *p`和`α≤10`)。该设计大大减少了后续RoI-wise子网的计算量，使检测系统内存友好。在池层上附加了一个廉价的单全连接层，池层很好地利用了特征表示进行分类和回归。

由于我们的轻量头部结构，我们的目标检测器能够在速度和精度之间做出最好的权衡，而不是使用大或小的骨干网络。如下图所示，我们的算法（点表示为Light-Head R-CNN），可以显著优于快速one-stage目标测器比如SSD[22]和YOLOv2[27]具有更快的计算速度。此外，我们的算法对大型骨干网络也很灵活。基于ResNet-101主干，我们可以超越现有的算法，包括two-stage目标检测器，如Mask R-CNN[7]和oen-stage目标检测器，如RetinaNet[20]。


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p1.png" /> 
</div>

*图1： Light-Head R-CNN与以往two-stage和pne-stage目标检测模型的比较。我们用不同的骨干来显示我们的结果(一个小的Xception, Resnet-50, Resnet-101)。感谢更好的设计原则，我们的light-head R-CNN显着优于所有竞争对手，并提供了一个新的上限。注意，这里报告的所有结果都是通过使用single-scale training。多尺度训练结果如表5所示。*

### 2.Related Works

得益于深度卷积网络的快速发展[16、32、34、9、15、38、12、39、12、14、11]，目标检测问题取得了很大进展。我们简要回顾了近年来在两个维度上的一些目标检测工作：

**精度展望**: R-CNN[6]是最早将深度神经网络特征应用到检测系统中的一种。手工设计的方法，如Seletive Search[37]，Edge Boxes[40]，MCG[1]，涉及生成R-CNN的候选区域。然后提出Fast R-CNN[5]加入训练对象分类和边界盒回归，通过多任务训练提高训练性能。继Fast R-CNN之后，Faster R-CNN[28]引入区域提案网络(RPN)，利用网络特性生成候选区域。得益于更丰富的候选区域，它略微提高了准确性。Faster R-CNN被视为R-CNN系列检测器的里程碑。下面的大部分工作通过将更多的计算带入网络来增强Faster R-CNN的速度。Dai等人提出了可变形卷积网络[3]，通过学习额外的偏移量来对几何变换建模，无需监督。Lin等人提出了特征金字塔网络(FPN)[19]，利用深卷积网络固有的多尺度金字塔结构构造特征金字塔。基于FPN，Mask R-CNN[7]通过在边界框识别的同时增加一个分支，进一步扩展了掩码预测器。RetinaNet[20]是另一种基于FPN的onet-stage目标检测器，它涉及到Focal Loss来解决类不平衡问题由极端的前背景比引起。

**速度展望**: 目标检测文献也一直在努力提高速度。回到原始的R-CNN，它通过整个网络单独转发每个候选区域。何凯明等人建议SPPnet[8]在候选框之间共享计算。无论是Fast/Faster R-CNN[5,28]，都通过统一检测管道来加速网络。R-FCN[17]在RoI子网络之间共享计算，当使用大量候选区域时，可以加快推理速度。另一个研究热点是无候选区域检测器。YOLO和YOLO v2[26,27]将对象检测简化为回归问题，直接预测边界框和相关的类概率，而无需生成候选区域。SSD[22]通过从不同层生成不同尺度的预测，进一步提高了性能。与基于box-center的检测器不同，DeNet[36]首先预测所有的box角，然后快速搜索非平凡边界框的角分布。

综上所述，从精度的角度来看，one-stage和two-stage目标检测都以相近的速度达到了最先进的精度。然而，从速度的角度来看，文献中缺乏具有竞争力的快速two-stage目标检测，与one-stage相比，具有接近精度。在本文中，我们试图设计一个更好更快的two-stage目标检测器，称为`Light-Head R-CNN`来填补这一空白。

### 3.Our Approach

在本节中，我们将首先介绍我们的light-head R-CNN，然后描述对象检测中的其他设计细节。

#### 3.1 Light-Head R-CNN

正如我们在第1节中所讨论的，传统的two-stage目标检测器通常包含一个沉重的头部(head)，这对计算速度有负面影响。`在我们的论文中，“Head”指的是连接到我们的主干网的结构`。更具体地说，将有两个组成部分:`R-CNN子网(ROI subnet)和ROI warping(变形)`。

##### 3.1.1 R-CNN subnet

Faster R-CNN采用功能强大的R-CNN，它利用两个大的全连通层或整个Resnet stage 5[28,29]作为第二阶段分类器，这有利于提高检测性能。因此，Faster R-CNN及其扩展在最具挑战性的基准测试(如COCO)中执行领先的精度。然而，计算可能是密集的，特别是当候选区域的数量很大时。为了加快RoI-wise子网的速度，R-FCN首先为每个区域生成一组score map，其通道数将为`#classes *p *p (p是后面的池化大小)`，然后沿着每个RoI进行池化，并对最终预测进行平均投票。使用无计算的R-CNN子网，R-FCN通过在RoI共享score map生成上进行更多的计算，得到了类似的结果。

如上所述，Faster R-CNN和R-FCN头部较重，但位置不同。从精度的角度来看，虽然Faster R-CNN在RoI分类上做得很好，但是为了减少第一个全连通层的计算量，通常会涉及到全局平均池化，这对空间定位是有害的。对于R-FCN，在位置敏感池化之后直接将预测结果进行池化处理，如果没有ROI-wise的计算层，其性能通常不如Faster R-CNN。从速度的角度来看，Faster R-CNN通过一个昂贵的R-CNN子网独立地传递每个RoI，这降低了网络速度，特别是当候选区域的数量很大时。R-FCN使用cost-free的R-CNN子网作为第二阶段探测器。但是由于R-FCN需要为RoI池化生成一个非常大的score map，整个网络仍然需要花费时间/内存。

考虑到这些问题，在我们新的Light-Head R-CNN中，`我们提议为我们的R-CNN子网使用一个简单、廉价的全连接层`，这在性能和计算速度之间做出了很好的权衡。图2(C)提供了Light-Head RCNN的概览。由于全连接层的计算和内存开销也依赖于ROI操作后的通道数，接下来我们将讨论如何设计ROI warping。

<div align=center>
<img src="zh-cn/img/lightheadrcnn/p2.png" /> 
</div>

*图2： Light-Head R-CNN示意图。我们的Light-Head R-CNN构建“薄”特征图之前的RoI warping，通过large可分离卷积。在我们的R- -CNN子网络，整个网络在保持准确性的同时非常高效。*

> 上图解释

上图介绍的是Faster R-CNN、R-FCN和Light-Head RCNN在结构上的对比。可以看出two-stage网络大都可以分为两部分：**ROI warping**和 **R-CNN subnet**。图中虚线框起来的部分是各网络的R-CNN subnet部分。这类算法的基本流程是这样的：通过base feature extractor（即特征提取网络）中某一层输出的feature map作为ROI warping的输入从而生成ROI，接着将ROI和feature map共同作为R-CNN subnet的输入完成 image classification和 object regression。

**Faster R-CNN**网络中，通过Resnet-101获得2048维特征图，接着是一个ROI pooling层，该层的输入包括2048维特征图和RPN中生成的ROI，输出是size统一的特征图（关于ROI pooling有不清楚的小伙伴，可以移步看一下SPP Net 和ROI pooling的源码解析。），再通过global average pool后接入两个全连接层，最后通过两个分支进行classification和location。在精度上，Faster R-CNN为了减少全连接层的计算量，使用了global average pool， 这会导致在一定程度上丢失位置信息；在速度上，Faster R-CNN的每一个ROI都要通过R-CNN subnet做计算，这必将引起计算量过大，导致检测速度慢。

**R-FCN**网络在实际使用中，conv5_x的2048维输出要接一个1024维`1*1`的filter用于降低“厚度”，接着用`p*p*(c+1)`维`1*1`的filter去卷积生成position-sensitive score map，也就是图中的彩色部分（从图中看`p=9`，但是在coco数据集上应用时`p=7`，所以`p*p*(c+1)=3969`，这也是本文的一个小漏洞吧），同时将conv4_x的feature map作为RPN的输入，生成ROI，将ROI 和position-sensitive score map共同作为输入，通过PSROI pooling层，得到`c+1`维`p*p`的特征图。最后经过一个global average pool层得到`c+1`维`1*1`的特征图，这`c+1`维就是对应该ROI的类别概率信息。相较于Faster R-CNN网络，R-FCN解决了每个ROI都要通过R-CNN subnet导致重复计算的问题。在精度上，R-FCN也使用了global average pool；在速度上，R-FCN的head虽然不用计算每一个ROI，但是其需要一个维度非常大的score map，这在一定程度上也会降低检测速度。


**Light-Head RCNN**主体和R-FCN差不多。针对R-FCN的score map维度过大的问题，作者用`10`代替了`class`，也就是说score map维度变成了`10*p*p=490`（作者称之为thinner feature map），因此降低了PSROI pooling和fc层的计算量；此外作者使用了large separable convolution代替`1*1` convolution，示意图如下图所示，可以看出作者借鉴了Inception V3 的思想，将`k*k`的卷积转化为`1*k`和`k*1`，同时采用图中左右两边的方法，最后通过padding融合feature map，得到size不变的特征图。将490维特征图和ROI作为PSROI 的输入则得到`10`维`p*p`的特征图，如果将490维特征图和ROI 作为ROI pooling的输入，则得到490维特征图，所以图中写了`10 or 490`。因为class更改为了10，所以没办法直接进行分类，所以接了个fc层做channel变换，再进行分类和回归


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p3.png" /> 
</div>

*图3：大可分卷积按顺序执行kX1和1Xk卷积。通过$C_{mid}$;$C_{out}$可以进一步控制计算复杂度。*


##### 3.1.2 Thin feature maps for ROI warping

在将候选区域提交到R-CNN子网，在这之前，需要进行RoI变形，使feature map的形状固定。

在Light-Head R-CNN中，我们提出生成小通道数的特征图 (thin feature map)，然后是传统的RoI变形。在实验中，我们发现在“瘦”特征图上进行RoI变形不仅可以提高精度，而且可以在训练和推断过程中节省内存和计算量。考虑到稀疏特征图上的PSRoI池化，我们可以引入更多的计算来增强R-CNN，减少通道。此外，如果我们在瘦特征图上应用RoI池化，我们可以减少R-CNN开销，同时放弃全局平均池以提高性能。此外，在不损失时间效率的情况下，可以使用大卷积来生成瘦特征图。

#### 3.2 Light-Head R-CNN for Object Detection

在上述讨论之后，我们给出了一般目标检测的实现细节。我们的方法如图2(C)所示。我们有两个设置:1)设置“L”来验证我们的算法在与大型骨干网络集成时的性能;2)设置“S”，验证算法在使用小型骨干网络时的有效性和效率。除非另有说明，否则设置“L”和设置“S”共享相同的其他设置。

**R-CNN子网**。在本文中，我们在R-CNN子网中只使用了一个具有2048(1024)个通道的FC层(没有Dropout)，然后是并行的FC来预测RoI分类和回归。每个边界框位置只应用4个通道，因为我们在不同的类之间共享回归。得益于强大的特征图对RoI的变形，一个简单的Light-Head R-CNN也可以在保持效率的同时取得显著的效果。

**RPN** (Region Proposal Network)是一种滑动窗口的类无关目标检测器，使用C4的特性。RPN预先定义了一组锚，这些锚由几个特定的尺度和纵横比控制。在我们的模型中，我们设置了三个纵横比`{1:2, 1:1, 2:1}`和五个尺度`{32²;64²;128²;256²}`;覆盖不同形状的物体。由于有许多候选区域彼此严重重叠，因此使用非最大抑制(NMS)来减少候选区域的数量。然后将其输入RoI预测子网络。我们将NMS的IoU阈值设为0.7。然后，我们根据锚点的IoU比率，使用GT真实边界框分配锚点训练标签。如果锚点的IoU超过0.7，并且带有任何ground-truth(GT)框，那么它将被设置为一个postive的标签。拥有最高IoU的地实值框的锚也将被分配一个postive的标签。同时，如果锚在所有ground-truth框中IoU小于0.3，则其标签将为负(negtive)。更多细节可以参考[28 Faster R-CNN]。


### 4. Experiments

在本节中，我们将对COCO [21,18] dataset评估我们的方法，它有80个对象类别。有`80k`训练集和`40k`验证集,这将进一步分为`35k` val-minusmini和`5k` mini-validation集。常见的设置后,我们将训练集和val-minusmini获得`115k`图像训练使用`5k `mini-validation图像进行验证。

#### 4.1 实现细节

我们的检测器是基于`8 Pascal TITAN XP gpu`端到端的训练，使用同步SGD，重量衰减为0.0001，动量为0.9。每个小批处理每个GPU有2张图像，每个图像有2000/1000 ROI用于培训/测试。通过在图像的右下角填充0，我们将微型批处理中的图像填充到相同的大小。前1.5M迭代的学习率设置为0.01(传递1张图像视为1次迭代)，后0.5M迭代为0.001。

所有实验均采用Resnet第五阶段的atrous[24,23,2]算法和在线难样本挖掘(OHEM)[31]技术。除非违反规定，我们的骨干网是基于预先训练的ImageNet[30]基本模型初始化的，池化大小设置为7。我们在基本模型中对阶段1和阶段2的参数进行了修正，为了加快实验速度，我们还对批量归一化进行了修正。除非另有说明，否则采用水平图像翻转增强。

在接下来的讨论中，我们将首先进行一系列消融实验来验证我们的方法的有效性。稍后，我们将在COCO测试数据集上与现有的检测器，如FPN[19]、Mask R-CNN[7]、RetinaNet[20]进行比较。

#### 4.2 消融实验

为了与现有方法进行比较，我们采用Resnet101作为消融研究的骨干网络，类似于第3.2节中描述的设置L。

##### 4.2.1 基线

根据公开可用的R-FCN code(<https://github.com/msracver/Deformable-ConvNets>)提供的详细设置，我们在实验中首先对R-FCN进行评估，记为B1，在COCO mini-validation set中实现了32.1%的mmAP。

通过修改基本设置，我们可以得到一个更强的基线，记作B2，不同之处在于:(1)将图像的短边调整为800像素，将长边的最大尺寸限制为1200像素。我们为RPN设置了5个锚$\{32^2;64^2;128^2;256^2,512^2\}$，因为图像比较大。(2)回归损失明显小于R-CNN分类损失。(3)根据损失大小选取256个排序样本进行反向传播。我们使用每幅图像2000 ROI进行训练，1000 ROI进行测试。如表1所示，我们的方法将mmAP改进了近3个点。

<div align=center>
<img src="zh-cn/img/lightheadrcnn/p4.png" /> 
</div>

*表1: 我们建议方法的基线。基线B1是原始的R-FCN。基线B2涉及到我们复制的实现细节，比如更平衡的损失和更大的图像大小。*


##### 4.2.2 用于RoI变形的小特性映射

根据4.2.1节中描述的基线设置，我们研究了减少特征图通道对ROI warping的影响。为了实现这个目标，我们设计了一个简单的网络结构，如图4所示。整个设计与原始的R-FCN完全相同，以便进行比较。(i)我们将PSRoI池化的feature map通道减少到490`(10x7x7)`。注意到它与原来的R-FCN有很大的不同，原来的R-FCN包括`3969个(81x7x7)`个channel。最后的预测涉及到一个简单的全连接层。



<div align=center>
<img src="zh-cn/img/lightheadrcnn/p5.png" /> 
</div>

*图4： 该网络对用来评估“薄”特征图的影响，我们保持网络与R-FCN相同，只是减少了用于PSRoI池的feature map通道。我们还添加了额外的FC来进行最终预测*

结果如表2所示。虽然通道数目已大幅减少，由3969个减至490个，但我们的表现大致相当。此外，值得注意的是，我们的Light-Head R-CNN设计使我们能够有效地集成特征金字塔网络[19]，如表5所示。对于原始的R-FCN，这几乎是不可能的，因为如果我们想在Conv2 (Resnet stage 2)这样的高分辨率特征映射上执行位置敏感池化(PSROI)，那么内存消耗将非常高。

R-CNN，我们也尝试将PSRoI池化替换为传统的RoI池化，其精度略有提高，为0.3增益。一种假设是RoI池化在第二阶段包含更多的特征(`49x`)，更多的计算可以提高精度。


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p6.png" /> 
</div>

*表2: 减少特征图通道对RoI warping的影响。我们展示了原始的R-FCN和thin feature map的R-FCN。同时，我们也展示了更强基线的影响。我们使用mmAP来表示mAP@[0.5:0.95]的结果。*

**Large separable convolution**: 在我们的设计中，ROI的通道数很小。在原有的实现中，采用了`1x1`卷积的方法对小通道进行变形，降低了特征映射的能力。我们使用`大的可分离卷积`来增强这些特征图，同时保持小的通道。图3显示了大内核的结构。我们设置的超参数是$k=15;C_{mid}=256;C_{out}=490$。在表3中，与我们复制的R-FCN设置B2的结果相比，大核生成的瘦feature map性能提高了0.7个百分点。

<div align=center>
<img src="zh-cn/img/lightheadrcnn/p7.png" /> 
</div>

*表3: 增强瘦特征图对RoI warping的影响。B2是我们实现的强R-FCN基线。*


##### 4.2.3 R-CNNsubnet

在这里，我们评估了R-CNN的轻版本在R-CNN子网中的影响，如图3所示。在RoI子网中采用了一个具有2048通道的单一全连接层(无Dropout)。由于我们池化的feature map在通道中很小(在我们的实验中是10个)，Light R-CNN对于每个区域的分类和回归非常快。Light R-CNN与我们复制的更强更快的R-CNN和R-FCN (B2)的比较如表4所示。

将大核特征图与小R-CNN相结合，实现了37.7 mmAP。在我们的实验中，在相同的基本设置下，更快的R-CNN和R-FCN得到了35.5/35.1 mmAP的结果，这远远低于我们的方法。更重要的是，由于非常薄的功能图和Light R-CNN，我们保持了时间效率，甚至成千上万的候选区域被利用。

<div align=center>
<img src="zh-cn/img/lightheadrcnn/p8.png" /> 
</div>

*表4： Light-Head R-CNN的有效性,R-FCN和Fast R-CNN的基线基于我们的设置L(3.2)。*


#### 4.3 LightHead R-CNN:高精度(尝试了各种trick)

> AliginPool, scale jitter,Multi-scale training

为了与最先进的目标检测器相结合，我们遵循第3.2节中描述的设置L。此外，我们还介绍了在RoIAlign[7]中提出的PSRoI池插值技术。结果如表6所示。它可以带来1.3个百分点的收益。我们还应用了常用的缩放抖动方法。在训练过程中，我们从`{600、700、800,900、1000}`像素中随机抽取尺度样本，然后将图像的短边调整到采样尺度中。图像的最大边缘被限制在1400像素以内，因为较短的边缘可以达到1000像素。由于数据的增加，训练时间也增加了。多尺度训练使mmAP提高了近1个点。最后，我们将原来的0.3阈值替换为0.5，用于非最大抑制(NMS)。通过提高召回率，特别针对人群案例，提高了mmAP的0.6个百分点。

表5还总结了来自最先进的检测器的结果，包括COCO test-dev数据集上的one-stage和two-stage方法。我们的单标度(单卡，batch=1)测试模型可以实现40.8%的mmAP，没有任何附加功能，明显超过所有竞争对手(40.8 vs 39.1)。验证了我们的Light-Head R-CNN将是大型骨干模型的一个很好的选择，并且可以在不增加计算量的情况下获得有希望的结果。图5显示了一些说明性结果。


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p9.png" /> 
</div>


#### 4.4 Light-Head R-CNN:高速度

较大的骨干，如Resnet 101是缓慢的计算速度的原因之一。为了验证Light-Head R-CNN的有效性，我们生成了一个高效的bottleNeck `Xception`(31.1 top 1 error，在ImageNet中为`224x224`)来评估我们的方法。骨干网体系结构如表7所示。采用Xception设计策略，将瓶颈结构中的所有卷积层都替换为channel-wise卷积。但是由于网络的浅层性，我们没有使用在恒等映射[10]中提出的预激活设计。我们的Light-Head R-CNN的实现细节可以在第3.2节的设置“S”中找到。

更具体地说，对于快速推理速度，我们做了如下改变:(i)在设置L时，我们用一个类似Xception的网络替换了Resnet-101主干网。(iii)我们将RPN卷积设置为256个通道，这是原来Faster R-CNN和R-FCN使用的一半。(iv)应用大的可分离卷积，$k=15;C_{mid}=64;C_{out}=490$`(10x7x7)`。由于我们使用的中间通道非常小，所以大的卷积核仍然可以有效地进行推理。(v)我们采用带Algin技术的PSPooling作为我们的RoI warping。注意，如果我们使用RoI-align，它将获得更好的结果。

将我们的方法与最近的快速检测器如YOLO、SSD和DeNet进行了比较。如表8所示，我们的方法在MS COCO上以102FPS获得30.7 mmAP，明显优于YOLO和SSD等快速检测器。图6显示了一些结果。


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p11.png" /> 
</div>

*表5： 在COCO test-dev上对单尺寸单模型测试结果的比较。所有实验均使用Resnet-101作为基本特征提取器(除了G-RMI使用Inception Resnet V2[33])。Light-Head R-CNN达到了一个新的最先进的精度。注意到test-dev的结果与mini-validation略有不同。“ms-train”是指多尺度的训练。*


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p12.png" /> 
</div>

*表6： 进一步改进我们的方法。在COCO小型验证集上对结果进行了评估。*

<div align=center>
<img src="zh-cn/img/lightheadrcnn/p13.png" /> 
</div>

*表7： 高效的Xception类似于我们的快速检测器的体系结构。Comp表示网络的复杂度(FLOPs)*

<div align=center>
<img src="zh-cn/img/lightheadrcnn/p14.png" /> 
</div>

*表8： 比较在COCO test-dev上的快速检测器的结果，通过一个微小的基模型，Light R-CNN在精度和速度上都取得了优异的性能，体现了我们设计方案的灵活性。*


<div align=center>
<img src="zh-cn/img/lightheadrcnn/p10.png" /> 
</div>



### 5.Conclusion

在本文中，我们提出了Light-Head R-CNN，它涉及到一个更好的two-stage目标检测器的设计原则。与传统的two-stage目标检测器比较，如 Faster R-CNN和R-FCN相比，它们通常有一个"沉重"的头部，我们的轻头部设计使我们能够在不影响计算速度的情况下显著提高检测结果。更重要的是，与YOLO、SSD等one-stage快速检测器相比，即使计算速度更快，我们也能获得更好的性能。例如，我们的Light-Head R-CNN加上小的Xception-like基模型可以在102 FPS的速度下实现30.7 mmAP。