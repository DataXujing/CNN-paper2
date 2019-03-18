## Faster R-CNN

------

### 0.摘要

最先进的目标检测网络依靠候选区域筛选来假设目标的位置。SPPnet和Fast R-CNN等研究已经减少了这些检测网络的运行时间，使得候选区域筛选的计算成为瓶颈。Faster R-CNN中，我们引入了一个候选区域的筛选网络（RPN）,该网络与检测网络共享全图像的卷积特征，从而使近乎零成本的候选区域筛选成为可能。RPN是一个全卷积网络，可以同时在每个位置预测目标边界和目标得分。RPN通过端到端的训练，可以生成高质量的候选区域，由Fast R-CNN用于检测。我们将RPN和Fast R-CNN通过共享卷积特征进一步合并为一个单一的网络——使用最近流行的具有“注意力”机制的神经网络术语，RPN组件告诉检测网络在哪里寻找目标。对于非常深的VGG-16模型，我们的检测系统在GPU上的帧率为5fps（包括所有步骤），同时在PASCAL VOC 2007，2012和MS COCO数据集上实现了最新的目标检测精度，每个图像只有300个候选区域。在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是多个比赛中获得第一名输入的基础。代码可公开获得。

### 1.引言

目标检测的最新进展是由候选区域筛选和基于区域的卷积神经网络（R-CNN）的成功驱动的。尽管最初开发的基于区域的CNN计算成本很高，但是由于在各种候选区域中共享卷积，所以其成本已经大大降低了。忽略花费在候选区域筛选上的时间，最新版本Fast R-CNN利用非常深的网络实现了接近实时的速率。现在，候选区域筛选是最新的检测系统中的计算瓶颈。

候选区域筛选方法通常依赖廉价的特征和简练的推断方案。选择性搜索是最流行的方法之一，它贪婪地合并基于设计的低级特征的超级像素。然而，与有效的检测网络（Fast R-CNN）相比，选择性搜索速度慢了一个数量级，在CPU实现中每张图像的时间为2秒。EdgeBoxes[6]目前提供了在候选区域质量和速度之间的最佳权衡，每张图像0.2秒。尽管如此，候选区域的筛选步骤仍然像检测网络那样消耗同样多的运行时间。

有人可能会注意到，基于候选区域的快速CNN利用GPU，而在研究中使用的候选区域的筛选方法在CPU上实现，使得运行时间比较不公平。加速候选区域筛选计算的一个显而易见的方法是将其在GPU上重新实现。这可能是一个有效的工程解决方案，但重新实现忽略了下游检测网络，因此错过了共享计算的重要机会。【用GPU，硬件设备解决这个问题，显然不合适】

在本文中，我们展示了算法的变化——用深度卷积神经网络计算候选区域——一个有效的解决方案，其中在给定检测网络计算的情况下候选区域计算接近零成本。为此，我们引入了新的候选区域筛选网络（RPN），它们共享最先进目标检测网络的卷积层（SPPnet, Fast R-CNN）。通过在测试时共享卷积，计算候选区域的成本很小（例如，每张图像10ms）。

基于区域的检测器所使用的卷积特征映射，如Fast R-CNN，也可以用于生成候选区域。在这些卷积特征之上，我们通过添加一些额外的卷积层来构建RPN，这些卷积层同时在网格上的每个位置上进行bounding boxes 回归和预测分类。因此RPN是一种全卷积网络（FCN），可以针对生成检测候选区域的任务进行端到端的训练。

RPN旨在有效预测具有广泛尺度和长宽比的候选区域。与使用图像金字塔（图1，a）或滤波器金字塔（图1，b）的流行方法[8]，[9]，[1]相比，我们引入新的“锚”盒（anchor boxes）作为多种尺度和长宽比的参考。我们的方案可以被认为是回归参考金字塔（图1，c），它避免了枚举多种比例或长宽比的图像或滤波器。这个模型在使用单尺度图像进行训练和测试时运行良好，从而有利于运行速度。

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p1.png" />
</div>

*图1：解决多尺度和尺寸的不同方案。（a）构建图像和特征映射金字塔，分类器以各种尺度运行。（b）在特征映射上运行具有多个比例/大小的滤波器的金字塔。（c）我们在回归函数中使用参考边界框金字塔。*

为了将RPN与Fast R-CNN 目标检测网络相结合，提出了一种训练方案，在微调候选区域删选任务和微调目标检测任务之间进行交替，同时保持候选区域的固定。该方案快速收敛，并产生两个任务之间共享的具有卷积特征的统一网络。

在PASCAL VOC检测基准数据集上综合评估了Faster R-CNN的方法，其中具有Fast R-CNN的RPN产生的检测精度优于使用选择性搜索的Fast R-CNN的强基准。同时，在测试时几乎免除了选择性搜索的所有计算负担——候选区域筛选的有效运行时间仅为10毫秒。使用非常深的模型，我们的检测方法在GPU上仍然具有5fps的帧率（包括所有步骤），因此在速度和准确性方面是实用的目标检测系统。论文还报告了在MS COCO数据集上的结果，并使用COCO数据研究了在PASCAL VOC上的改进。代码可公开获得https://github.com/shaoqingren/faster_rcnn（在MATLAB中）和https://github.com/rbgirshick/py-faster-rcnn（在Python中）。

RPN和Faster R-CNN的框架已经被采用并推广到其他方法，如3D目标检测[13]，基于部件的检测[14]，实例分割[15]和图像标题[16]。该快速和有效的目标检测系统也已经在Pinterest[17]的商业系统中建立了。

在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是ImageNet检测，ImageNet定位，COCO检测和COCO分割中几个第一名参赛者的基础。RPN完全从数据中学习候选区域，因此可以从更深入和更具表达性的特征（例如[18]中采用的101层残差网络）中轻松获益。Faster R-CNN和RPN也被这些比赛中的其他几个主要参赛者所使用。这些结果表明，Faster R-CNN的方法不仅是一个实用合算的解决方案，而且是一个提高目标检测精度的有效方法。

### 2.相关工作

**候选目标区域** 候选区域筛选方法方面有大量的文献。候选区域筛选方法的综述和比较可以在[19]，[20]，[21]中找到。广泛使用的目标提议方法包括基于超像素分组（例如，选择性搜索[4]，CPMC[22]，MCG[23]）和那些基于滑动窗口的方法（例如窗口中的目标[24]，EdgeBoxes[6]）。目标提议方法被采用为独立于检测器（例如，选择性搜索[4]目标检测器，R-CNN[5]和Fast R-CNN[2]）的外部模块。

用于目标检测的深度网络。R-CNN方法[5]端到端地对CNN进行训练，将提议区域分类为目标类别或背景。R-CNN主要作为分类器，并不能预测目标边界（除了通过边界框回归进行细化）。其准确度取决于区域提议模块的性能（参见[20]中的比较）。一些论文提出了使用深度网络来预测目标边界框的方法[25]，[9]，[26]，[27]。在OverFeat方法[9]中，训练一个全连接层来预测假定单个目标定位任务的边界框坐标。然后将全连接层变成卷积层，用于检测多个类别的目标。MultiBox方法[26]，[27]从网络中生成区域提议，网络最后的全连接层同时预测多个类别不相关的边界框，并推广到OverFeat的“单边界框”方式。这些类别不可知的边界框框被用作R-CNN的提议区域[5]。与我们的全卷积方案相比，MultiBox提议网络适用于单张裁剪图像或多张大型裁剪图像（例如224×224）。MultiBox在提议区域和检测网络之间不共享特征。稍后在我们的方法上下文中会讨论OverFeat和MultiBox。与我们的工作同时进行的，DeepMask方法[28]是为学习分割提议区域而开发的。

卷积[9]，[1]，[29]，[7]，[2]的共享计算已经越来越受到人们的关注，因为它可以有效而准确地进行视觉识别。OverFeat论文[9]计算图像金字塔的卷积特征用于分类，定位和检测。共享卷积特征映射的自适应大小池化（SPP）[1]被开发用于有效的基于区域的目标检测[1]，[30]和语义分割[29]。Fast R-CNN[2]能够对共享卷积特征进行端到端的检测器训练，并显示出令人信服的准确性和速度。


### 3.FASTER R-CNN

我们的目标检测系统，称为Faster R-CNN，由两个模块组成。第一个模块是提议区域的深度全卷积网络，第二个模块是使用提议区域的Fast R-CNN检测器[2]。整个系统是一个单个的，统一的目标检测网络（图2）。使用最近流行的“注意力”[31]机制的神经网络术语，RPN模块告诉Fast R-CNN模块在哪里寻找。在第3.1节中，我们介绍了区域提议网络的设计和属性。在第3.2节中，我们开发了用于训练具有共享特征模块的算法。

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p2.png" />
</div>

*图2：Faster R-CNN是一个单一，统一的目标检测网络。RPN模块作为这个统一网络的“注意力”。*


#### 3.1 区域提议网络

区域提议网络（RPN）以任意大小的图像作为输入，输出一组矩形的目标提议，每个提议都有一个目标得分。我们用全卷积网络[7]对这个过程进行建模，我们将在本节进行描述。因为我们的最终目标是与Fast R-CNN目标检测网络[2]共享计算，所以我们假设两个网络共享一组共同的卷积层。在我们的实验中，我们研究了具有5个共享卷积层的Zeiler和Fergus模型[32]（ZF）和具有13个共享卷积层的Simonyan和Zisserman模型[3]（VGG-16）。

为了生成区域提议，我们在最后的共享卷积层输出的卷积特征映射上滑动一个小网络。这个小网络将输入卷积特征映射的n×n空间窗口作为输入。每个滑动窗口映射到一个低维特征（ZF为256维，VGG为512维，后面是ReLU[33]）。这个特征被输入到两个子全连接层——一个边界框回归层（reg）和一个边界框分类层（cls）。在本文中，我们使用n=3，注意输入图像上的有效感受野是大的（ZF和VGG分别为171和228个像素）。图3（左）显示了这个小型网络的一个位置。请注意，因为小网络以滑动窗口方式运行，所有空间位置共享全连接层。这种架构通过一个n×n卷积层，后面是两个子1×1卷积层（分别用于reg和cls）自然地实现。

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p3.png" />
</div>

*图3：左：区域提议网络（RPN）。右：在PASCAL VOC 2007测试集上使用RPN提议的示例检测。我们的方法可以检测各种尺度和长宽比的目标。*

这里因为论文解释的并不是很详细，在此给读者详细的解释RPN网络的具体操作和结构：

首先，输入图片表示为 Height × Width × Depth 的张量(多维数组)形式，经过预训练 CNN 模型的处理，得到卷积特征图(conv feature map). 即，将 CNN 作为特征提取器，送入下一个部分.
这种技术在迁移学习(Transfer Learning)中比较普遍，尤其是，采用在大规模数据集训练的网络权重，来对小规模数据集训练分类器. 后面会详细介绍，论文中使用了ZF和VGG网络，如下面面的两个图所示：

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p5.jpg" />
</div>

*Faster R-CNN 结构*

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p6.jpg" />
</div>

*VGG16 网络结构*

其对应的就是下图中第一个红框的结构，

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p6.png" />
</div>

进入RPN网络结构（上图中第2个红框结构）：

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p7.png" />
</div>


+ 1.输入图片经卷积网络(如 VGGNet 和 ResNet)处理后, 会输出最后一个卷积层的 feature maps； 

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p8.png" />
</div>

+ 2.在 feature maps 上进行滑窗操作(sliding window). 滑窗尺寸为 n×n, 如 3×3. 
对于每个滑窗, 会生成 9 个 anchors（关于anchors我们将在下一节做详细的介绍）, anchors 具有相同的中心 center=xa,ya, 但 anchors 具有 3 种不同的长宽比(aspect ratios) 和 3 种不同的尺度(scales), 计算是相对于原始图片尺寸的, 如下图:

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p9.png" />
</div>

+ 3. 从 feature maps 中提取 3×3 的空间特征(上图中红色方框部分), 并将其送入一个小网络. 该网络具有两个输出任务分支: classification(cls) 和 regression(reg).regression 分支输出预测的边界框bounding-box: (x, y, w, h).
classification 分支输出一个概率值, 表示 bounding-box 中是否包含 object (classid = 1), 或者是 background (classid = 0), no object.
实际上步骤 3 中预测的 4 个值不是直接预测 H, W, x, y，很显然由于特征图上每个点都是共享权值的，它们根本没法对不同的长宽和位置做出直接的预测（想象一下输入的特征只是图像的卷积特征，完全没有当前 anchor box 的位置大小信息，显然不可能预测出 groud truth 的绝对位置和大小）。这 4 个值是预测如何经过平移与缩放使得当前这个 anchor box 能与 groud truth 尽可能重合（见 R-CNN 论文附录C）

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p10.png" />
</div>


##### 3.1.1 锚点

在每个滑动窗口位置，我们同时预测多个区域提议，其中每个位置可能提议的最大数目表示为k。因此，reg层具有4k个输出，编码k个边界框的坐标，cls层输出2k个分数，估计每个提议是目标或不是目标的概率。相对于我们称之为锚点的k个参考边界框，k个提议是参数化的。锚点位于所讨论的滑动窗口的中心，并与一个尺度和长宽比相关（图3左）。默认情况下，我们使用3个尺度和3个长宽比，在每个滑动位置产生k=9个锚点。对于大小为W×H（通常约为2400）的卷积特征映射，总共有WHk个锚点。

**平移不变的锚点** 我们的方法的一个重要特性是它是平移不变的，无论是在锚点还是计算相对于锚点的区域提议的函数。如果在图像中平移目标，提议应该平移，并且同样的函数应该能够在任一位置预测提议。平移不变特性是由我们的方法保证的。作为比较，MultiBox方法[27]使用k-means生成800个锚点，这不是平移不变的。所以如果平移目标，MultiBox不保证会生成相同的提议。

平移不变特性也减小了模型的大小。MultiBox有(4+1)×800维的全连接输出层，而我们的方法在k=9个锚点的情况下有(4+2)×9维的卷积输出层。因此，对于VGG-16，我们的输出层具有2.8×104个参数（对于VGG-16为512×(4+2)×9），比MultiBox输出层的6.1×106个参数少了两个数量级（对于MultiBox [27]中的GoogleNet[34]为1536×(4+1)×800）。如果考虑到特征投影层，我们的提议层仍然比MultiBox少一个数量级。我们期望我们的方法在PASCAL VOC等小数据集上有更小的过拟合风险。

**多尺度锚点作为回归参考**我们的锚点设计提出了一个新的方案来解决多尺度（和长宽比）。如图1所示，多尺度预测有两种流行的方法。第一种方法是基于图像/特征金字塔，例如DPM[8]和基于CNN的方法[9]，[1]，[2]中。图像在多个尺度上进行缩放，并且针对每个尺度（图1（a））计算特征映射（HOG[8]或深卷积特征[9]，[1]，[2]）。这种方法通常是有用的，但是非常耗时。第二种方法是在特征映射上使用多尺度（和/或长宽比）的滑动窗口。例如，在DPM[8]中，使用不同的滤波器大小（例如5×7和7×5）分别对不同长宽比的模型进行训练。如果用这种方法来解决多尺度问题，可以把它看作是一个“滤波器金字塔”（图1（b））。第二种方法通常与第一种方法联合采用[8]。

由于这种基于锚点的多尺度设计，我们可以简单地使用在单尺度图像上计算的卷积特征，Fast R-CNN检测器也是这样做的[2]。多尺度锚点设计是共享特征的关键组件，不需要额外的成本来处理尺度。

anchor在目标检测中是一个非常重要的概念，在这里和读者做详细的介绍，这里对Faster R-CNN源码中的generate_anchors.py的解读，帮助理解anchor的生成过程:

*来源于：[faster R-CNN中anchors 的生成过程（generate_anchors源码解析）](https://blog.csdn.net/sinat_33486980/article/details/81099093)*

```python
# main函数
if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()   #最主要的就是这个函数
    print time.time() - t
    print a
    from IPython import embed; embed()

```

```python
# 进入到generate_anchors函数中
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
 
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    print ("base anchors",base_anchor)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    print ("anchors after ratio",ratio_anchors)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    print ("achors after ration and scale",anchors)
    return anchors
```

参数有三个：

+ 1.`base_size=16`

这个参数指定了最初的类似感受野的区域大小，因为经过多层卷积池化之后，feature map上一点的感受野对应到原始图像就会是一个区域，这里设置的是16，也就是feature map上一点对应到原图的大小为16x16的区域。也可以根据需要自己设置。

+ 2.`ratios=[0.5,1,2]`

这个参数指的是要将16x16的区域，按照1:2,1:1,2:1三种比例进行变换，如下图所示：

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p11.png" />
</div>

*宽高比变换*

+ 3.`scales=2**np.arange(3, 6)`

这个参数是要将输入的区域，的宽和高进行三种倍数，2^3=8，2^4=16，2^5=32倍的放大，如16x16的区域变成(16x8)x(16x8)=128x128的区域，(16x16)x(16x16)=256x256的区域，(16x32)x(16x32)=512x512的区域，如下图所示

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p12.png" />
</div>

*面积放大变换*

接下来看第一句代码：

```python
base_anchor = np.array([1, 1, base_size, base_size]) - 1
 
'''base_anchor值为[ 0,  0, 15, 15]'''
```
表示最基本的一个大小为16x16的区域，四个值，分别代表这个区域的左上角和右下角的点的坐标。

```python
ratio_anchors = _ratio_enum(base_anchor, ratios)

```

这一句是将前面的16x16的区域进行ratio变化，也就是输出三种宽高比的anchors，这里调用了`_ratio_enum`函数，其定义如下：

```python
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    size = w * h   #size:16*16=256
    size_ratios = size / ratios  #256/ratios[0.5,1,2]=[512,256,128]
    #round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
    ws = np.round(np.sqrt(size_ratios)) #ws:[23 16 11]
    hs = np.round(ws * ratios)    #hs:[12 16 22],ws和hs一一对应。as:23&12
    #给定一组宽高向量，输出各个预测窗口，也就是将（宽，高，中心点横坐标，中心点纵坐标）的形式，转成
    #四个坐标值的形式
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  
    return anchors
```

输入参数为一个anchor(四个坐标值表示)和三种宽高比例`（0.5,1,2）`

在这个函数中又调用了一个`_whctrs`函数，这个函数定义如下，其主要作用是将输入的anchor的四个坐标值转化成（宽，高，中心点横坐标，中心点纵坐标）的形式。

```python
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

```

通过这个函数变换之后将原来的anchor坐标`（0，0，15，15）`转化成了`w:16,h:16,x_ctr=7.5,y_ctr=7.5`的形式，接下来按照比例变化的过程见_ratio_enum的代码注释。最后该函数输出的变换了三种宽高比的anchor如下：

```python
ratio_anchors = _ratio_enum(base_anchor, ratios)
'''[[ -3.5,   2. ,  18.5,  13. ],
    [  0. ,   0. ,  15. ,  15. ],
    [  2.5,  -3. ,  12.5,  18. ]]'''

```

进行完上面的宽高比变换之后，接下来执行的是面积的scale变换，

```python
 anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
```

这里最重要的是`_scale_enum`函数，该函数定义如下，对上一步得到的`ratio_anchors`中的三种宽高比的anchor，再分别进行三种scale的变换，也就是三种宽高比，搭配三种scale，最终会得到9种宽高比和scale 的anchors。这就是论文中每一个点对应的9种anchors。


```python
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
 
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

```

`scale_enum`函数中也是首先将宽高比变换后的每一个`ratio_anchor`转化成（宽，高，中心点横坐标，中心点纵坐标）的形式，再对宽和高均进行scale倍的放大，然后再转换成四个坐标值的形式。最终经过宽高比和scale变换得到的9种尺寸的anchors的坐标如下：

```python
anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
'''
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]
'''

```

下面这个表格对比了9种尺寸的anchor的变换：

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p13.png" />
</div>

以我的理解，得到的这些anchors的坐标是相对于原始图像的，因为feature map的大小一般也就是60x40这样的大小，而上面得到的这些坐标都是好几百，因此是相对于原始大图像而设置的这9种组合的尺寸，这些尺寸基本上可以包含图像中的任何物体，如果画面里出现了特大的物体，则这个scale就要相应的再调整大一点，来包含特大的物体。

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p14.png" />
</div>


因为是对提取的 Convwidth×Convheight×ConvDepth 卷积特征图进行处理，因此，在 Convwidth×Convheight的每个点创建 anchors. 需要理解的是，虽然 anchors 是基于卷积特征图定义的，但最终的 anchos 是相对于原始图片的.由于只有卷积层和 pooling 层，特征图的维度是与原始图片的尺寸成比例关系的. 即，数学地表述，如果图片尺寸 `w×h`，特征图的尺寸则是 `w/r×h/r`. 其中，r 是下采样率(subsampling ratio). 如果在卷积特征图空间位置定义 anchor，则最终的图片会是由 r 像素划分的 anchors 集. 在 VGG 中， `r=16`.


<div align=center>
<img src="zh-cn/img/faster-R-CNN/p14.jpg" />
</div>

*原始图片上的 Anchor Centers*

##### 3.1.2 损失函数

为了训练RPN，我们为每个锚点分配一个二值类别标签（是目标或不是目标）。我们给两种锚点分配一个正标签：（i）具有与实际边界框的重叠最高交并比（IoU）的锚点，或者（ii）具有与实际边界框的重叠超过0.7 IoU的锚点。注意，单个真实边界框可以为多个锚点分配正标签。通常第二个条件足以确定正样本；但我们仍然采用第一个条件，因为在一些极少数情况下，第二个条件可能找不到正样本。对于所有的真实边界框，如果一个锚点的IoU比率低于0.3，我们给非正面的锚点分配一个负标签。既不正面也不负面的锚点不会有助于训练目标函数。

根据这些定义，我们对目标函数Fast R-CNN[2]中的多任务损失进行最小化。我们对图像的损失函数定义为：

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p15.png" />
</div>

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p16.png" />
</div>

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p17.png" />
</div>

然而，我们的方法通过与之前的基于RoI（感兴趣区域）方法[1]，[2]不同的方式来实现边界框回归。在[1]，[2]中，对任意大小的RoI池化的特征执行边界框回归，并且回归权重由所有区域大小共享。在我们的公式中，用于回归的特征在特征映射上具有相同的空间大小（3×3）。为了说明不同的大小，学习一组k个边界框回归器。每个回归器负责一个尺度和一个长宽比，而k个回归器不共享权重。因此，由于锚点的设计，即使特征具有固定的尺度/比例，仍然可以预测各种尺寸的边界框。

##### 3.1.3 训练RPN

RPN可以通过反向传播和随机梯度下降（SGD）进行端对端训练[35]。我们遵循[2]的“以图像为中心”的采样策略来训练这个网络。每个小批量数据都从包含许多正面和负面示例锚点的单张图像中产生。对所有锚点的损失函数进行优化是可能的，但是这样会偏向于负样本，因为它们是占主导地位的。取而代之的是，我们在图像中随机采样256个锚点，计算一个小批量数据的损失函数，其中采样的正锚点和负锚点的比率可达1:1。如果图像中的正样本少于128个，我们使用负样本填充小批量数据。

我们通过从标准方差为0.01的零均值高斯分布中提取权重来随机初始化所有新层。所有其他层（即共享卷积层）通过预训练的ImageNet分类模型[36]来初始化，如同标准实践[5]。我们调整ZF网络的所有层，以及VGG网络的conv3_1及其之上的层以节省内存[2]。对于60k的小批量数据，我们使用0.001的学习率，对于PASCAL VOC数据集中的下一个20k小批量数据，使用0.0001。我们使用0.9的动量和0.0005的重量衰减[37]。我们的实现使用Caffe[38]。


#### 3.2 RPN和Fast R-CNN共享特征

到目前为止，我们已经描述了如何训练用于区域提议生成的网络，没有考虑将利用这些提议的基于区域的目标检测CNN。对于检测网络，我们采用Fast R-CNN[2]。接下来我们介绍一些算法，学习由RPN和Fast R-CNN组成的具有共享卷积层的统一网络（图2）。

独立训练的RPN和Fast R-CNN将以不同的方式修改卷积层。因此，我们需要开发一种允许在两个网络之间共享卷积层的技术，而不是学习两个独立的网络。我们讨论三个方法来训练具有共享特征的网络：

（一）交替训练。在这个解决方案中，我们首先训练RPN，并使用这些提议来训练Fast R-CNN。由Fast R-CNN微调的网络然后被用于初始化RPN，并且重复这个过程。这是本文所有实验中使用的解决方案。

（二）近似联合训练。在这个解决方案中，RPN和Fast R-CNN网络在训练期间合并成一个网络，如图2所示。在每次SGD迭代中，前向传递生成区域提议，在训练Fast R-CNN检测器将这看作是固定的、预计算的提议。反向传播像往常一样进行，其中对于共享层，组合来自RPN损失和Fast R-CNN损失的反向传播信号。这个解决方案很容易实现。但是这个解决方案忽略了关于提议边界框的坐标（也是网络响应）的导数，因此是近似的。在我们的实验中，我们实验发现这个求解器产生了相当的结果，与交替训练相比，训练时间减少了大约25−50%。这个求解器包含在我们发布的Python代码中。

（三）非近似的联合训练。如上所述，由RPN预测的边界框也是输入的函数。Fast R-CNN中的RoI池化层[2]接受卷积特征以及预测的边界框作为输入，所以理论上有效的反向传播求解器也应该包括关于边界框坐标的梯度。在上述近似联合训练中，这些梯度被忽略。在一个非近似的联合训练解决方案中，我们需要一个关于边界框坐标可微分的RoI池化层。这是一个重要的问题，可以通过[15]中提出的“RoI扭曲”层给出解决方案，这超出了本文的范围。

**四步交替训练**。在本文中，我们采用实用的四步训练算法，通过交替优化学习共享特征。在第一步中，我们按照3.1.3节的描述训练RPN。该网络使用ImageNet的预训练模型进行初始化，并针对区域提议任务进行了端到端的微调。在第二步中，我们使用由第一步RPN生成的提议，由Fast R-CNN训练单独的检测网络。该检测网络也由ImageNet的预训练模型进行初始化。此时两个网络不共享卷积层。在第三步中，我们使用检测器网络来初始化RPN训练，但是我们修正共享的卷积层，并且只对RPN特有的层进行微调。现在这两个网络共享卷积层。最后，保持共享卷积层的固定，我们对Fast R-CNN的独有层进行微调。因此，两个网络共享相同的卷积层并形成统一的网络。类似的交替训练可以运行更多的迭代，但是我们只观察到可以忽略的改进。


#### 3.3 实现细节

我们在单尺度图像上训练和测试区域提议和目标检测网络[1]，[2]。我们重新缩放图像，使得它们的短边是s=600像素[2]。多尺度特征提取（使用图像金字塔）可能会提高精度，但不会表现出速度与精度的良好折衷[2]。在重新缩放的图像上，最后卷积层上的ZF和VGG网络的总步长为16个像素，因此在调整大小（〜500×375）之前，典型的PASCAL图像上的总步长为〜10个像素。即使如此大的步长也能提供良好的效果，尽管步幅更小，精度可能会进一步提高。

对于锚点，我们使用了3个尺度，边界框面积分别为128^2，256^2和512^2个像素，以及1:1，1:2和2:1的长宽比。这些超参数不是针对特定数据集仔细选择的，我们将在下一节中提供有关其作用的消融实验。如上所述，我们的解决方案不需要图像金字塔或滤波器金字塔来预测多个尺度的区域，节省了大量的运行时间。图3（右）显示了我们的方法在广泛的尺度和长宽比方面的能力。表1显示了使用ZF网络的每个锚点学习到的平均提议大小。我们注意到，我们的算法允许预测比基础感受野更大。这样的预测不是不可能的——如果只有目标的中间部分是可见的，那么仍然可以粗略地推断出目标的范围。

跨越图像边界的锚盒需要小心处理。在训练过程中，我们忽略了所有的跨界锚点，所以不会造成损失。对于一个典型的1000×600的图片，总共将会有大约20000（≈60×40×9）个锚点。跨界锚点被忽略，每张图像约有6000个锚点用于训练。如果跨界异常值在训练中不被忽略，则会在目标函数中引入大的，难以纠正的误差项，且训练不会收敛。但在测试过程中，我们仍然将全卷积RPN应用于整张图像。这可能会产生跨边界的提议边界框，我们剪切到图像边界。

一些RPN提议互相之间高度重叠。为了减少冗余，我们在提议区域根据他们的cls分数采取非极大值抑制（NMS）。我们将NMS的IoU阈值固定为0.7，这就给每张图像留下了大约2000个提议区域。正如我们将要展示的那样，NMS不会损害最终的检测准确性，但会大大减少提议的数量。在NMS之后，我们使用前N个提议区域来进行检测。接下来，我们使用2000个RPN提议对Fast R-CNN进行训练，但在测试时评估不同数量的提议。

### 4.实验

论文分别在PASCAL VOC，MS COCO数据集上进行实验，详细的自己查看Faster R-CNN的paper原文。

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p18.png" />
</div>

*使用Faster R-CNN系统在PASCAL VOC 2007测试集上选择的目标检测结果示例。该模型是VGG-16，训练数据是07+12 trainval（2007年测试集中73.2%的mAP）。我们的方法检测广泛的尺度和长宽比目标。每个输出框都与类别标签和[0，1]之间的softmax分数相关联。使用0.6的分数阈值来显示这些图像。获得这些结果的运行时间为每张图像198ms，包括所有步骤。*

<div align=center>
<img src="zh-cn/img/faster-R-CNN/p19.png" />
</div>

*使用Faster R-CNN系统在MS COCO test-dev数据集上选择的目标检测结果示例。该模型是VGG-16，训练数据是COCO训练数据（在测试开发数据集上为42.7%的mAP@0.5）。每个输出框都与一个类别标签和[0, 1]之间的softmax分数相关联。使用0.6的分数阈值来显示这些图像。对于每张图像，一种颜色表示该图像中的一个目标类别。*


### 5.结论

我们已经提出了RPN来生成高效，准确的区域提议。通过与下游检测网络共享卷积特征，区域提议步骤几乎是零成本的。我们的方法使统一的，基于深度学习的目标检测系统能够以接近实时的帧率运行。学习到的RPN也提高了区域提议的质量，从而提高了整体的目标检测精度。


### Reference


[1] K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling in deep convolutional networks for visual recognition,” in European Conference on Computer Vision (ECCV), 2014.

[2] R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision (ICCV), 2015.

[3] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in International Conference on Learning Representations (ICLR), 2015.

[4] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders, “Selective search for object recognition,” International
Journal of Computer Vision (IJCV), 2013.

[5] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

[6] C. L. Zitnick and P. Dollár, “Edge boxes: Locating object proposals from edges,” in European Conference on Computer Vision(ECCV),2014.

[7] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[8] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan, “Object detection with discriminatively trained part-based models,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2010.

[9] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun, “Overfeat: Integrated recognition, localization and detection using convolutional networks,” in International Conference on Learning Representations (ICLR), 2014.

[10] S. Ren, K. He, R. Girshick, and J. Sun, “FasterR-CNN: Towards real-time object detection with region proposal networks,” in
Neural Information Processing Systems (NIPS), 2015.

[11] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The PASCAL Visual Object Classes Challenge 2007 (VOC2007) Results,” 2007.

[12] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft COCO: Common Objects in Context,” in European Conference on Computer Vision (ECCV), 2014.

[13] S. Song and J. Xiao, “Deep sliding shapes for amodal 3d object detection in rgb-d images,” arXiv:1511.02300, 2015.

[14] J. Zhu, X. Chen, and A. L. Yuille, “DeePM: A deep part-based model for object detection and semantic part localization,” arXiv:1511.07131, 2015.

[15] J. Dai, K. He, and J. Sun, “Instance-aware semantic segmentation via multi-task network cascades,” arXiv:1512.04412, 2015.

[16] J. Johnson, A. Karpathy, and L. Fei-Fei, “Densecap: Fully convolutional localization networks for dense captioning,” arXiv:1511.07571, 2015.

[17] D. Kislyuk, Y. Liu, D. Liu, E. Tzeng, and Y. Jing, “Human curation and convnets: Powering item-to-item recommendations on pinterest,” arXiv:1511.04003, 2015.

[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” arXiv:1512.03385, 2015.

[19] J. Hosang, R. Benenson, and B. Schiele, “How good are detection proposals, really?” in British Machine Vision Conference (BMVC), 2014.

[20] J. Hosang, R. Benenson, P. Dollar, and B. Schiele, “What makes for effective detection proposals?” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2015.

[21] N. Chavali, H. Agrawal, A. Mahendru, and D. Batra, “Object-Proposal Evaluation Protocol is ’Gameable’,” arXiv: 1505.05836, 2015.

[22] J. Carreira and C. Sminchisescu, “CPMC: Automatic object segmentation using constrained parametric min-cuts,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2012.

[23] P. Arbelaez, J. Pont-Tuset, J. T. Barron, F. Marques, and J. Malik, “Multiscale combinatorial grouping,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

[24] B. Alexe, T. Deselaers, and V. Ferrari, “Measuring the objectness of image windows,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2012.

[25] C. Szegedy, A. Toshev, and D. Erhan, “Deep neural networks for object detection,” in Neural Information Processing Systems (NIPS), 2013.

[26] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov, “Scalable object detection using deep neural networks,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

[27] C. Szegedy, S. Reed, D. Erhan, and D. Anguelov, “Scalable, high-quality object detection,” arXiv:1412.1441 (v1), 2015.

[28] P. O. Pinheiro, R. Collobert, and P. Dollar, “Learning to segment object candidates,” in Neural Information Processing Systems (NIPS), 2015.

[29] J. Dai, K. He, and J. Sun, “Convolutional feature masking for joint object and stuff segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[30] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun, “Object detection networks on convolutional feature maps,” arXiv:1504.06066, 2015.

[31] J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, “Attention-based models for speech recognition,” in Neural Information Processing Systems (NIPS), 2015.

[32] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional neural networks,” in European Conference on Computer Vision (ECCV), 2014.

[33] V. Nair and G. E. Hinton, “Rectified linear units improve restricted boltzmann machines,” in International Conference on Machine Learning (ICML), 2010.

[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, and A. Rabinovich, “Going deeper with convolutions,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[35] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel, “Backpropagation applied to handwritten zip code recognition,” Neural computation, 1989.

[36] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, “ImageNet Large Scale Visual Recognition Challenge,” in International Journal of Computer Vision (IJCV), 2015.

[37] A. Krizhevsky, I. Sutskever, and G. Hinton, “Imagenet classification with deep convolutional neural networks,” in Neural Information Processing Systems (NIPS), 2012.

[38] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell, “Caffe: Convolutional architecture for fast feature embedding,” arXiv:1408.5093, 2014.

[39] K. Lenc and A. Vedaldi, “R-CNN minus R,” in British Machine Vision Conference (BMVC), 2015.