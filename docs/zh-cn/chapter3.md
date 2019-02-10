## R-CNN系列 & SPP-net
------

我们本部分的学习路线为:R-CNN, Selective Search, SPP-net

### 1.R-CNN

R-CNN系列论文(R-CNN,fast R-CNN,faster R-CNN,mask R-CNN)是深度学习进行目标检测的鼻祖论文，都是沿用了R-CNN的思路，我们本节内容来自《Rich feature hierarchies for accurate object detection and semantic segmentation》(2014 CVRR)的R-CNN的论文。

其实在R-CNN之前，overfeat已经是用深度学习的方法在做目标检测(关于overfeat的相关学习资料，已经放在了我的Github的repo中),但是R-CNN是第一个可以真正以工业级应用的解决方案。(这也是我们为什么介绍R-CNN系列的主要原因),可以说改变了目标检测的主要研究思路，紧随其后的系列文章都沿用R-CNN。

<div align=center>
<img src="zh-cn/img/R-CNN/pic1.png" /> 
</div>
**图1：CV中的主要问题:Classify,localization(单目标),detection(多目标)**

**0.摘要：**

过去几年，在权威数据集PASCAL上，物体检测的效果已经达到一个稳定水平。效果最好的方法是融合了多种低维图像特征和高维上下文环境的复杂融合系统。在这篇论文里，我们提出了一种简单并且可扩展的检测算法，可以将mAP在VOC2012最好结果的基础上提高30%以上——达到了53.3%。我们的方法结合了两个关键的因素：

1.在候选区域上自下而上使用大型卷积神经网络(CNNs)，用以定位和分割物体。

2.当带标签的训练数据不足时，先针对辅助任务进行有监督预训练，再进行特定任务的调优，就可以产生明显的性能提升。

因为我们把region proposal（定位）和CNNs结合起来，所以该方法被称为R-CNN： Regions with CNN features。把R-CNN效果跟OverFeat比较了下（OverFeat是最近提出的在与我们相似的CNN特征下采用滑动窗口进行目标检测的一种方法，Overfeat:改进了AlexNet，并用图像缩放和滑窗方法在test数据集上测试网络；提出了一种图像定位的方法；最后通过一个卷积网络来同时进行分类，定位和检测三个计算机视觉任务，并在ILSVRC2013中获得了很好的结果。），结果发现RCNN在200类ILSVRC2013检测数据集上的性能明显优于OVerFeat。项目地址:<https://github.com/rbgirshick/rcnn>(MatLab)

**1.介绍**

特征很重要。在过去十年，各类视觉识别任务基本都建立在对SIFT[29]和HOG[7]特征的使用。但如果我们关注一下PASCAL VOC对象检测[15]这个经典的视觉识别任务，就会发现，2010-2012年进展缓慢，取得的微小进步都是通过构建一些集成系统和采用一些成功方法的变种才达到的。 【描述现状】

SIFT和HOG是块方向直方图(blockwise orientation histograms)，两篇论文已经更新在Github的repo中，一种类似大脑初级皮层V1层复杂细胞的表示方法。但我们知道识别发生在多个下游阶段，（我们是先看到了一些特征，然后才意识到这是什么东西）也就是说对于视觉识别来说，更有价值的信息，是层次化的，多个阶段的特征。 【关于SIFT&HOG】

"神经认知机",一种受生物学启发用于模式识别的层次化、移动不变性模型，算是这方面最早的尝试,但神经认知机缺乏监督学习算法。Lecun等人的工作表明基于反向传播的随机梯度下降(SGD)对训练卷积神经网络（CNNs）非常有效，CNNs被认为是继承自neocognitron的一类模型。 【神经认知机】

CNNs在1990年代被广泛使用，但随即便因为SVM的崛起而淡出研究主流。2012年，Krizhevsky等人在ImageNet大规模视觉识别挑战赛(ILSVRC)上的出色表现重新燃起了世界对CNNs的兴趣（AlexNet）。他们的成功在于在120万的标签图像上使用了一个大型的CNN，并且对LeCUN的CNN进行了一些改造（比如ReLU和Dropout Regularization）。 【CNN的崛起】

这个ImangeNet的结果的重要性在ILSVRC2012 workshop上得到了热烈的讨论。提炼出来的核心问题是：ImageNet上的CNN分类结果在何种程度上能够应用到PASCAL VOC挑战的物体检测任务上？【CNN何时使用到目标检测】

我们通过连接图像分类和目标检测，回答了这个问题。本论文是第一个说明在PASCAL VOC的物体检测任务上CNN比基于简单类HOG特征的系统有大幅的性能提升。我们主要关注了两个问题：使用深度网络定位物体和在小规模的标注数据集上进行大型网络模型的训练。 【R-CNN解决的问题】

与图像分类不同的是检测需要定位一个图像内的许多物体。一个方法是将框定位看做是回归问题。但Szegedy等人的工作说明这种策略并不work（在VOC2007上他们的mAP是30.5%，而我们的达到了58.5%）。【将定位问题单纯作为回归解决效果并不好】

另一个可替代的方法是使用【滑动窗口探测器】，通过这种方法使用CNNs至少已经有20年的时间了，通常用于一些特定的种类如人脸，行人等。为了获得较高的空间分辨率，这些CNNs都采用了两个卷积层和两个池化层。我们本来也考虑过使用滑动窗口的方法，但是由于网络层次更深，输入图片有非常大的感受野（195×195）and 步长（32×32），这使得采用滑动窗口的方法充满挑战。【感受野大，滑动窗口出来的边界不准确】

我们是通过操作”recognition using regions”[21]范式，解决了CNN的定位问题。
+ 测试时，对这每张图片，产生了接近2000个与类别无关的region proposal,
+ 对每个CNN抽取了一个固定长度的特征向量，
+ 然后借助专门针对特定类别数据的线性SVM对每个区域进行分类。

我们不考虑region的大小，使用放射图像变形的方法来对每个不同形状的region proposal产生一个固定长度的作为CNN输入的特征向量（也就是把不同大小的proposal放到同一个大小）。图2展示了我们方法的全貌并突出展示了一些实验结果。由于我们结合了Region proposals[21]和CNNs，所以起名R-CNN：Regions with CNN features。【R-CNN的由来】

<div align=center>
<img src="zh-cn/img/R-CNN/pic2.png" /> 
</div>
**图2：R-CNN目标检测系统过程. （1）获取一张输入图片，（2）产生2000个与类别无关的region proposal，（3）用大型的卷积计算备选区域的特征，（4）使用线性SVM对每一个定位进行分类**

检测中面对的第二个挑战是标签数据太少，现在可获得的数据远远不够用来训练一个大型卷积网络。传统方法多是采用无监督与训练，再进行有监督调优。本文的第二个核心贡献是在辅助数据集（ILSVRC）上进行有监督预训练，再在小数据集上针对特定问题进行调优。这是在训练数据稀少的情况下一个非常有效的训练大型卷积神经网络的方法。我们的实验中，针对检测的调优将mAP提高了8个百分点。调优后，我们的系统在VOC2010上达到了54%的mAP，远远超过高度优化的基于HOG的可变性部件模型（deformable part model，DPM）
【DPM:多尺度形变部件模型，连续获得07-09的检测冠军，2010年其作者Felzenszwalb Pedro被VOC授予”终身成就奖”。DPM把物体看成了多个组成的部件（比如人脸的鼻子、嘴巴等），用部件间的关系来描述物体，这个特性非常符合自然界很多物体的非刚体特征。DPM可以看做是HOG+SVM的扩展，很好的继承了两者的优点，在人脸检测、行人检测等任务上取得了不错的效果，但是DPM相对复杂，检测速度也较慢，从而也出现了很多改进的方法。】【挑战2及解决办法】

R-CNN计算高效： 原因都是小型矩阵的乘积，特征在不同类别间共享；HOG-like特征的一个优点是简单性：能够很容易明白提取到的特征是什么（可视化出来）。介绍技术细节之前，我们提醒大家由于R-CNN是在推荐区域上进行操作，所以可以很自然地扩展到语义分割任务上。只要很小的改动，我们就在PASCAL VOC语义分割任务上达到了很有竞争力的结果，在VOC2011测试集上平均语义分割精度达到了47.9%。【R-CNN的其他应用】

**2.用R-CNN做目标检测**

我们的物体检测系统有三个模块构成。

+ 第一个，产生类别无关的region proposal。这些推荐定义了一个候选检测区域的集合；
+ 第二个是一个大型卷积神经网络，用于从每个区域抽取特定大小的特征向量；
+ 第三个是一个指定类别的线性SVM。

本部分，将展示每个模块的设计，并介绍他们的测试阶段的用法，以及参数是如何学习的细节，最后给出在PASCAL VOC 2010-12和ILSVRC2013上的检测结果。

**2.1模块设计**

【region proposal：区域推荐】 近来有很多研究都提出了产生类别无关区域推荐的方法比如: objectness（物体性）[1]，selective search（选择性搜索）[39]，category-independent object proposals(类别无关物体推荐)[14]，constrained parametric min-cuts（受限参最小剪切, CPMC)[5]，multi-scal combinatorial grouping(多尺度联合分组)[3]，以及Ciresan[6]等人的方法,将CNN用在规律空间块裁剪上以检测有丝分裂细胞，也算是一种特殊的区域推荐类型。由于R-CNN对特定区域推荐算法是不关心的，所以我们采用了选择性搜索[39]以方便和前面的工作进行可控的比较。[region proposal方法，建议自行学习]

【Feature extraction: 特征提取】我们使用Krizhevsky等人所描述的CNN的一个Caffe实现版本[24]对每个推荐区域抽取一个4096维度的特征向量把一个输入为277*277大小的图片，通过五个卷积层和两个全连接层进行前向传播,最终得到一个4096-D的特征向量。读者可以参考AlexNet获得更多的网络架构细节。为了计算region proposal的特征，我们首先要对图像进行转换，使得它符合CNNC的输入（架构中的CNN只能接受固定大小：277*277）这个变换有很多办法，我们使用了最简单的一种。无论候选区域是什么尺寸和宽高比，我们都把候选框变形成想要的尺寸,。具体的，变形之前，我们先在候选框周围加上16的padding,再进行**各向异性缩放**。这种形变使得mAp提高了3到5个百分点。在补充材料中，作者对比了各向异性和各向同性缩放缩放方法。

**2.2测试阶段的物体检测**

测试阶段，在测试图像上使用selective search抽取2000个推荐区域（实验中，我们使用了选择性搜索的快速模式）（关于selective search我们在下文中会详细讲解）然后变形每一个推荐区域，再通过CNN前向传播计算出特征。然后我们使用对每个类别训练出的SVM给整个特征向量中的每个类别单独打分。【对每一个框使用每个类别的SVM进行打分】然后给出一张图像中所有的打分区域，然后使用NMS【贪婪非最大化抑制算法】（每个类别是独立进行的），拒绝掉一些和高分区域的IOU大于阈值的候选框。


【**运行时间的分析**】两个特性让检测变得很高效。首先，所有的CNN参数都是跨类别共享的。（参数共享）其次，通过CNN计算的特征向量相比其他通用方法（比如spatial pyramids with bag-of-visual-word encodings）维度是很低的。（低维特征）这种共享的结果就是计算推荐区域特征的耗时可以分摊到所有类别的头上（GPU：每张图13s，CPU：每张图53s）。

唯一的和具体类别有关的计算是特征向量和SVM权重和点积，以及NMS实践中，所有的点积都可以批量化成一个单独矩阵间运算。特征矩阵的典型大小是2000×4096，SVM权重的矩阵是4096xN，其中N是类别的数量。

分析表明R-CNN可以扩展到上千个类别，而不需要借用近似技术（如hashing）。及时有10万个类别，矩阵乘法在现代多核CPU上只需要10s而已。但这种高效不仅仅是因为使用了区域推荐和共享特征。

**2.3训练**

【**有监督的预训练 **】我们在大型辅助训练集ILSVRC2012分类数据集（没有约束框数据）上预训练了CNN。预训练采用了Caffe的CNN库。总体来说，我们的CNN十分接近krizhevsky等人的网络的性能，在ILSVRC2012分类验证集在top-1错误率上比他们高2.2%。差异主要来自于训练过程的简化。

【**特定领域的参数调优 **】为了让我们的CNN适应新的任务（即检测任务）和新的领域（变形后的推荐窗口）。我们只使用变形后的推荐区域对CNN参数进行SGD训练。我们替换掉了ImageNet专用的1000-way分类层，换成了一个随机初始化的21-way分类层，（其中20是VOC的类别数，1代表背景）而卷积部分都没有改变，我们对待所有的推荐区域，如果其和真实标注的框的IoU>= 0.5就认为是正例，否则就是负例，SGD开始的learning_rate为0.001（是初始化预训练时的十分之一），这使得调优得以有效进行而不会破坏初始化的成果。每轮SGD迭代，我们统一使用32个正例窗口（跨所有类别）和96个背景窗口，即每个mini-batch的大小是128。另外我们倾向于采样正例窗口，因为和背景相比他们很稀少。

【**目标种类分类器**】思考一下检测汽车的二分类器。很显然，一个图像区域紧紧包裹着一辆汽车应该就是正例。同样的，没有汽车的就是背景区域，也就是负例。较为不明确的是怎样标注哪些只和汽车部分重叠的区域。我们使用IoU重叠阈值来解决这个问题，低于这个阈值的就是负例。这个阈值我们选择了0.3，是在验证集上基于{0, 0.1, … 0.5}通过网格搜索得到的。我们发现认真选择这个阈值很重要。如果设置为0.5，可以提升mAP5个点，设置为0，就会降低4个点。正例就严格的是标注的框

一旦特征提取出来，并应用标签数据，我们优化了每个类的线性SVM。由于训练数据太大，难以装进内存，我们选择了标准的hard negative mining method【难负例挖掘算法，用途就是负例数据不平衡，而负例分赛代表性又不够的问题，hard negative就是每次把那些顽固的棘手的错误，在送回去训练，训练到你的成绩不在提升为止，这个过程叫做hard negative mining】

高难负例挖掘算法收敛很快，实践中只要在所有图像上经过一轮训练，mAP就可以基本停止增加了。 附录B中，讨论了，为什么在fine-tunning和SVM训练这两个阶段，我们定义得正负样例是不同的。【fine-tunning阶段是由于CNN对小样本容易过拟合，需要大量训练数据，故对IoU限制宽松： IoU>0.5的建议框为正样本，否则为负样本； SVM这种机制是由于其适用于小样本训练，故对样本IoU限制严格：Ground Truth为正样本，与Ground Truth相交IoU＜0.3的建议框为负样本。】

我们也会讨论为什么训练一个分类器是必要的，而不只是简单地使用来自调优后的CNN的最终fc8层的输出。【为什么单独训练了一个SVM而不是直接用softmax，作者提到，刚开始时只是用了ImageNet预训练了CNN，并用提取的特征训练了SVMs，此时用正负样本标记方法就是前面所述的0.3,后来刚开始使用fine-tuning时，也使用了这个方法，但是发现结果很差，于是通过调试选择了0.5这个方法，作者认为这样可以加大样本的数量，从而避免过拟合。然而，IoU大于0.5就作为正样本会导致网络定位准确度的下降，故使用了SVM来做检测，全部使用ground-truth样本作为正样本，且使用非正样本的，且IoU大于0.3的“hard negatives”，提高了定位的准确度】

**2.4在PASCAL VOC 2010-12上的结果**

在数据集： PASCAL 2010-12:

<div align=center>
<img src="zh-cn/img/R-CNN/pic3.png" /> 
</div>
**原paper的Table1**

在数据集ILSVR2013数据集上得到了相似的结果

**3.可视化、消融、模型的错误**

**3.1可视化学习到的特征**（如何展示CNN每层学到的东西，了解）

直接可视化第一层filters非常容易理解，它们主要捕获方向性边缘和对比色。难以理解的是后面的层。Zeiler and Fergus提出了一种可视化的很棒的反卷积办法。我们则使用了一种简单的非参数化方法，直接展示网络学到的东西。这个想法是单一输出网络中一个特定单元（特征），然后把它当做一个正确类别的物体检测器来使用。 
方法是这样的，先计算所有抽取出来的推荐区域（大约1000万），计算每个区域所导致的对应单元的激活值，然后按激活值对这些区域进行排序，然后进行最大值抑制，最后展示分值最高的若干个区域。这个方法让被选中的单元在遇到他想激活的输入时“自己说话”。我们避免平均化是为了看到不同的视觉模式和深入观察单元计算出来的不变性。 
我们可视化了第五层的池化层pool5，是卷积网络的最后一层，feature_map(卷积核和特征数的总称)的大小是6 x 6 x 256 = 9216维。忽略边界效应，每个pool5单元拥有195×195的感受野，输入是227×227。pool5中间的单元，几乎是一个全局视角，而边缘的单元有较小的带裁切的支持。 
图4的每一行显示了对于一个pool5单元的最高16个激活区域情况，这个实例来自于VOC 2007上我们调优的CNN，这里只展示了256个单元中的6个（附录D包含更多）。我们看看这些单元都学到了什么。第二行，有一个单元看到狗和斑点的时候就会激活，第三行对应红斑点，还有人脸，当然还有一些抽象的模式，比如文字和带窗户的三角结构。这个网络似乎学到了一些类别调优相关的特征，这些特征都是形状、纹理、颜色和材质特性的分布式表示。而后续的fc6层则对这些丰富的特征建立大量的组合来表达各种不同的事物。

**3.2消融研究（Ablation studies）**

ablation study 就是为了研究模型中所提出的一些结构是否有效而设计的实验。如你提出了某某结构，但是要想确定这个结构是否有利于最终的效果，那就要将去掉该结构的网络与加上该结构的网络所得到的结果进行对比，这就是ablation study。也就是（控制变量法）

【**没有调优的各层性能**】

为了理解哪一层对于检测的性能十分重要，我们分析了CNN最后三层的每一层在VOC2007上面的结果。Pool5在3.1中做过剪短的表述。最后两层下面来总结一下。 

fc6是一个与pool5连接的全连接层。为了计算特征，它和pool5的feature map（reshape成一个9216维度的向量）做了一个4096×9216的矩阵乘法，并添加了一个bias向量。中间的向量是逐个组件的半波整流（component-wise half-wave rectified）【Relu（x<- max(0,x)）】 

fc7是网络的最后一层。跟fc6之间通过一个4096×4096的矩阵相乘。也是添加了bias向量和应用了ReLU。 

我们先来看看没有调优的CNN在PASCAL上的表现，没有调优是指所有的CNN参数就是在ILSVRC2012上训练后的状态。分析每一层的性能显示来自fc7的特征泛化能力不如fc6的特征。这意味29%的CNN参数，也就是1680万的参数可以移除掉，而且不影响mAP。更多的惊喜是即使同时移除fc6和fc7，仅仅使用pool5的特征，只使用CNN参数的6%也能有非常好的结果。可见CNN的主要表达力来自于卷积层，而不是全连接层。这个发现提醒我们也许可以在计算一个任意尺寸的图片的稠密特征图（dense feature map）时使仅仅使用CNN的卷积层。这种表示可以直接在pool5的特征上进行滑动窗口检测的实验。 

【**调优后的各层性能**】

我们来看看调优后在VOC2007上的结果表现。提升非常明显，mAP提升了8个百分点，达到了54.2%。fc6和fc7的提升明显优于pool5，这说明pool5从ImageNet学习的特征通用性很强，在它之上层的大部分提升主要是在学习领域相关的非线性分类器。

【**对比其他特征学习方法**】

R-CNN是最好的，我们的mAP要多大约20个百分点，61%的相对提升。

**3.3网络结构**
**3.4 检测错误分析**

两个直接省略！！！

**3.5Bounding-box回归**

基于错误分析，我们使用了一种简单的方法减小定位误差。受到DPM[17]中使用的约束框回归训练启发，我们训练了一个线性回归模型在给定一个选择区域的pool5特征时去预测一个新的检测窗口。详细的细节参考附录C。表1、表2和图4的结果说明这个简单的方法，修复了大量的错位检测，提升了3-4个百分点。

关于BoundingBox-Regression参考下文



**4.结论**

最近几年，物体检测陷入停滞，表现最好的检测系统是复杂的将多低层级的图像特征与高层级的物体检测器环境与场景识别相结合。本文提出了一种简单并且可扩展的物体检测方法，达到了VOC 2012数据集相对之前最好性能的30%的提升。 
我们取得这个性能主要通过两个方面：第一是应用了自底向上的候选框训练的高容量的卷积神经网络进行定位和分割物体。另外一个是使用在标签数据匮乏的情况下训练大规模神经网络的一个方法。我们展示了在有监督的情况下使用丰富的数据集（图片分类）预训练一个网络作为辅助性的工作是很有效的，然后采用稀少数据（检测）去调优定位任务的网络。我们猜测“有监督的预训练+特定领域的调优”这一范式对于数据稀少的视觉问题是很有效的。 
最后,我们注意到能得到这些结果，将计算机视觉中经典的工具和深度学习(自底向上的区域候选框和卷积神经网络）组合是非常重要的。而不是违背科学探索的主线，这两个部分是自然而且必然的结合。




------

###  2.PASCAL  & ILSVRC
 
> Pattern Analysis, Statical Modeling and Computational Learning  Visual Object Classes

[主页]<http://host.robots.ox.ac.uk/pascal/VOC/>

+ Provides standardised image data sets for object class recognition
+ Provides a common set of tools for accessing the data sets and annotations
+ Enables evaluation and comparison of different methods 
+ Ran challenges evaluating performance on object class recognition (from 2005-2012, now finished)

提供了2005-2012年的数据集，数据集的[参考格式]<https://www.cnblogs.com/whlook/p/7220105.html>

<div align=center>
<img src="zh-cn/img/R-CNN/pic_voc.png" /> 
</div>

+ Large Scale Visual Recognition Challenge (ILSVRC)

Stanford Vison Lab

ImageNet比赛

[主页]<http://www.image-net.org/challenges/LSVRC/>

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) evaluates algorithms for object detection and image classification at large scale. One high level motivation is to allow researchers to compare progress in detection across a wider variety of objects -- taking advantage of the quite expensive labeling effort. Another motivation is to measure the progress of computer vision for large scale image indexing for retrieval and annotation.

------

###   3. 目标检测中用到的一些评价指标

模型的好坏是相对的，什么样的模型好不仅取决于数据和算法，还取决于任务需求，因此选取一个合理的模型评价指标非常有必要。

+ IOU 

IOU是由预测的包围盒与地面真相包围盒的重叠区域（交集），除以他们之间的联合区域（并集），gt代表针织框

<div align=center>
<img src="zh-cn/img/R-CNN/pic_IOU.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/pic_IOU1.png" /> 
</div>

+ Precision，Recall,......

一般模型常用的错误率(Error)和精度(accuracy)就能解决(一般的机器学习任务),精度和错误率虽然常用，但不能满足所有需求

<div align=center>
<img src="zh-cn/img/R-CNN/pic_p.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/pic_p2.png" /> 
</div>

其他常用的：
ROC（AUC为ROC曲线下的面积)，P-R曲线，lift曲线，若当值，K-S值（二分类用的多一些），混淆矩阵，F1(F-score, F-Measure $\aplpha=1$ )

基于自己的学习任务，同时也可以修改(比如加一些惩罚)或自定义其他的评价指标。


+ AP & mAP

P: precision

AP: average precision,每一类别P值的平均值

mAP: mean average precision,对所有类别的AP取平均

目标检测中的模型的分类和定位都需要进行评估，每个图像都可能具有不同类别的不同目标。

** 计算mAP的过程**

[Ground Truth的定义]
对于任何算法，度量总是与数据的真实值(Ground Truth)进行比较。我们只知道训练、验证和测试数据集的Ground Truth信息。对于物体检测问题，Ground Truth包括图像，图像中的目标的类别以及图像中每个目标的边界框。 

下图给出了一个真实的图像(JPG/PNG)和其他标注信息作为文本(边界框坐标(X, Y, 宽度和高度)和类)，其中上图的红色框和文本标签仅仅是为了更好的理解，手工标注可视化显示。 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_mpa1.png" /> 
</div>
**标注图像：Ground Truth**

对于上面的例子，我们在模型在训练中得到了下图所示的目标边界框和3组数字定义的ground truth(假设这个图像是1000*800px，所有这些坐标都是构建在像素层面上的) 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map2.png" /> 
</div>
**模型需要预测的：关于详细的一些模型预测label的设定建议学习吴恩达的deeplearning.ai关于卷积网络学习的网课**

开始计算mAP的步骤：

+ 1.假设原始图像和真实的标注信息(ground truth)如上所示，训练和验证数据以相同的方式都进行了标注。该模型将返回大量的预测，但是在这些模型中，大多数都具有非常低的置信度分数，因此我们只考虑高于某个置信度分数的预测信息。我们通过我们的模型运行原始图像，在置信阈值确定之后，下面是目标检测算法返回的带有边框的图像区域(bounding boxes)。 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map3.png" /> 
</div>
**预测结果**

但是怎样在实际中量化这些检测区域的正确性呢？ 
首先我们需要知道每个检测的正确性。测量一个给定的边框的正确性的度量标准是loU-交幷比(检测评价函数)，这是一个非常简单的视觉量。 
下面给出loU的简单的解释。(我们在第一部分已经给出定义)

+  2.IoU计算

loU(交并比)是模型所预测的检测框和真实(ground truth)的检测框的交集和并集之间的比例。这个数据也被称为Jaccard指数。为了得到交集和并集值，我们首先将预测框叠加在ground truth实际框上面，如下图所示： 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map4.png" /> 
</div>

现在对于每个类，预测框和真实框重叠的区域就是交集区域，预测框和真实框的总面积区域就是并集框。 
在上面的目标马的交集和联合看起来是这样的：

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map5.png" /> 
</div>

交集包括重叠区域(青色区域), 并集包括橙色和青色区域


+  3.识别正确的检测和计算精度 

我们使用loU看检测是否正确需要设定一个阈值，最常用的阈值是0.5，即如果loU>0.5，则认为是真实的检测(true detection)，否则认为是错误的检测(false detection)。我们现在计算模型得到的每个检测框的loU值。用计算出的loU值与设定的loU阈值(例如0.5)比较，就可以计算出每个图像中每个类的正确检测次数(A)。对于每个图像，我们都有ground truth的数据(即知道每个图像的真实目标信息),因此也知道了该图像中给定类别的实际目标(B)的数量。因此我们可以使用这个公式来计算该类模型的精度(A/B) 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map6.png" /> 
</div>

即给定一张图像的类别C的Precision=图像正确预测(True Positives)的数量除以在该图像上这一类的总的目标数量。 

假如现在有一个给定的类，验证集中有100个图像，并且我们知道每个图像都有其中的所有类(基于ground truth)。所以我们可以得到100个精度值，计算这100个精度值的平均值，得到的就是该类的平均精度。

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map7.png" /> 
</div>

即一个C类的平均精度=在验证集上所有的图像对于类C的精度值的和/有类C这个目标的所有图像的数量。

+  4.计算最终mAP

现在假如我们整个集合中有20个类，对于每个类别，我们都先计算loU，接下来计算精度,然后计算平均精度。所有我们现在有20个不同的平均精度值。使用这些平均精度值，我们可以轻松的判断任何给定类别的模型的性能。 

但是问题是使用20个不同的平均精度使我们难以度量整个模型，所以我们可以选用一个单一的数字来表示一个模型的表现(一个度量来统一它们),我们可以取所有类的平均精度值的平均值，即mAP(均值平均精度)。

<div align=center>
<img src="zh-cn/img/R-CNN/pic_map8.png" /> 
</div>

MAP=所有类别的平均精度求和除以所有类别 

使用MAP值时我们需要满足一下条件： 
(1) MAP总是在固定的数据集上计算 
(2)它不是量化模型输出的绝对度量，但是是一个比较好的相对度量。当我们在流行的公共数据集上计算这个度量时，这个度量可以很容易的用来比较不同目标检测方法 
(3)根据训练中类的分布情况，平均精度值可能会因为某些类别(具有良好的训练数据)非常高(对于具有较少或较差数据的类别)而言非常低。所以我们需要mAP可能是适中的，但是模型可能对于某些类非常好，对于某些类非常不好。因此建议在分析模型结果的同时查看个各类的平均精度(AP)，这些值也可以作为我们是不是需要添加更多训练样本的一个依据。


------

### 4.各向异性，各向同性缩放

R-CNN的论文中提到了各向同性，各向异性缩放的概念，这里做一个详细解释：

当我们输入一张图片时，我们要搜索出所有可能是物体的区域，R-CNN采用的就是Selective Search方法，通过这个算法我们搜索出2000个候选框。然后从R-CNN的总流程图中可以看到，搜出的候选框是矩形的，而且是大小各不相同。然而CNN对输入图片的大小是有固定的，如果把搜索到的矩形选框不做处理，就扔进CNN中，肯定不行。因此对于每个输入的候选框都需要缩放到固定的大小。

下面我们讲解要怎么进行缩放处理，为了简单起见我们假设下一阶段CNN所需要的输入图片大小是个正方形图片227*227。因为我们经过selective search 得到的是矩形框，paper试验了两种不同的处理方法：

**各向异性缩放：**
这种方法很简单，就是不管图片的长宽比例，管它是否扭曲，进行缩放就是了，全部缩放到CNN输入的大小227*227，如下图(D)所示；

**各向同性缩放：**
因为图片扭曲后，估计会对后续CNN的训练精度有影响，于是作者也测试了“各向同性缩放”方案。有两种办法：

+ 先扩充后裁剪

直接在原始图片中，把bounding box的边界进行扩展延伸成正方形，然后再进行裁剪；如果已经延伸到了原始图片的外边界，那么就用bounding box中的颜色均值填充；如下图(B)所示;

+ 先裁剪后扩充

先把bounding box图片裁剪出来，然后用固定的背景颜色填充成正方形图片(背景颜色也是采用bounding box的像素颜色均值),如下图(C)所示;

对于上面的异性、同性缩放，文献还有个padding处理，上面的示意图中第1、3行就是结合了padding=0, 第2、4行结果图采用padding=16的结果。经过最后的试验，作者发现采用各向异性缩放、padding=16的精度最高。（也就是最后一个图） 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_featureext.png" /> 
</div>


------

### 5.NMS:非极大值抑制

先假设有n个（假设有6个）候选框，根据分类器类别分类概率做排序，从小到大分别属于车辆的概率分别为A<=B<=C<=D<=E<=F。

（1）从最大概率的矩形框开始（F），分别判断A-E与F的IOU是否大于某个设定的阈值

（2）假设B,D与F的IOU超过F,那就扔掉B,D，并标记第一个矩形框F,是我们保留下来的

（3）从剩余矩形框A,C.E中选择概率最大的E，然后判断E与A,C的IOU(重叠度），重叠度大于一定的阈值，那么就扔掉，标记E是我们保留下来的第2个矩形框

（4）一直重复这个过程，找到所有被曾经保留下来的矩形框。

> 为什么需要NMS?

在测试过程完成到第4步之后[section7中的步骤]，获得2000×20维矩阵表示每个建议框是某个物体类别的得分情况，此时会遇到下图所示情况，同一个车辆目标会被多个建议框包围，这时需要非极大值抑制操作去除得分较低的候选框以减少重叠框。

<div align=center>
<img src="zh-cn/img/R-CNN/pic_NMS1.png" /> 
</div>

------

### 6.边框回归：BoundingBox-Regression(BBR)

>首先考虑R-CNN中为什么要做BBR?

Bounding Boxregression是 RCNN中使用的边框回归方法，在RCNN的论文中，作者指出：主要的错误是源于mislocalization。为了解决这个问题，作者使用了bounding box regression。这个方法使得mAp提高了3到4个点。 

>BBR的输入 是什么？

<div align=center>
<img src="zh-cn/img/R-CNN/pic_BBR1.png" /> 
</div>

对于预测框P,我们有一个ground truth是G：当0.1< IoU < 0.5时出现重复，这种情况属于作者说的poor localiazation, 但注意：我们使用的并不是这样的框进行BBR(网上很多地方都在这里出现了误导),作者是用iou>0.6的进行BBR,也就是IOU<0.6的Bounding Box会直接被舍弃，不进行BBR。这样做是为了满足线性转换的条件。否则会导致训练的回归模型不 work.
>（当 P跟 G 离得较远，就是复杂的非线性问题了，此时用线性回归建模显然不合理。)

至于为什么当IoU较大的时候，我们才认为是线性变化，我找到一个觉得解释的比较清楚的，截图在下面： 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_BBR2.png" /> 
</div>

线性回归就是给定输入的特征向量 X, 学习一组参数 W, 使得经过线性回归后的值跟真实值 Y(Ground Truth)非常接近. 即Y≈WX 。

边框回归的目的既是：给定(Px,Py,Pw,Ph)(Px,Py,Pw,Ph)寻找一种映射ff， 使得f(Px,Py,Pw,Ph)=(Gx^,Gy^,Gw^,Gh^)f(Px,Py,Pw,Ph)=(Gx^,Gy^,Gw^,Gh^) 并且(Gx^,Gy^,Gw^,Gh^)≈(Gx,Gy,Gw,Gh)



例如上图：我们现在要讲P框进行bbr,gt为G框，那么我们希望经过变换之后，P框能接近G框（比如，上图的G^框）。现在进行变换,过程如下： 

我们用一个四维向量（x,y,w,h）来表示一个窗口，其中x,y,w,h分别代表框的中心点的坐标以及宽，高。我们要从P得到G^，需要经过平移和缩放。 

<div align=center>
<img src="zh-cn/img/R-CNN/pic_BBR3.png" /> 
</div>

其实这并不是真正的BBR，因为我们只是把P映射回G^,得到一个一般变换的式子，那为什么不映射回最优答案G呢？于是，P映射回G而不是G^，那我们就能得到最优变换（这才是最终的BBR）：

<div align=center>
<img src="zh-cn/img/R-CNN/pic_BBR4.png" /> 
</div>

> 这里为什么会将tw,th写成exp形式？ 

是因为tw,th代表着缩放的尺寸，这个尺寸是>0的，所以使用exp的形式正好满足这种约束。 
也就是，我们将转换d换成转换t,就得到了P到G的映射。 di -> ti。 
现在我们只需要学习 这四个变换dx(P),dy(P),dw(P),dh(P)，然后最小化t和d之间的距离，最小化这个loss，即可。

注意：此时看起来我们只要输入P的四维向量，就可以学习,然后求出，但是，其实我们输入的是pool5之后的features，记做φ5，因为如果只是单纯的靠坐标回归的话，CNN根本就没有发挥任何作用，但其实，bb的位置应该有CNN计算得到的features来fine-tune。所以，我们选择将pool5的feature作为输入。 


<div align=center>
<img src="zh-cn/img/R-CNN/pic_BBR5.png" /> 
</div>

loss为：

<div align=center>
<img src="zh-cn/img/R-CNN/pic_BBR6.png" /> 
</div>

最后，我们只需要利用梯度下降或最小二乘求解w即可。
另外不要认为BBR和分类信息没有什么关系，是针对每一类都会训练一个BBR


------

### 7.R-CNN测试的一般步骤

+  1.输入一张多目标图像，采用selective search算法提取约2000个建议框；

+  2.先在每个建议框周围加上16个像素值为建议框像素平均值的边框，再直接变形为227×227的大小；

+ 3.先将所有建议框像素减去该建议框像素平均值后【预处理操作】，再依次将每个227×227的建议框输入AlexNet CNN网络获取4096维的特征【比以前的人工经验特征低两个数量级】，2000个建议框的CNN特征组合成2000×4096维矩阵；

+ 4.将2000×4096维特征与20个SVM组成的权值矩阵4096×20相乘【20种分类，SVM是二分类器，则有20个SVM】，获得2000×20维矩阵表示每个建议框是某个物体类别的得分；

+ 5.分别对上述2000×20维矩阵中每一列即每一类进行非极大值抑制剔除重叠建议框，得到该列即该类中得分最高的一些建议框；

+ 6.分别用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。

------

### 8.R-CNN的训练过程

+  1.有监督预训练

<div align=center>
<img src="zh-cn/img/R-CNN/pic_train1.png" /> 
</div>

ILSVRC样本集上仅有图像类别标签，没有图像物体位置标注； 
采用AlexNet CNN网络进行有监督预训练，学习率=0.01； 
该网络输入为227×227的ILSVRC训练集图像，输出最后一层为4096维特征->1000类的映射，训练的是网络参数。

+ 2.特定样本下的微调

<div align=center>
<img src="zh-cn/img/R-CNN/pic_train2.png" /> 
</div>

PASCAL VOC 2007样本集上既有图像中物体类别标签，也有图像中物体位置标签； 
采用训练好的AlexNet CNN网络进行PASCAL VOC 2007样本集下的微调，学习率=0.001【0.01/10为了在学习新东西时不至于忘记之前的记忆】； 
mini-batch为32个正样本和96个负样本【由于正样本太少】； 
该网络输入为建议框【由selective search而来】变形后的227×227的图像，修改了原来的1000为类别输出，改为21维【20类+背景】输出，训练的是网络参数。


+ 3.SVM训练

<div align=center>
<img src="zh-cn/img/R-CNN/pic_train3.png" /> 
</div>

由于SVM是二分类器，需要为每个类别训练单独的SVM； 
SVM训练时输入正负样本在AlexNet CNN网络计算下的4096维特征，输出为该类的得分，训练的是SVM权重向量； 
由于负样本太多，采用hard negative mining的方法在负样本中选取有代表性的负样本，该方法具体见

+ 4.Bounding-box regression训练

<div align=center>
<img src="zh-cn/img/R-CNN/pic_train4.png" /> 
</div>


结果怎么样?

PASCAL VOC 2010测试集上实现了53.7%的mAP；

PASCAL VOC 2012测试集上实现了53.3%的mAP；

计算Region Proposals和features平均所花时间：13s/image on a GPU；53s/image on a CPU


还存在什么问题?

很明显，最大的缺点是对一张图片的处理速度慢，这是由于一张图片中由selective search算法得出的约2k个建议框都需要经过变形处理后由CNN前向网络计算一次特征，这其中涵盖了对一张图片中多个重复区域的重复计算，很累赘；

知乎上有人说R-CNN网络需要两次CNN前向计算，第一次得到建议框特征给SVM分类识别，第二次对非极大值抑制后的建议框再次进行CNN前向计算获得Pool5特征，以便对建议框进行回归得到更精确的bounding-box，这里文中并没有说是怎么做的，个人认为也可能在计算2k个建议框的CNN特征时，在硬盘上保留了2k个建议框的Pool5特征，虽然这样做只需要一次CNN前向网络运算，但是耗费大量磁盘空间；

训练时间长，虽然文中没有明确指出具体训练时间，但由于采用RoI-centric sampling【从所有图片的所有建议框中均匀取样】进行训练，那么每次都需要计算不同图片中不同建议框CNN特征，无法共享同一张图的CNN特征，训练速度很慢；

整个测试过程很复杂，要先提取建议框，之后提取每个建议框CNN特征，再用SVM分类，做非极大值抑制，最后做bounding-box回归才能得到图片中物体的种类以及位置信息；同样训练过程也很复杂，ILSVRC 2012上预训练CNN，PASCAL VOC 2007上微调CNN，做20类SVM分类器的训练和20类bounding-box回归器的训练；这些不连续过程必然涉及到特征存储、浪费磁盘空间等问题。

------

### 9. HOG(Histogram of Oriented Gradient)
方向梯度直方图特征是一种在计算机视觉和图像处理中用来进行物体检测的特征描述子。它通过计算和统计图像局部区域的梯度方向直方图来构成特征。Hog特征结合SVM分类器已经被广泛应用于图像识别中，尤其在行人检测中获得了极大的成功。需要提醒的是，HOG+SVM进行行人检测的方法是法国研究人员Dalal在2005的CVPR上提出的，而如今虽然有很多行人检测算法不断提出，但基本都是以HOG+SVM的思路为主。

其思想是 在一副图像中，局部目标的表象和形状（appearance and shape）能够被梯度或边缘的方向密度分布很好地描述。（本质：梯度的统计信息，而梯度主要存在于边缘的地方）

**梯度的概念：**

在图像中梯度的概念也是像素值变换最快的方向，把边缘（在图像合成中单一物体的轮廓叫做边缘）引入进来，边缘与梯度保持垂直方向。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p1.png" /> 
</div>

具体在HOG中方向梯度的实现：首先用[-1,0,1]梯度算子对原图像做卷积运算，得到x方向（水平方向，以向右为正方向）的梯度分量gradscalx，然后用[1,0,-1]T梯度算子对原图像做卷积运算，得到y方向（竖直方向，以向上为正方向）的梯度分量gradscaly。然后再用以下公式计算该像素点的梯度大小和方向。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p2.jpg" /> 
</div>

**直方图**

这个就不做解释了！！！

**方向梯度直方图HOG的提取**

方向梯度直方图为图像局部区域的梯度特征量统计，我们为什么要提取这个东东呢？

HOG主要应用于行人检测方面，以行人照片为例。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p3.png" /> 
</div>

上图是一张行人图的四种表示方式，原三色图，灰度图，边缘图，梯度图，人脑根据前期学习与先验知识很容易理解到图像中包含着一个行人，并可以根据一定情况将其从图像中抠选出来，但计算机是怎么思考的呢？怎样让计算机理解以上图像中包含的是一个行人呢？前三个图像现在情况不适用，所以选取梯度图，现在的梯度图同样也是人脑处理理解的平面结果，计算机是办不到的，需要将直观地的梯度图像转换成一种计算机容易理解的数据特征语言。

对于64X128的图像而言，每8X8的像素组成一个cell，每2X2个cell组成一个块(block)，以8个像素为步长，那么，水平方向将有7个扫描窗口，垂直方向将有15个扫描窗口。也就是说，64X128的图片，总共有36X7X15=3780个特征。这里截取梯度图的一部分画图进行理解，尺寸与比例并不精确。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p4.png" /> 
</div>

单独将其中一个8X8的小格拿出来，方向梯度中指的方向范围为2π，360°，为了画直方图我们还需要选取合适的组距也就是bin，这里组距选取2π/9，也就是最后的直方图组数为9。下图为8X8像素的cell对应的方向梯度（未全部画出，共有8X8=64个）。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p5.png" /> 
</div>

　将上面的64个方向梯度，按着直方图的参数设置进行画图，其中梯度的大小在统计数量中呈线性关系，比如梯度大小为2，则直方图对应增加2个单位，
画出的对应直方图假设如下所示：

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p6.png" /> 
</div>

把上图中单个cell对应的方向直方图转换为单维向量，也就是按规定组距对对应方向梯度个数进行编码，（8,10,6,12,4,5,8,6,14），得到单个cell的9个特征，每个block（扫描窗口）包含2X2个cell也就是2X2X9=36个特征，一个64X128大小的图像最后得到的特征数为36X7X15=3780个。这样将一幅直观的梯度图通过分解提取变为计算机容易理解的特征向量。
　　以上工作为HOG提取的主要内容，最后得到对应的行人的由方向梯度直方图HOG提取到的特征向量，但是计算机还是不知道这个数据数组代表了什么意思，什么时候这组向量代表行人，什么时候代表其他东西，怎样train，最后通过不断地学习，而后在检测积累的基础上对对未知图像检测识别有没有行人呢？那就是后一步SVM要做的事了。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p7.jpg" /> 
</div>
------

### 10.LBP（Local Binary Pattern)

LBP（Local Binary Pattern，局部二值模式）是一种用来描述图像局部纹理特征的算子；它具有旋转不变性和灰度不变性等显著的优点。它是首先由T. Ojala, M.Pietikäinen, 和D. Harwood 在1994年提出，用于纹理特征提取。而且，提取的特征是图像的局部的纹理特征；

**LBP特征的描述**

原始的LBP算子定义为在3X3的窗口内，以窗口中心像素为阈值，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3X3邻域内的8个点经比较可产生8位二进制数（通常转换为十进制数即LBP码，共256种），即得到该窗口中心像素点的LBP值，并用这个值来反映该区域的纹理信息。如下图所示：

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p8.jpg" /> 
</div>

**LBP的改进版本**

原始的LBP提出后，研究人员不断对其提出了各种改进和优化。

（1）圆形LBP算子

基本的 LBP算子的最大缺陷在于它只覆盖了一个固定半径范围内的小区域，这显然不能满足不同尺寸和频率纹理的需要。为了适应不同尺度的纹理特征，并达到灰度和旋转不变性的要求，Ojala等对 LBP 算子进行了改进，将 3×3邻域扩展到任意邻域，并用圆形邻域代替了正方形邻域，改进后的 LBP 算子允许在半径为 R 的圆形邻域内有任意多个像素点。从而得到了诸如半径为R的圆形区域内含有P个采样点的LBP算子；

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p9.jpg" /> 
</div>

（2）LBP旋转不变模式

从 LBP 的定义可以看出，LBP 算子是灰度不变的，但却不是旋转不变的。图像的旋转就会得到不同的 LBP值。

Maenpaa等人又将 LBP算子进行了扩展，提出了具有旋转不变性的 LBP 算子，即不断旋转圆形邻域得到一系列初始定义的 LBP值，取其最小值作为该邻域的 LBP 值。

下图 给出了求取旋转不变的 LBP 的过程示意图，图中算子下方的数字表示该算子对应的 LBP值，图中所示的 8 种 LBP模式，经过旋转不变的处理，最终得到的具有旋转不变性的 LBP值为 15。也就是说，图中的 8种 LBP 模式对应的旋转不变的 LBP模式都是00001111。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p10.jpg" /> 
</div>

一个LBP算子可以产生不同的二进制模式，对于半径为R的圆形区域内含有P个采样点的LBP算子将会产生P2种模式。很显然，随着邻域集内采样点数的增加，二进制模式的种类是急剧增加的。例如：5×5邻域内20个采样点，有220＝1,048,576种二进制模式。如此多的二值模式无论对于纹理的提取还是对于纹理的识别、分类及信息的存取都是不利的。同时，过多的模式种类对于纹理的表达是不利的。例如，将LBP算子用于纹理分类或人脸识别时，常采用LBP模式的统计直方图来表达图像的信息，而较多的模式种类将使得数据量过大，且直方图过于稀疏。因此，需要对原始的LBP模式进行降维，使得数据量减少的情况下能最好的代表图像的信息。

为了解决二进制模式过多的问题，提高统计性，Ojala提出了采用一种“等价模式”（Uniform Pattern）来对LBP算子的模式种类进行降维。Ojala等认为，在实际图像中，绝大多数LBP模式最多只包含两次从1到0或从0到1的跳变。因此，Ojala将“等价模式”定义为：当某个LBP所对应的循环二进制数从0到1或从1到0最多有两次跳变时，该LBP所对应的二进制就称为一个等价模式类。如00000000（0次跳变），00000111（只含一次从0到1的跳变），10001111（先由1跳到0，再由0跳到1，共两次跳变）都是等价模式类。除等价模式类以外的模式都归为另一类，称为混合模式类，例如10010111（共四次跳变）

通过这样的改进，二进制模式的种类大大减少，而不会丢失任何信息。模式数量由原来的2P种减少为 P ( P-1)+2种，其中P表示邻域集内的采样点数。对于3×3邻域内8个采样点来说，二进制模式由原始的256种减少为58种，这使得特征向量的维数更少，并且可以减少高频噪声带来的影响。

**LBP特征用于检测的原理**

显而易见的是，上述提取的LBP算子在每个像素点都可以得到一个LBP“编码”，那么，对一幅图像（记录的是每个像素点的灰度值）提取其原始的LBP算子之后，得到的原始LBP特征依然是“一幅图片”（记录的是每个像素点的LBP值）。


<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p11.jpg" /> 
</div>

LBP的应用中，如纹理分类、人脸分析等，一般都不将LBP图谱作为特征向量用于分类识别，而是采用LBP特征谱的统计直方图作为特征向量用于分类识别。

因为，从上面的分析我们可以看出，这个“特征”跟位置信息是紧密相关的。直接对两幅图片提取这种“特征”，并进行判别分析的话，会因为“位置没有对准”而产生很大的误差。后来，研究人员发现，可以将一幅图片划分为若干的子区域，对每个子区域内的每个像素点都提取LBP特征，然后，在每个子区域内建立LBP特征的统计直方图。如此一来，每个子区域，就可以用一个统计直方图来进行描述；整个图片就由若干个统计直方图组成；

例如：一幅100X100像素大小的图片，划分为10X10=100个子区域（可以通过多种方式来划分区域），每个子区域的大小为10X10像素；在每个子区域内的每个像素点，提取其LBP特征，然后，建立统计直方图；这样，这幅图片就有10X10个子区域，也就有了10X10个统计直方图，利用这10X10个统计直方图，就可以描述这幅图片了。之后，我们利用各种相似性度量函数，就可以判断两幅图像之间的相似性了；

**对LBP特征向量进行提取的步骤**

（1）首先将检测窗口划分为16×16的小区域（cell）；

（2）对于每个cell中的一个像素，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于中心像素值，则该像素点的位置被标记为1，否则为0。这样，3X3邻域内的8个点经比较可产生8位二进制数，即得到该窗口中心像素点的LBP值；

（3）然后计算每个cell的直方图，即每个数字（假定是十进制数LBP值）出现的频率；然后对该直方图进行归一化处理。

（4）最后将得到的每个cell的统计直方图进行连接成为一个特征向量，也就是整幅图的LBP纹理特征向量；

然后便可利用SVM或者其他机器学习算法进行分类了。

------

### 11.Haar特征

积分图就是只遍历一次图像就可以求出图像中所有区域像素和的快速算法，大大的提高了图像特征值计算的效率。

积分图主要的思想是将图像从起点开始到各个点所形成的矩形区域像素之和作为一个数组的元素保存在内存中，当要计算某个区域的像素和时可以直接索引数组的元素，不用重新计算这个区域的像素和，从而加快了计算（这有个相应的称呼，叫做动态规划算法）。积分图能够在多种尺度下，使用相同的时间（常数时间）来计算不同的特征，因此大大提高了检测速度。

我们来看看它是怎么做到的。

积分图是一种能够描述全局信息的矩阵表示方法。积分图的构造方式是位置（i,j）处的值ii(i,j)是原图像(i,j)左上角方向所有像素的和：

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p12.jpg" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p13.jpg" /> 
</div>

积分图构建算法：

1）用s(i,j)表示行方向的累加和，初始化s(i,-1)=0;

2）用ii(i,j)表示一个积分图像，初始化ii(-1,i)=0；

3）逐行扫描图像，递归计算每个像素(i,j)行方向的累加和s(i,j)和积分图像ii(i,j)的值

s(i,j)=s(i,j-1)+f(i,j)

ii(i,j)=ii(i-1,j)+s(i,j)

4）扫描图像一遍，当到达图像右下角像素时，积分图像ii就构造好了。

积分图构造好之后，图像中任何矩阵区域的像素累加和都可以通过简单运算得到如图所示。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p14.jpg" /> 
</div>

设D的四个顶点分别为α、β、γ、δ，则D的像素和可以表示为

Dsum = ii( α )+ii( β)-(ii( γ)+ii( δ ));

而Haar-like特征值无非就是两个矩阵像素和的差，同样可以在常数时间内完成。所以矩形特征的特征值计算，只与此特征矩形的端点的积分图有关，所以不管此特征矩形的尺度变换如何，特征值的计算所消耗的时间都是常量。这样只要遍历图像一次，就可以求得所有子窗口的特征值。

**Haar-like矩形特征拓展**

Lienhart R．等对Haar-like矩形特征库作了进一步扩展，加入了旋转45。角的矩形特征。扩展后的特征大致分为4种类型：边缘特征、线特征环、中心环绕特征和对角线特征：

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p15.jpg" /> 
</div>

在特征值的计算过程中，黑色区域的权值为负值，白色区域的权值为正值。而且权值与矩形面积成反比（使两种矩形区域中像素数目一致）；

竖直矩阵特征值计算：

对于竖直矩阵，与上面2处说的一样。

45°旋角的矩形特征计算：

对于45°旋角的矩形，我们定义RSAT(x,y)为点(x,y)左上角45°区域和左下角45°区域的像素和。

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p16.jpg" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p17.jpg" /> 
</div>

为了节约时间，减少重复计算，可按如下递推公式计算：

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p18.jpg" /> 
</div>

而计算矩阵特征的特征值，是位于十字行矩形RSAT(x,y)之差。可参考下图：

<div align=center>
<img src="zh-cn/img/R-CNN/HOG/p19.jpg" /> 
</div>



------

### 12.SIFT(Scale Invariant Feature Transform)

1999年British Columbia大学大卫.劳伊教授总结了现有的基于不变量技术的检测方法，并正式的提出一种基于尺度空间的，对图像缩放，旋转，甚至放射变换保持不变形的图像局部特征描述算子-SIFT(尺度不变特征变换),这种算法在2004年被加以完善。

**SIFT简介**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/1.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/2.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/3.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/4.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/5.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/6.png" /> 
</div>

**SIFT算法实现细节**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/7.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/8.png" /> 
</div>

**关键点检测的相关概念**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/9.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/10.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/11.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/12.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/13.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/14.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/15.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/16.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/17.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/18.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/19.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/20.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/21.png" /> 
</div>

**关键点加测--DOG**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/22.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/23.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/24.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/25.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/26.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/27.png" /> 
</div>

**DOG局部极值检测**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/28.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/29.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/30.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/31.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/32.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/33.png" /> 
</div>

**关键点方向分配**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/34.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/35.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/36.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/37.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/38.png" /> 
</div>

**关键点描述**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/39.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/40.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/41.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/42.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/43.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/44.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/45.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/46.png" /> 
</div>

**关键点匹配**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/47.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/48.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/49.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/50.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/51.png" /> 
</div>

**消除错配点**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/52.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/53.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/54.png" /> 
</div>

**SIFT算法的应用**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/55.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/56.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/57.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/58.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/59.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/60.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/61.png" /> 
</div>

**SIFT算法的宽展与改进**

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/62.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/63.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/64.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/65.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/66.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/SIFT/67.png" /> 
</div>


------
### 13.BOW(bag-of-words)


**bag-of-words简介**

Bag-of-words是信息检索领域常用的文档表示方法。在信息检索中，BOW模型假定对于一个文档，忽略他的单词顺序和语法、句法等要素。将其仅仅看做是若干词汇的集合，文档中每个单词的出现都是独立的，不依赖于其他单词是否出现。也就是说文档中任意一个位置的出现的任何单词，都不受该文档语义影响而独立选择。

例如：

```
1. Bob likes to play basketball, Jim likes too.
2. Bob also likes to play football games.
```

基于文档构建词表(对中文来说要先分词)

```
Vocabulary = {
	1:'Bob',2:'like',3:'to',4:'play',5:'basketball',6:'also',7:'football',8:'games',9:'Jim',10:'too'
}
```

构建10个不同的单词，利用词的索引号，上面两个文档用一个10维的向量表示：

```
D: [1,2,3,4,5,6,7,8,9,10]

1. [1,2,1,1,1,0,0,0,1,1]
2. [1,1,1,1,0,1,1,1,0,0]
```

缺点：

+ 维数灾难

+ 无法保留次序信息

+ 存在语义鸿沟的问题

+ 不包含任何语义信息

以上向量也可以用直方图表示，词相当于直方图的箱的取值，新来的文档可以映射到直方图上。

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p1.png" /> 
</div>

并不是所有的词都用来建立词表

+ 相似的词： walking,walk,walks统一用walk。(词的聚类)(中文不存在)

+ 停用词： a，an,the,了，的等

+ 高频词 TF-IDF思想

BOW实现步骤：

1. 词汇表的建立： 聚类找类中心-vocabulary

2. 样本训练： 对每个文档进行训练，得到每个文档的低位表示

3. 新样本的识别： 词表单词到低维表示，到分类器预测

**bag-of-visual-words(视觉词袋模型)**

BOW应用在图像领域，可以构建视觉词袋模型。为了表示图像，可以将图像看做文档，即若干个‘视觉单词’的集合，同样的，视觉单词想回见没有顺序。

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p2.png" /> 
</div>

由于图像中的单词不像文本文档中的那样是分词后得到或现成的，我们首先需要从图像中提取出相互独立的视觉单词，这通常需要三个步骤：

+ 特征检测

+ 特征表示

+ 词汇表的生成

SIFT算法（上节已经介绍）是提取图像中局部不变特征的应用最广泛的算法，因此可以使用SIFT算法从图像中提取不变特征点，作为视觉单词，并构造词汇表，用词汇表中的单词表示一幅图像

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p3.png" /> 
</div>


**bag-of-visual-words模型建立步骤**

1.利用SIFT算法，从每类图像中提取视觉单词，将所有的视觉单词集合在一起

以SIFT 128维为例，现有3张训练图片，对每一张训练图片都提取SIFT的128维特征，那么做种可以得到M=N1+N2+N3个128维的特征，Ni代表第i张图特征点的个数

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p4.png" /> 
</div>


2.利用k-means算法构造词库表-vocabulary

SIFT提取的视觉单词向量，根据距离的远近，可以利用k-means算法将词意相近的词汇合并，作为词汇表中的基础词汇，假设我们将k设为4，那么词汇表的构造如下：

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p5.png" /> 
</div>

经过聚类词汇表中的单词数为4个（这里只是举例子，k是超参数,一百在几百上千）


3.利用视觉词袋量化图像特征，利用词频表示图像

利用SIFT算法，可以从每张图像中提取很多特征点，这些特征点都可以用词汇表中的单词近似替代，通过统计词汇表中每个单词在图像中出现的次数，可以将图像表示为一个k=4维的特征向量：

```
人  脸：[3,30,3,20]
自行车：[20,3,3,3]
吉  他：[8,12,32,7]
```

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p6.png" /> 
</div>

总结过程如下：

针对‘人脸、自行车、吉他‘这三个文档，抽取出一部分特征，构造一个词表，此表中包含4个视觉单词：

<div align=center>
<img src="zh-cn/img/R-CNN/BOW/p7.png" /> 
</div>

最终’人脸，自行车，吉他’这三个文档皆可以用一个4维向量表示，最后根据三个文档想用部分出现的次数画成对应的直方图。

bag-of-visual-words模型建好后，对于新图片，同样：

1.先提取SIFT特征

2.用词表中的单词将图像表示成数值向量直方图

3.通过分类器进行分类（SVM），看他属于哪一类图片

BOW识别率并不是很高60%-80%之间，一方面是数据量巨大的问题，另一方面是因为图像之间的相似度是很大的。

------

传统的的图像特征提取办法我们就介绍到这里，这样我们就可以很自如的阅读下面章节介绍的一些图像分类和目标检测的论文了！

------

### 14.Efficient Graph-Based Image Segmentation(基于图的图像分割)

该算法是基于图的贪心聚类算法，实现简单，速度比较快，精度也还行。不过，目前直接用它做分割的应该比较少。

**图的概念**

因为该算法是将照片用加权图抽象化表示，所以补充图的一些基本概念。

图：是由顶点集 v（vertices）和边集 E（edges）组成，表示为G=(V,E)，在本文中即为单个的像素点，连接一对顶点的边(vi,vj)具有权重w(vi,vj)，本文中的意义为顶点之间的*不相似度**，所用的是无向图。

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p1.png" /> 
</div>

树：特殊的图，图中任意两个顶点，都有路径相连接，但是没有回路。如上图中加粗的边所连接而成的图。如果看成一团乱连的珠子，只保留树中的珠子和连线，那么随便选个珠子，都能把这棵树中所有的珠子都提起来。如果，i和h这条边也保留下来，那么h,I,c,f,g就构成了一个回路。


最小生成树（MST, minimum spanning tree）：特殊的树，给定需要连接的顶点，选择边权之和最小的树。上图即是一棵MST

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p2.png" /> 
</div>

本文中，初始化时每一个像素点都是一个顶点，然后逐渐合并得到一个区域，确切地说是连接这个区域中的像素点的一个MST。如图，棕色圆圈为顶点，线段为边，合并棕色顶点所生成的MST，对应的就是一个分割区域。分割后的结果其实就是森林。

**相似性**

其实就是聚类算法，那应该依据何种规则判定何时该合二为一，何时该继续划清界限呢？

对于孤立的两个像素点，所不同的是颜色，自然就用颜色的距离来衡量两点的相似性，本文中是使用RGB的距离，即

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p3.png" /> 
</div>

当然也可以选择其他色彩空间的距离计算或其他的距离定义方法。

**形状相似**

前面提到的用颜色信息来聚类，修改相似性衡量标准，可以聚类成我们想要的特定形状。比如我们希望得到很多长条形的区域，那么可以用聚类后的所形成的区域的 **面积/周长 + 亮度值的差** 衡量两个子图或者两个像素之间的相似度。因为长条形的面积/周长会比较小。

**全局阈值到自适应阈值**

上面提到应该用亮度值之差来衡量两个像素点之间的差异性。对于两个区域（子图）或者一个区域和一个像素点的相似性，最简单的方法即只考虑连接二者的边的不相似度。

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p4.png" /> 
</div>

如图，已经形成了棕色和绿色两个区域，现在通过紫色边来判断这两个区域是否合并。那么我们就可以设定一个阈值，当两个像素之间的差异（即不相似度）小于该值时，合二为一。迭代合并，最终就会合并成一个个区域，这就是区域生长的基本思想：星星之火，可以燎原。

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p5.png" /> 
</div>

显然，上面这张图应该聚成右图所思的3类，高频区h,斜坡区s,平坦区p。如果我们设置一个全局阈值，那么如果h区要合并成一块的话，那么该阈值要选很大，但是那样就会把p和s区域也包含进来，分割结果太粗。如果以p为参考，那么阈值应该选特别小的值，那样的话，p区是会合并成一块，但是，h区就会合并成特别特别多的小块，如同一面支离破碎的镜子，分割结果太细。

显然，全局阈值并不合适，那么自然就得用自适应阈值。对于p区该阈值要特别小，s区稍大，h区巨大。

对于两个区域（原文中叫Component,实质上是一个MST,单独的一个像素点也可以看成一个区域）,本文使用了非常直观，但抗干扰性并不强的方法。先来两个定义，原文依据这两个附加信息来得到自适应阈值。


一个区域的类内差异Int(C):
<div align=center>
<img src="zh-cn/img/R-CNN/seg/p6.png" /> 
</div>

可以近似理解为一个区域内部最大的亮度差异值，定义是MST中不相似度最大的一条边。

两个区域的类间差异Diff(C1,C2)
<div align=center>
<img src="zh-cn/img/R-CNN/seg/p7.png" /> 
</div>

即连接两个区域所有边中，不相似度最小的边的不相似度，也就是两个区域最相似的地方的不相似度。

那么直观的判断是否合并的标准
<div align=center>
<img src="zh-cn/img/R-CNN/seg/p8.png" /> 
</div>

等价条件：

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p9.png" /> 
</div>

这个和聚类的思想一样一样的！

 特殊情况，当二者都是孤立的像素值时Int(C1)=0，所有像素都是"零容忍"只有像素值完全一样才能合并，自然会导致过分割。所以刚开始的时候，应该给每个像素点设定一个可以容忍的范围，当生长到一定程度时，就应该去掉该初始容忍值的作用。原文条件如下

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p10.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p11.png" /> 
</div>

其中|C|为区域C所包含的像素点的个数，如此，随着区域逐渐扩大，这一项的作用就越来越小，最后几乎可以忽略不计。那么k就是一个可以控制所形成的的区域的大小，如果k=0，那么，几乎每个像素都成为了一个独立的区域，如果k是无穷大，显然整张图片都会聚成一块。所以，k越大，分割后的图片也就越大。


**算法步骤**

+ Step 1: 计算每一个像素点与其8邻域或4邻域的不相似度。

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p12.png" /> 
</div>

如左边所示，实线为只计算4领域，加上虚线就是计算8邻域，由于是无向图，按照从左到右，从上到下的顺序计算的话，只需要计算右图中灰色的线即可。

+ Step 2: 将边按照不相似度non-decreasing排列（从小到大）排序得到e1,e2,...,eN

+ Step 3: 选择e1

+ Step 4: 对当前选择的边en进行合并判断，假设所链接的顶点为(vi,vj)。如果满足合并条件：

    - (1)vi,vj不属于同一个区域：

    <div align=center>
    <img src="zh-cn/img/R-CNN/seg/p14.png" /> 
    </div>

    - (2)不相似度不大于二者内部的不相似度，则执行Step4。否则执行Step5.

+ Step 5: 更新阈值及类标号。
更新类标号：将Id(vi)，Id(vj)的类标号统一为Id(vi)的标号。更新该类的不相似度的阈值为：

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p15.png" /> 
</div>

注意：由于不相似度小的边先合并，所以，wij即为当前合并后的区域的最大的边，即

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p16.png" /> 
</div>

+ Step 6: 如果n<=N,则按照排好的顺序，选择下一条边执行Step 4，否则结束

**结果展示**

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p17.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p18.png" /> 
</div>



Segmentation parameters: | sigma | k| min 
----|----|----|----
|0.5|500|50

+ Sigma：先对原图像进行高斯滤波去噪，sigma即为高斯核的标准差
+ k: 控制合并后的区域的大小
+ min: 后处理参数，分割后会有很多小区域，当区域Ci像素点的个数|Ci|小于min时，选择与其差异最小的区域Cj合并即

<div align=center>
<img src="zh-cn/img/R-CNN/seg/p19.png" /> 
</div>

------

### 15.Selective Search for Object Regognition（论文解读）

物体识别，在之前的做法主要是基于穷举搜索（Exhaustive Search）：选择一个窗口扫描整张图像（image），改变窗口的大小，继续扫描整张图像。这种做法是比较原始直观，改变窗口大小，扫描整张图像，非常耗时。若能过滤掉一些无用的box将会节省大量时间。这就是本文中Selective Search(选择性搜索)的优点。

选择性搜索（Selective Search)综合了穷举搜索（Exhausticve Search)和分割（Segmentation)的方法，意在找到一些可能的目标位置集合。作者将穷举搜索和分割结合起来，采取组合策略保证搜索的多样性，其结果达到平均最好重合率为0.879。能够大幅度降低搜索空间，提高程序效率，减小计算量。

**基础介绍**

图像（Image）包含的信息非常的丰富，其中的物体（Object）有不同的形状（shape）、尺寸（scale）、颜色（color）、纹理（texture），要想从图像中识别出一个物体非常的难，还要找到物体在图像中的位置，这样就更难了。下图给出了四个例子，来说明物体识别（Object Recognition）的复杂性以及难度。

（a）中的场景是一张桌子，桌子上面放了碗，瓶子，还有其他餐具等等。比如要识别“桌子”，我们可能只是指桌子本身，也可能包含其上面的其他物体。这里显示出了图像中不同物体之间是有一定的层次关系的。

（b）中给出了两只猫，可以通过纹理（texture）来找到这两只猫，却又需要通过颜色（color）来区分它们。

（c）中变色龙和周边颜色接近，可以通过纹理（texture）来区分。

（d）中的车辆，我们很容易把车身和车轮看做一个整体，但它们两者之间在纹理（texture）和颜色（color）方面差别都非常地大。

<div align=center>
<img src="zh-cn/img/R-CNN/pic_SS1.png" /> 
</div>

 上面简单说明了一下在做物体识别（Object Recognition）过程中，不能通过单一的策略来区分不同的物体，需要充分考虑图像物体的多样性（diversity）。另外，在图像中物体的布局有一定的层次（hierarchical）关系，考虑这种关系才能够更好地对物体的类别（category）进行区分。

 在深入介绍Selective Search之前，先说说其需要考虑的几个问题：

 1.适应不同尺度（Capture All Scales）：穷举搜索（Exhaustive Selective）通过改变窗口大小来适应物体的不同尺度，选择搜索（Selective Search）同样无法避免这个问题。算法采用了图像分割（Image Segmentation）以及使用一种层次算法（Hierarchical Algorithm）有效地解决了这个问题。

2.多样化（Diversification）：单一的策略无法应对多种类别的图像。使用颜色（color）、纹理（texture）、大小（size）等多种策略对（【1】中分割好的）区域（region）进行合并。

3.速度快（Fast to Compute）：算法，就像功夫一样，唯快不破！

**区域合并算法**


<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p1.jpg" /> 
</div>

**输入:** 彩色图片

**输出:**物体位置的可能结果L

1. 使用Efficient Graph-Based Image Segmentation的方法获取原始分割区域R={r1,r2,...,rn}
2. 初始化相似度集合S为空集
3. 计算两两相邻区域之间的相似度（见下一部分），将其添加到相似度集合S中
4. 从相似度集合S中找出，相似度最大的两个区域ri和rj，将其合并成为一个区域rt,从相似度集合中出去原先与ri和rj相邻区域之间计算的相似度，计算rt与相邻区域（原先与ri或rj相邻的区域）的相似度，将其结果添加到相似度集合S中，同时将新区域rt添加到区域集合R中。
5. 获取每个区域的Bounding Boxes，这个结果就是物体位置的可能结果L

**多样化策略**

论文中作者给出了两个方面的多样化策略：颜色空间多样化，相似多样化

*颜色空间多样化*

作者采用了8中不同的颜色方式，主要是为了考虑场景以及光照条件等。
主要使用的颜色空间有：（1）RGB，（2）灰度I，（3）Lab，（4）rgI（归一化的rg通道加上灰度），（5）HSV，（6）rgb（归一化的RGB），（7）C，（8）H（HSV的H通道）

*相似度计算多样化*

在区域合并的时候有说道计算区域之间的相似度，论文章介绍了四种相似度的计算方法。

1.颜色（color）相似度

使用L1-norm归一化获取图像每个颜色通道的25 bins的直方图，这样每个区域都可以得到一个75维的向量Ci={ci1,...,cin},区域之间颜色相似度通过下面的公式计算：

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p2.jpg" /> 
</div>

在区域合并过程中使用需要对新的区域进行计算其直方图，计算方法：


<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p3.jpg" /> 
</div>

2.纹理（texture）相似度

这里的纹理采用SIFT特征。具体做法是对每个颜色通道的8个不同方向计算方差σ=1的高斯微分（Gaussian Derivative），每个通道每个方向获取10 bins的直方图（L1-norm归一化），这样就可以获取到一个240维的向量Ti={ti1,...,tin}.区域之间纹理相似度计算方式和颜色相似度计算方式类似，合并之后新区域的纹理特征计算方式和颜色特征计算相同：

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p4.jpg" /> 
</div>


3.大小（size）相似度

这里的大小是指区域中包含像素点的个数。使用大小的相似度计算，主要是为了尽量让小的区域先合并：

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p5.jpg" /> 
</div>

其中size（im）表示图像的像素大小。

4.吻合（fit）相似度

这里主要是为了衡量两个区域是否更加“吻合”，检查两个区域间的重合度，如果一个区域包含另一个区域，逻辑上应该合并两者，如果两个区域相隔甚远，合并起来就会出现很奇怪的图形，作者使用一个Bounding Box(BB)包含两个Region，然后就可以计算了,其指标是合并后的区域的Bounding Box（能够框住区域的最小矩形（没有旋转））越小，其吻合度越高。其计算方式：

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p6.jpg" /> 
</div>

最后将上述相似度计算方式组合到一起，可以写成如下:

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p7.jpg" /> 
</div>

这里的ai可以取0或者1。

**给区域打分**

通过上述的步骤我们能够得到很多很多的区域，但是显然不是每个区域作为目标的可能性都是相同的，因此我们需要衡量这个可能性，这样就可以根据我们的需要筛选区域建议个数。

这篇文章做法是，给予最先合并的图片块较大的权重，比如最后一块完整图像权重为1，倒数第二次合并的区域权重为2以此类推。但是当我们策略很多，多样性很多的时候，这个权重就会有太多的重合，排序就成了问题。文章做法是给他们乘以一个随机数，看运气，然后对于相同的区域多次出现的也叠加权重，毕竟多个方法都说你是目标，也是有理由的。这样我就得到了所有区域的目标分数，也就可以根据自己的需要选择需要多少个区域了。


**使用选择搜索（selective search)进行物体识别**

通过前面的区域合并，可以得到一系列物体的位置假设L。接下来的任务就是如何从中找出物体的真正位置并确定物体的类别。常用的物体识别特征有HOG（Histograms of oriented gradients）和 bag-of-words 两种特征。在穷举搜索（Exhaustive Search）方法中，寻找合适的位置假设需要花费大量的时间，能选择用于物体识别的特征不能太复杂，只能使用一些耗时少的特征。由于选择搜索（Selective Search）在得到物体的位置假设这一步效率较高，其可以采用诸如SIFT等运算量大，表示能力强的特征。在分类过程中，系统采用的是SVM。

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p8.jpg" /> 
</div>


<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p9.png" /> 
</div>

特征生成

系统在实现过程中，使用color-SIFT特征以及spatial pyramid divsion方法。在一个尺度下σ=1.2下抽样提取特征。使用SIFT、Extended Opponent SIFT、RGB-SIFT特征，在四层金字塔模型 1×1、2×2、3×3、4×4，提取特征，可以得到一个维的特征向量。

训练过程

  训练方法采用SVM。首先选择包含真实结果（ground truth）的物体窗口作为正样本（positive examples），选择与正样本窗口重叠20%~50%的窗口作为负样本（negative examples）。 在选择样本的过程中剔除彼此重叠70%的负样本，这样可以提供一个较好的初始化结果。 在重复迭代过程中加入hard negative examples（得分很高的负样本）由于训练模型初始化结果较好，模型只需要迭代两次就可以了。（样本的筛选很重要！！）


测试的过程

基本和训练过程相同: 首先用Selective Search方法得到测试图像上候选区域 ; 然后提取每个区域的特征向量; 送入已训练好的SVM进行软分类 ; 将这些区域按照概率值进行排序 ; 把概率值小于0.5的区域去除 ; 对那些概率值大于0.5的,计算每个区域与比它分数更高的区域之间的重叠程度,如果重叠程度大于30%,则把这个区域也去除了; 最后剩下的区域为目标区域.


**性能评价**

 很自然地，通过算法计算得到的包含物体的Bounding Boxes与真实情况（ground truth）的窗口重叠越多，那么算法性能就越好。这里使用的指标是平均最高重叠率ABO（Average Best Overlap）。对于每个固定的类别 c，每个真实情况（ground truth）表示为 

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p9.jpg" /> 
</div>

令计算得到的位置假设L中的每个值l，那么 ABO的公式表达为：

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p10.jpg" /> 
</div>

重叠率的计算方式：

<div align=center>
<img src="zh-cn/img/R-CNN/select-search/p11.jpg" /> 
</div>

上面结果给出的是一个类别的ABO，对于所有类别下的性能评价，很自然就是使用所有类别的ABO的平均值MABO（Mean Average Best Overlap）来评价。

```python
pip install selectivesearch
```

```python
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np


def main():

    # 加载图片数据
    img = skimage.data.astronaut() 

    '''
    执行selective search，regions格式如下
    [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
    ]
    '''
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    #计算一共分割了多少个原始候选区域
    temp = set()
    for i in range(img_lbl.shape[0]):
        for j in range(img_lbl.shape[1]):    
            temp.add(img_lbl[i,j,3]) 
    print(len(temp))       #286
    
    #计算利用Selective Search算法得到了多少个候选区域
    print(len(regions))    #570
    #创建一个集合 元素不会重复，每一个元素都是一个list(左上角x，左上角y,宽,高)，表示一个候选区域的边框
    candidates = set()
    for r in regions:
        #排除重复的候选区
        if r['rect'] in candidates:
            continue
        #排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)  
        if r['size'] < 2000:
            continue
        #排除扭曲的候选区域边框  即只保留近似正方形的
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    #在原始图像上绘制候选区域边框
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()
    

if __name__ == "__main__":
    main()

```
-------

### 16.SPP-net

**摘要**

当前深度卷积神经网络（CNNs）都需要输入的图像尺寸固定（比如224×224）。这种人为的需要导致面对任意尺寸和比例的图像或子图像时降低识别的精度。本文中，我们给网络配上一个叫做“空间金字塔池化”(spatial pyramid pooling,)的池化策略以消除上述限制。这个我们称之为SPP-net的网络结构能够产生固定大小的表示（representation）而不关心输入图像的尺寸或比例。金字塔池化对物体的形变十分鲁棒。由于诸多优点，SPP-net可以普遍帮助改进各类基于CNN的图像分类方法。在ImageNet2012数据集上，SPP-net将各种CNN架构的精度都大幅提升，尽管这些架构有着各自不同的设计。在PASCAL VOC 2007和Caltech101数据集上，SPP-net使用单一全图像表示在没有调优的情况下都达到了最好成绩。SPP-net在物体检测上也表现突出。使用SPP-net，只需要从整张图片计算一次特征图（feature map），然后对任意尺寸的区域（子图像）进行特征池化以产生一个固定尺寸的表示用于训练检测器。这个方法避免了反复计算卷积特征。在处理测试图像时，我们的方法在VOC2007数据集上，达到相同或更好的性能情况下，比R-CNN方法快24-102倍。在ImageNet大规模视觉识别任务挑战（ILSVRC）2014上，我们的方法在物体检测上排名第2，在物体分类上排名第3，参赛的总共有38个组。本文也介绍了为了这个比赛所作的一些改进。

**简介**

我们看到计算机视觉领域正在经历飞速的变化，这一切得益于深度卷积神经网络（CNNs）和大规模的训练数据的出现。近来深度网络对图像分类 ，物体检测 和其他识别任务 ，甚至很多非识别类任务上都表现出了明显的性能提升。

然而，这些技术再训练和测试时都有一个问题，这些流行的CNNs都需要输入的图像尺寸是固定的（比如224×224），这限制了输入图像的长宽比和缩放尺度。当遇到任意尺寸的图像是，都是先将图像适应成固定尺寸，方法包括裁剪和变形，如图1（上）所示。但裁剪会导致信息的丢失，变形会导致位置信息的扭曲，就会影响识别的精度。另外，一个预先定义好的尺寸在物体是缩放可变的时候就不适用了。


<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p2.png" /> 
</div>

+ crop：不能包含完整的区域                        
+ warp：几何失真

那么为什么CNNs需要一个固定的输入尺寸呢？CNN主要由两部分组成，卷积部分和其后的全连接部分。卷积部分通过滑窗进行计算，并输出代表激活的空间排布的特征图（feature map）。事实上，卷积并不需要固定的图像尺寸，他可以产生任意尺寸的特征图。而另一方面，根据定义，全连接层则需要固定的尺寸输入。因此固定尺寸的问题来源于全连接层，也是网络的最后阶段。本文引入一种空间金字塔池化( spatial pyramid pooling，SPP)层以移除对网络固定尺寸的限制。尤其是，将SPP层放在最后一个卷积层之后。SPP层对特征进行池化，并产生固定长度的输出，这个输出再喂给全连接层（或其他分类器）。换句话说，在网络层次的较后阶段（也就是卷积层和全连接层之间）进行某种信息“汇总”，可以避免在最开始的时候就进行裁剪或变形。下图展示了引入SPP层之后的网络结构变化。我们称这种新型的网络结构为SPP-net。

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p3.png" /> 
</div>


<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p4.png" /> 
</div>

**什么是空间金字塔池化**

以下图为例进行解释说明：

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p5.png" /> 
</div>

空间金字塔池化（普遍称谓：空间金字塔匹配spatial pyramid matching, SPM），是一种词袋(Bag-of-Words, BoW)模型的扩展。词袋模型是计算机视觉领域最成功的方法之一。

 黑色图片代表卷积之后的特征图，接着我们以不同大小的块来提取特征，分别是4X4，2X2，1X1，将这三张网格放到下面这张特征图上，就可以得到16+4+1=21种不同的块(Spatial bins)，我们从这21个块中，每个块提取出一个特征(比如最大池化)，这样刚好就是我们要提取的21维特征向量。这种以不同的大小格子的组合方式来池化的过程就是空间金字塔池化（SPP）。比如，要进行空间金字塔最大池化，其实就是从这21个图片块中，分别计算每个块的最大值，从而得到一个输出单元，最终得到一个21维特征的输出。

从整体过程来看，就是如下图所示:

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p6.png" /> 
</div>

输出向量大小为Mk，M=#bins， k=#filters，作为全连接层的输入。

例如上图，所以Conv5计算出的feature map也是任意大小的，现在经过SPP之后，就可以变成固定大小的输出了，以上图为例，一共可以输出（16+4+1）*256的特征。

**金字塔池化的意义是什么？**

总结而言，当网络输入的是一张任意大小的图片，这个时候我们可以一直进行卷积、池化，直到网络的倒数几层的时候，也就是我们即将与全连接层连接的时候，就要使用金字塔池化，使得任意大小的特征图都能够转换成固定大小的特征向量，这就是空间金字塔池化的意义（多尺度特征提取出固定大小的特征向量）,实验表明多尺度在深度网络上的精度非常重要。

**网络的训练**

单一尺寸训练

如前人的工作一样，我们首先考虑接收裁剪成224×224图像的网络。裁剪的目的是数据增强。对于一个给定尺寸的图像，我们先计算空间金字塔池化所需要的块（bins）的大小。试想一个尺寸是axa（也就是13×13）的conv5之后特征图。对于nxn块的金字塔级，我们实现一个滑窗池化过程，窗口大小为win = 上取整[a/n]，步幅str = 下取整[a/n]. 对于l层金字塔，我们实现l个这样的层。然后将l个层的输出进行连接输出给全连接层。图4展示了一个cuda卷积网络风格的3层金字塔的样例。(3×3, 2×2, 1×1)。
单一尺寸训练的主要目的是开启多级别池化行为。实验表明这是获取精度的一个原因。

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p7.png" /> 
</div>

多尺寸训练

携带SPP的网络可以应用于任意尺寸，为了解决不同图像尺寸的训练问题，我们考虑一些预设好的尺寸。现在考虑这两个尺寸：180×180,224×224。我们使用缩放而不是裁剪，将前述的224
的区域图像变成180大小。这样，不同尺度的区域仅仅是分辨率上的不同，而不是内容和布局上的不同。对于接受180输入的网络，我们实现另一个固定尺寸的网络。本例中，conv5输出的特征图尺寸是axa=10×10。我们仍然使用win = 上取整[a/n]，str = 下取整[a/n]，实现每个金字塔池化层。这个180网络的空间金字塔层的输出的大小就和224网络的一样了。
这样，这个180网络就和224网络拥有一样的参数了。换句话说，训练过程中，我们通过使用共享参数的两个固定尺寸的网络实现了不同输入尺寸的SPP-net。
为了降低从一个网络（比如224）向另一个网络（比如180）切换的开销，我们在每个网络上训练一个完整的epoch，然后在下一个完成的epoch再切换到另一个网络（权重保留）。依此往复。实验中我们发现多尺寸训练的收敛速度和单尺寸差不多。
多尺寸训练的主要目的是在保证已经充分利用现在被较好优化的固定尺寸网络实现的同时，模拟不同的输入尺寸。除了上述两个尺度的实现，我们也在每个epoch中测试了不同的sxs输入，s是从180到224之间均匀选取的。后面将在实验部分报告这些测试的结果。
注意，上面的单尺寸或多尺寸解析度只用于训练。在测试阶段，是直接对各种尺寸的图像应用SPP-net的。


**SPP-Net用于物体检测**

深度网络已经被用于物体检测。我们简要回顾一下最先进的R-CNN。R-CNN首先使用选择性搜索从每个图像中选出2000个候选窗口。然后将每个窗口中的图像区域变形到固定大小227×227。一个事先训练好的深度网络被用于抽取每个窗口的特征。然后用二分类的SVM分类器在这些特征上针对检测进行训练。R-CNN产生的引人注目的成果。但R-CNN在一张图像的2000个窗口上反复应用深度卷积网络，十分耗时。在测试阶段的特征抽取式主要的耗时瓶颈。

对卷积层可视化发现：输入图片的某个位置的特征反应在特征图上也是在相同位置。基于这一事实，对某个ROI区域的特征提取只需要在特征图上的相应位置提取就可以了。

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p15.png" /> 
</div>

我们将SPP-net应用于物体检测。只在整张图像上抽取一次特征。然后在每个特征图的候选窗口上应用空间金字塔池化，形成这个窗口的一个固定长度表示（见下图）。因为只应用一次卷积网络，我们的方法快得多。我们的方法是从特征图中直接抽取特征，而R-CNN则要从图像区域抽取。之前的一些工作中，可变性部件模型(Deformable Part Model, DPM)从HOG特征图的窗口中抽取图像，选择性搜索方法从SIFT编码后的特征图的窗口中抽取特征。Overfeat也是从卷积特征图中抽取特征，但需要预定义的窗口尺寸。作为对比，我们的特征抽取可以在任意尺寸的深度卷积特征图窗口上。

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p8.png" /> 
</div>


我们使用选择性搜索的“fast”模式对每张图片产生2000个候选窗口。然后缩放图像以满足min(w;h) = s，并且从整张图像中抽取特征图。我们暂时使用ZF-5的SPP-net模型（单一尺寸训练）。在每个候选窗口，我们使用一个4级空间金字塔（1×1, 2×2, 3×3, 6×6, 总共50块）。每个窗口将产生一个12800（256×50）维的表示。这些表示传递给网络的全连接层。然后我们针对每个分类训练一个二分线性SVM分类器。我们使用真实标注的窗口去生成正例。负例是那些与正例窗口重叠不超过30%的窗口（使用IoU比例）。

如果一个负例与另一个负例重叠超过70%就会被移除。我们使用标准的难负例挖掘算法（standard hard negative mining ）训练SVM。这个步骤只迭代一次。对于全部20个分类训练SVM小于1个小时。测试阶段，训练器用来对候选窗口打分。然后在打分窗口上使用最大值抑制算法（30%的阈值）。

通过多尺度特征提取，我们的方法可以得到改进。将图像缩放成min(w;h) = s 属于 S = {480; 576; 688; 864; 1200 }，然后针对每个尺度计算conv5的特征图。一个结合这些这些不同尺度特征的策略是逐个channel的池化。但我们从经验上发现另一个策略有更好的效果。对于每个候选窗口，我们选择一个单一尺度s 属于 S，令缩放后的候选窗口的像素数量接近与224×224。然后我们从这个尺度抽取的特征图去计算窗口的特征。如果这个预定义的尺度足够密集，窗口近似于正方形。我们的方法粗略地等效于将窗口缩放到224×224，然后再从中抽取特征。但我们的方法在每个尺度只计算一次特征图，不管有多少个候选窗口。

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p9.png" /> 
</div>


*具体步骤解析：*

使用SPP-net进行物体检测的流程如下：

①首先通过选择性搜索，对待检测的图片进行搜索出2000个候选窗口。这一步和R-CNN一样。

②特征提取，这点和R-CNN是不同的，具体差别上面已经讲诉。

③最后一步也是和R-CNN一样，采用SVM算法进行特征向量分类识别。
这样一来，就有个问题需要考虑，如何找到原始图片的候选框区域与feature map中提出特征的对应位置呢，因为候选框是通过一整张原图片进行检测得到的，而feature maps的大小和原始图片的大小是不同的，feature maps是经过原始图片卷积、下采样等一系列操作后得到的。那么我们要如何在feature maps中找到对应的区域呢？这个答案可以在文献中的最后面附录中找到答案：APPENDIX A：

Mapping a Window to Feature Maps。这个作者直接给出了一个很方便我们计算的公式：假设(x’,y’)表示特征图上的坐标点，坐标点(x,y)表示原输入图片上的点，那么它们之间有如下转换关系：

(x,y)=(S*x’,S*y’)

其中S的就是CNN中所有的strides的乘积。比如paper所用的ZF-5：

S=2X2X2X2=16

而对于Overfeat-5/7就是S=12，这个可以看一下下面的表格：


<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p11.png" /> 
</div>
需要注意的是Strides包含了池化、卷积的stride。自己计算一下Overfeat-5/7(前5层)是不是等于12。

反过来，我们希望通过(x,y)坐标求解(x’,y’)，那么计算公式如下：

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p12.png" /> 
</div>

因此我们输入原图片检测到的windows，可以得到每个矩形候选框的四个角点，然后我们再根据公式：

Left、Top:

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p13.png" /> 
</div>

Right、Bottom：


<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p14.png" /> 
</div>


<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p16.png" /> 
</div>


最后，用一张图来完整的描述SPP-Net。

<div align=center>
<img src="zh-cn/img/R-CNN/sppnet/p10.png" /> 
</div>



------

### Reference
<https://blog.csdn.net/v1_vivian/article/details/78599229>

<https://github.com/broadinstitute/keras-rcnn>

<https://blog.csdn.net/Katherine_hsr/article/details/79266880>

<https://blog.csdn.net/v1_vivian/article/details/80245397>

<https://blog.csdn.net/bryant_meng/article/details/78613881?utm_source=blogxgwz1>

<https://blog.csdn.net/v1_vivian/article/details/80292569>

<https://blog.csdn.net/zijin0802034/article/details/77685438?utm_source=blogxgwz0>

<https://blog.csdn.net/mao_kun/article/details/50576003>

Selective Search for Object Recognition

Efficient Graph-Based Image Segmentation

<http://cs.brown.edu/people/pfelzens/segment/>

SIFT: Distance image features from scale-invariant keypoints

HOG: Histograms of oriented gradients for human detection

DPM: Object detection with discriminatively trained part based models

SIFT<https://wenku.baidu.com/view/87270d2c2af90242a895e52e.html?sxts=1547523076821>

BOW<https://wenku.baidu.com/view/6370f28d26fff705cc170aab.html?sxts=1547522805171>







