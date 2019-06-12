## R-FCN

------

### 1.简介

**R-FCN贡献**

+ 提出Position-sensitive score maps来解决目标检测的位置敏感性问题；
+ 区域为基础的，全卷积网络的二阶段目标检测框架；
+ 比Faster-RCNN快2.5-20倍（在K40GPU上面使用ResNet-101网络可以达到 170ms/image）;

**R-FCN与传统二阶段网络的异同点**

<div align=center>
<img src="zh-cn/img/r-fcn/p1.png" />
</div>

*R-FCN与传统二阶段网络的异同点*

+ 相同点：都是two-stage的，最终输出的结果都是相应的类别和对应的BB；
+ 不同点：如上图所示，可以看到和Faster R-CNN相比，R-FCN具有更深的共享卷积网络层，这样可以获得更加抽象的特征；同时，它没有RoI-wise subnetwork，不像Faster R-CNN的feature map左右都有对应的网络层，它是真正的全卷积网络架构；从图中的表格可以看出Faster R-CNN的共享卷积子网络是91层，RoI-wise子网络是10层，而R-FCN只有共享卷积子网络，深度为101层。与R-CNN相比，最大的不同就是直接获得整幅图像的feature map，再提取对应的ROI，而不是直接在不同的ROI上面获得相应的feature map。

**分类网络的位置不敏感性**：简单来讲，对于分类任务而言，希望网络有一个很好地分类性能，随着某个目标在图片中不断的移动，网络仍然可以准确的将你区分为对应的类别。实验表明，深的全卷积网络能够具备这个特性，如ResNet-101等。

**检测网络的位置敏感性**：简单来讲，对于检测任务而言，网络有一个好的检测性能，可以准确的输出目标所在的位置值。随着某个目标的移动，网络希望能够和它一起移动，仍然能够准确的检测到它，即对目标位置的移动很敏感。需要计算对应的偏差值，需要计算我的预测和GT的重合率等。但是，深的全卷积网路不具备这样的一个特征。

总之，分类网络的位置不敏感性和检测网络的位置敏感性的一个矛盾问题，而目标检测中不仅要分类也要定位，那么如何解决这个问题呢，R-FCN提出了Position-sensitive score maps来解决这个问题；

### 2.R-FCN网络的设计动机

Faster R-CNN是首个利用CNN来完成proposals预测的，从此之后很多的目标检测网络都开始使用Faster R-CNN的思想。而Faster R-CNN系列的网络都可以分成2个部分：ROI Pooling之前的共享全卷积网络和ROI Pooling之后的ROI-wise子网络（用来对每个ROI进行特征提出，并进行回归和分类）。第1部分就是直接用普通分类网络的卷积层，用来提取共享特征，然后利用ROI Pooling在最后一层网络形成的feature map上面提取针对各个RoIs的特征向量，然后将所有RoIs的特征向量都交给第2部分来处理（即所谓的分类和回归），而第二部分一般都是一些全连接层，在最后有2个并行的loss函数：`softmax`和`smooth L1`，分别用来对每一个RoI进行分类和回归，这样就可以得到每个RoI的真实类别和较为精确的坐标信息啦`(x, y, w, h）`。

需要注意的是第1部分通常使用的都是像VGG、GoogleNet、ResNet之类的基础分类网络，这些网络的计算都是所有RoIs共享的，在一张图片上面进行测试的时候只需要进行一次前向计算即可。而对于第2部分的RoI-wise subnetwork，它却不是所有RoIs共享的，主要的原因是因为这一部分的作用是“对每个RoI进行分类和回归”，所以不能进行共享计算。那么问题就在这里，首先第1部分的网络具有“位置不敏感性”，而如果我们将一个分类网络比如ResNet的所有卷积层都放置在第1部分用来提取特征，而第2部分只剩下全连接层，这样的目标检测网络是位置不敏感的translation-invariance，所以其检测精度会较低，而且这样做也会浪费掉分类网络强大的分类能力（does not match the network's superior classification accuracy）。而ResNet论文中为了解决这个问题，做出了一点让步，即将RoI Pooling层不再放置在ResNet-101网络的最后一层卷积层之后而是放置在了“卷积层之间”，这样RoI Pooling Layer之前和之后都有卷积层，并且RoI Pooling Layer之后的卷积层不是共享计算的，它们是针对每个RoI进行特征提取的，所以这种网络设计，其RoI Pooling层之后就具有了位置敏感性translation-variance，但是这样做会牺牲测试速度，因为所有的RoIs都需要经过若干层卷积计算，这样会导致测试速度很慢。R-FCN就是针对这个问题提出了自己的解决方案，在速度和精度之间进行折中。

### 3.R-FCN架构分析

#### (1).R-FCN算法步骤


<div align=center>
<img src="zh-cn/img/r-fcn/p2.png" />
</div>

*R-FCN算法步骤*

如图所示，先来分析一下R-FCN算法的整个运行步骤，对整个算法有一个宏观的理解，接下来再对不同的细节进行详细的分析。

+ 选择一张需要处理的图片，并对这张图片进行相应的预处理操作；
+ 将预处理后的图片送入一个预训练好的分类网络中（这里使用了ResNet-101网络的Conv4之前的网络），固定其对应的网络参数；
+ 在预训练网络的最后一个卷积层获得的feature map上存在3个分支，第1个分支就是在该feature map上面进行RPN操作，获得相应的ROI；第2个分支就是在该feature map上获得一个`KxKx（C+1）`维的位置敏感得分映射（position-sensitive score map），用来进行分类；第3个分支就是在该feature map上获得一个`4xKxK`维的位置敏感得分映射，用来进行回归；
+ 在`KxKx（C+1）`维的位置敏感得分映射和`4xKxK`维的位置敏感得分映射上面分别执行位置敏感的ROI池化操作（Position-Sensitive Rol Pooling，这里使用的是平均池化操作），获得对应的类别和位置信息。

这样，就可以在测试图片中获得想要的类别信息和位置信息啦。


#### (2).Position-Sensitive Score Map解析

上图是R-FCN的网络结构图，其主要设计思想就是“位置敏感得分图position-sensitive score map”。现在来解释一下其设计思路。如果一个RoI中含有一个类别C的物体，将该RoI划分为`KxK` 个区域，其分别表示该物体的各个部位，比如假设该RoI中含有的目标是人，`K=3`，那么就将“人”划分成了9个子区域，top-center区域毫无疑问应该是人的头部，而bottom-center应该是人的脚部，我们将RoI划分为`KxK`个子区域是希望这个RoI在其中的每一个子区域都应该含有该类别C的物体的各个部位，即如果是人，那么RoI的top-center区域就应该含有人的头部。当所有的子区域都含有各自对应的该物体的相应部位后，那么分类器才会将该RoI判断为该类别。也就是说物体的各个部位和RoI的这些子区域是“一一映射”的对应关系。

现在知道了一个RoI必须是`KxK`个子区域都含有该物体的相应部位，我们才能判断该RoI属于该物体，如果该物体的很多部位都没有出现在相应的子区域中，那么就该RoI判断为背景类别。那么现在的问题就是网络如何判断一个RoI的 `KxK`个子区域都含有相应部位呢？前面是假设知道每个子区域是否含有物体的相应部位，那么就能判断该RoI是否属于该物体还是属于背景。那么现在的任务就是判断RoI子区域是否含有物体的相应部位。

这其实就是position-sensitive score map设计的核心思想了。R-FCN会在共享卷积层的最后一层网络上接上一个卷积层，而该卷积层就是位置敏感得分图position-sensitive score map，该score map的含义如下所述，首先它就是一层卷积层，它的height和width和共享卷积层的一样（即具有同样的感受野），但是它的通道个数为`KxKx(C+1)` 。其中C表示物体类别种数，再加上1个背景类别，所以共有`(C+1)`类，而每个类别都有 `KxK`个score maps。现在只针对其中的一个类别来进行说明，假设我们的目标属于人这个类别，那么其有 `KxK` 个score maps，每一个score map表示原始图像中的哪些位置含有人的某个部位，该score map会在含有对应的人体的某个部位的位置有高的响应值，也就是说每一个score map都是用来描述人体的其中一个部位出现在该score map的何处，而在出现的地方就有高响应值”。既然是这样，那么只要将RoI的各个子区域对应到属于"人"的每一个score map上然后获取它的响应值就好了。但是要注意的是，由于一个score map都是只属于一个类别的一个部位的，所以RoI的第 i个子区域一定要到第i张score map上去寻找对应区域的响应值，因为RoI的第i个子区域需要的部位和第i张score map关注的部位是对应的。那么现在该RoI的`KxK`个子区域都已经分别在属于人的`KxK`个score maps上找到其响应值了，那么如果这些响应值都很高，那么就证明该RoI是"人"。当然这有点不严谨，因为我们只是在属于"人"的 `KxK`个score maps上找响应值，我们还没有到属于其它类别的score maps上找响应值呢，万一该RoI的各个子区域在属于其它类别的上的score maps的响应值也很高，那么该RoI就也有可能属于其它类别呢？是吧，如果2个类别的物体本身就长的很像呢？这就会涉及到一个比较的问题，那个类别的响应值高，我就将它判断为哪一类目标。

这就是position-sensitive score map的全部思想了。


#### (3).Position-Sensitive Rol Pooling解析

上面只是简单的讲解了一下ROl的`KxK`个子区域在各个类别的score maps上找到其每个子区域的响应值，并没有详细的解释这个“找到”是如何找的？这就是位置敏感Rol池化操作（Position-sensitive RoI pooling），其字面意思是池化操作是位置敏感的，下面对它进行解释说明。

如上图所示，通过RPN提取出来的RoI区域，其是包含了`x,y,w,h`的4个值，也就是说不同的RoI区域能够对应到score map的不同位置上，而一个RoI会被划分成`KxK`个bins（也就是子区域。每个子区域bin的长宽分别是` h/k` 和 `w/k` ），每个bin都对应到score map上的某一个区域。既然该RoI的每个bin都对应到score map上的某一个子区域，那么池化操作就是在该bin对应的score map上的子区域执行，且执行的是平均池化。在前面已经讲了，第i个bin应该在第i个score map上寻找响应值，那么也就是在第i个score map上的第i个bin对应的位置上进行平均池化操作。由于有`(C+1)`个类别，所以每个类别都要进行相同方式的池化操作。

<div align=center>
<img src="zh-cn/img/r-fcn/p3.png" />
</div>

*Position-Sensitive Rol Pooling解析*

上图已经很明显的画出了池化的方式，对于每个类别，它都有`KxK`个score maps，那么按照上述的池化方式，ROI可以针对该类别可以获得`KxK`个值，那么一共有`(C+1)`个类别，那么一个RoI就可以得到`KxKx(C+1)`个值，就是上图的特征图。那么对于每个类别，该类别的`KxK`个值都表示该RoI属于该类别的响应值，那么将这`KxK`个数相加就得到该类别的score，那么一共有`(C+1)`个scores，那么在这`(C+1)`个数上面使用简单的`softmax`函数就可以得到各个类别的概率了。


#### (4).Position-Sensitive Regression解析

前面的position-sensitive score map和Position-sensitive RoI pooling得到的值是用来分类的，那么自然需要相应的操作得到对应的值来进行回归操作。按照position-sensitive score map和Position-sensitive RoI pooling思路，其会让每一个RoI得到`(C+1)`个数作为每个类别的score，那么现在每个RoI还需要 4个数作为回归偏移量，也就是`x,y,w,h`的偏移量，所以仿照分类设计的思想，我们还需要一个类似于position-sensitive score map的用于回归的score map。那么应该如何设置这个score map呢，论文中给出了说明：即在ResNet的共享卷积层的最后一层上面连接一个与position-sensitive score map并行的score maps，该score maps用来进行regression操作，我们将其命名为regression score map，而该regression score map的维度应当是 `4xKxK`，然后经过Position-sensitive RoI pooling操作后，每一个RoI就能得到4个值作为该RoI的`x,y,w,h`的偏移量了，其思路和分类完全相同。


#### (5).训练

通过预先计算的区域提议，很容易端到端训练R-FCN架构。根据[6]，我们定义的损失函数是每个RoI的交叉熵损失和边界框回归损失的总和：

$$L(s,t_{x,y,w,h})=L_{cls}(s_{c^\ast})+\lambda[c^{\ast}>0]L_{reg}(t,t^\ast)$$

<!-- <div align=center>
<img src="zh-cn/img/r-fcn/p4.png" />
</div>
 -->


这里 c\*是RoI的真实标签（c\*=0表示背景）。L_cls(s_c\*)=−log(s_c\*)是分类的交叉熵损失，L_reg是[6]中定义的边界框回归损失，t\*表示真实的边界框。[c\*>0]是一个指标，如果参数为true，则等于1，否则为0。我们将平衡权重设置为λ=1，如[6]中所示。我们将正样本定义为与真实边界框重叠的交并比（IoU）至少为0.5的ROI，否则为负样本。

我们的方法很容易在训练期间采用在线难例挖掘（OHEM）(online hard example mining)[22]。我们可忽略的每个RoI计算使得几乎零成本的样例挖掘成为可能。假设每张图像有$N$个提议，在前向传播中，我们评估所有$N$个提议的损失。然后，我们按损失对所有的RoI（正例和负例）进行分类，`并选择具有最高损失的B个RoI`。反向传播[11]是基于选定的样例进行的。由于我们每个RoI的计算可以忽略不计，所以前向传播时间几乎不受$N$的影响，与[22]中的OHEM Fast R-CNN相比，这可能使训练时间加倍。我们在下一节的表3中提供全面的时间统计。

我们使用`0.0005`的权重衰减和`0.9`的动量。默认情况下，我们使用单尺度训练：调整图像的大小，使得尺度（图像的较短边）为600像素[6，18]。每个GPU拥有1张图像，并为反向传播选择B=128个RoI。我们用8个GPU来训练模型（所以有效的最小批数据大小是8×）。在VOC上我们对R-FCN进行微调，使用`0.001`学习率进行2万次迭代和使用`0.0001`学习率进行1万次迭代。为了使R-FCN与RPN共享特征，我们采用[18]中的`四步交替训练`，交替训练RPN和R-FCN。

#### (6).推断

在RPN和R-FCN之间计算共享的特征映射（在一个单一尺度的图像上）。然后，RPN部分提出RoI，R-FCN部分在其上评估类别分数并回归边界框。在推断过程中，我们评估了300个RoI进行公平比较，如[18]中那样。作为标准实践，使用0.3的IoU阈值[7]，通过非极大值抑制（NMS）对结果进行后处理。

#### (7).空洞卷积和步长

我们的全卷积架构享有FCN广泛使用的语义分割的网络修改的好处[15，2]。特别的是，我们将ResNet-101的有效步长从32像素降低到了16像素，增加了分数图的分辨率。conv4阶段[9]（stride = 16）之前和之后的所有层都保持不变；第一个conv5块中的stride=2操作被修改为stride=1，并且conv5阶段的所有卷积滤波器都被“hole algorithm”[15,2]（“Algorithm atrous”[16]）修改来弥补减少的步幅。为了进行公平的比较，RPN是在conv4阶段（与R-FCN共享）之上计算的，就像[9]中Faster R-CNN的情况那样，所以RPN不会受空洞行为的影响。下表显示了R-FCN的消融结果（`k×k=7×7`，没有难例挖掘）。这个空洞窍门提高了2.6点的mAP。

#### (8).可视化

在下面图中，当`k×k=3×3`时，我们可视化R-FCN学习的位置敏感分数图。期望这些专门的分数图将在目标特定的相对位置被强烈激活。例如，“顶部中心敏感”分数图大致在目标的顶部中心位置附近呈现高分数。如果一个候选框与一个真实目标精确重叠，则RoI中的大部分k2组块都被强烈地激活，并且他们的投票导致高分。相反，如果一个候选框与一个真实的目标没有正确的重叠，那么RoI中的一些k2组块没有被激活，投票分数也很低。

<div align=center>
<img src="zh-cn/img/r-fcn/p5.png" />
</div>

*位置敏感得分映射表现1*

<div align=center>
<img src="zh-cn/img/r-fcn/p6.png" />
</div>

*位置敏感得分映射表现2*

### 4.R-FCN性能分析

#### (1).定量结果分析

<div align=center>
<img src="zh-cn/img/r-fcn/p7.png" />
</div>

*使用ResNet-101全卷积策略*

如上表所示，作者测试了不同大小的ROI对性能的影响（使用了预训练的ResNet-101网络，在VOC 07数据集上面进行测试），可以看到如果使用`1x1`的ROI，显示输出失败，具体原因不得而知。当使用`7x7`的ROI时，能够获得最好的结果，这也是论文中最终使用`7x7`大小的ROI的原因吧，作者应该是做了很多的验证工作。


<div align=center>
<img src="zh-cn/img/r-fcn/p8.png" />
</div>

*Faster R-CNN与R-FCN性能比较*

如上表所示，比较了Faster R-CNN和R-FCN的性能，从表中可以看出与Faster R-CNN相比，R-FCN有更快的运行速度，大概是2.5倍以上。另外，可以发现性能稍微有一点点提升，当调整ROI的个数时，发现300个ROI时能够获得最好的性能。

<div align=center>
<img src="zh-cn/img/r-fcn/p9.png" />
</div>

*预训练网络的深度对性能的影响*

如上表所示，随着预训练网络层数的加深，检测性能在不断的得到提高，使用VGG和ResNet网络还是有很大的性能差异，但是过深的网络并没有提高其性能，可能的原因是网络发生了过拟合情况.

<div align=center>
<img src="zh-cn/img/r-fcn/p10.png" />
</div>

*COCO数据集的训练结果*

如上表所示，采用了COCO数据集进行性能验证，与Faster R-CNN相比，R-FCN可以实现3倍的加速，准确率可以提升2个百分点。

#### (2).定性结果分析

<div align=center>
<img src="zh-cn/img/r-fcn/p11.png" />
</div>

*VOC 2007检测结果*

<div align=center>
<img src="zh-cn/img/r-fcn/p12.png" />
</div>

*COCO检测结果*

以上是R-FCN算法在VOC2007和COCO数据集上面的性能表现，总体上看效果还是挺不错的，具体的效果需要你自己去尝试，根据自己的需求去选择合适的算法。


### 5.总结

与Faster R-CNN相比，R-FCN具有更快的运行速度（2.5倍以上），稍微提高了一点检测精度，在速度和准确率之间进行了折中，提出position-sensitive score map来解决检测的位置敏感性问题。算法中的很多细节值得我们进行深入的研究和分析。


### 6.参考文献

[1] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.

[2] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. In ICLR, 2015.

[3] J. Dai, K. He, Y. Li, S. Ren, and J. Sun. Instance-sensitive fully convolutional networks.arXiv:1603.08678, 2016.

[4] D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable object detection using deep neural networks. In CVPR, 2014.

[5] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes (VOC) Challenge. IJCV, 2010.

[6] R. Girshick. Fast R-CNN. In ICCV, 2015.

[7] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[8] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV. 2014.

[9] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[10] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

[11] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[12] K. Lenc and A. Vedaldi. R-CNN minus R. In BMVC, 2015.

[13] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[14] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed. SSD: Single shot multibox detector. arXiv:1512.02325v2, 2015.

[15] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[16] S. Mallat. A wavelet tour of signal processing. Academic press, 1999.

[17] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016.

[18] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

[19] S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. arXiv:1504.06066, 2015.

[20] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

[21] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.

[22] A. Shrivastava, A. Gupta, and R. Girshick. Training region-based object detectors with online hard example mining. In CVPR, 2016.

[23] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[24] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[25] C. Szegedy, A. Toshev, and D. Erhan. Deep neural networks for object detection. In NIPS, 2013.

[26] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[27] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders. Selective search for object recognition. IJCV, 2013.

[28] C. L. Zitnick and P. Dollár. Edge boxes: Locating object proposals from edges. In ECCV, 2014.

[29] https://blog.csdn.net/WZZ18191171661/article/details/79481135

[30] https://blog.csdn.net/Quincuntial/article/details/79198612