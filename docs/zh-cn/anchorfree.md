<div align=center>
<img src="zh-cn/img/anchorfree/p1.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/anchorfree/p2.png" /> 
</div>

## 1.Stacked Hourglass Networks for Human Pose Estimation

!> 论文地址：https://arxiv.org/abs/1603.06937

!> 引入该论文的目的是在`CornerNet`中我们会用到沙漏网络的backbone

### 1.动机

捕获每个尺度的特征，并将特征结合

### 2.模型结构

- 数据预处理

  要做姿态估计，首先要把人物从图片中找到，本文没有用到人体检测的模型，需要用到以下操作来将人物从原图中裁剪出来。

  **操作**：拿到一张输入图片，如果图片中有一个人，根据人的标注数据（中心和尺度），把人物从图片中裁剪出来，并调整为`256x256`（像素）；如果图片中有多个人，谁在图片中心就对谁进行姿态估计，然后利用上述方法。FLIC是根据标注将人放到图片中间作为target person，MPII直接根据标注得到target person；当人物离得非常近，甚至遮挡重叠的时候，不管以谁为中心，裁剪后的图片都会包括多个人，那么就对多个人分别进行姿态估计, 如下图所示。

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p1.png" /> 
</div>

- 模型结构

本教程按照`模块`,`子网络`,`完整网络`，`网络输入，输出`的过程进行讲解。

**1.模块**

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p2.png" /> 
</div>

这里运用了Resnet里面的残差模块(信息参考本站教程)
Resnet原文有两个残差块，一个是上面那个，另外是下面这个

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p3.png" /> 
</div>


之所以选了上一个，个人理解是：下面这个参数更多，内存利用率高（简单计算一下：参数主要出现在卷积层的卷积核上，一个上面的残差块参数为`1x1+3x3+1x1=11`，下面这个参数为`3x3x2=18`，你一个残差块比下面多`7`个参数，别忘了，残差块构成沙漏，沙漏再堆叠形成最终的网络，所以参数会更多），关于更细致的讲解请参考ResNet V1和V2教程

进一步理解：上图的前一个`1x1`的卷积起到降维的作用（256到128），后一个`1x1`起升维的作用（128到256）。整个残差块没有对图片的大小（宽高）产生影响，只是改变了它的通道数(深度)。
如果上下两条路径的维度不同，下面路径可以添加一个`1x1`的卷积，起到改变维度的作用。如下图：

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p4.png" /> 
</div>

Residual模块提取了较高层次的特征（上面一层），同时保留了原有层次的信息（下面一层）。不改变数据尺寸，只改变数据深度。

**2.Hourglass子网络**

Hourglass(沙漏)是本文的核心部件，由Residual模块组成。根据阶数不同，有不同的复杂程度。

*一阶Hourglass*

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p5.png" /> 
</div>

上下两个半路包含若干Residual模块(浅绿),逐步提取更深层次的特征。但上半路在原尺度进行，下半路经历了先降采样（`红色/2`)再升采样(`红色*2`)的过程。降采样使用max pooling, 升采样使用最近邻插值。另一种进行升采样的方法是反卷积层。

*二阶Hourglass*

把一阶模块的灰框内部分替换成一个一阶Hourglass（输入通道256，输出通道N），得到二阶Hourglass：

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p6.png" /> 
</div>

两个层次的下半路组成了一条两次降采样，再两次升采样的过程。两个层次的下半路则分别在原始尺寸(OriSzie)和`1/2`原始尺寸，辅助升采样。

*四阶Hourglass*

将上述得到的二阶Hourglass再重复操作两次，得到四阶Hourglass。本文使用的是四阶Hourglass：

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p7.png" /> 
</div>


再看下论文里面的这幅图

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p8.png" /> 
</div>

两幅图是不是很像？论文里没有仔细讲解残差块形成沙漏的过程

规律：

每次降采样之前，分出上半路保留原尺度信息；

每次升采样之后，和上一个尺度的数据相加；

两次降采样之间，使用三个Residual模块提取特征；

两次相加之间，使用一个Residual模块提取特征。

`n`阶Hourglass子网络提取了从原始尺度到`1/2n`尺度的特征。不改变数据尺寸，只改变数据通道数(深度)。

注意：上面仅仅得到一个Hourglass，要得到完整的网络，还要经历堆叠的过程。


**3.完整网络结构**

*一级网络*

以一个Hourglass（深绿色）为中心，可以从彩色图像预测`K`个人体部位的响应图：

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p9.png" /> 
</div>

原始图片经过一次降采样(橙色)，输入到Hourglass子网络中。Hourglass的输出结果经过两个线性模块(灰色)，得到最终的响应图。期间使用Residual模块(浅绿)和卷积层(白色)逐步提取特征。

*二级网络*

包含两个Hourglass

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p10.png" /> 
</div>

对应论文里面的这张图

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p11.png" /> 
</div>


规律：

第一个Hourglass包括以下三部分：

1.第一个Hourglass的输入数据（图三橙色pooling之后输出的那条线、下图中虚线）

2.第一个Hourglass的输出（上图`<512><256>`残差块输出的那条线、下图中间那条线）

3.第一级预测结果（即heatmap的重映射，使用`1x1`的卷积是为了保证通道数相同）
这三路数据通过串接（concat）和相加进行融合

本文结构是以两个Hourglass（深绿色）为中心的二级网络。二级网络重复了一级网络的后半结构。第二个Hourglass的输入包含三路： 第一个Hourglass的输入数据 ，第一个Hourglass的输出数据 ，第一级预测结果 。这三路数据通过串接（concat）和相加进行融合，它们的尺度不同，体现了当下流行的跳级结构思想。多级网络可以依据上面的规律进行推导，论文中用到了8级网络

**4.代价函数与训练**

对于`H×W×3`   的输入图像，每一个hourglass级都会生成一个`H/2×W/2×K`的响应图。`ground truth heatmap`都是一样的，根据关键点`label`生成一个二维的高斯图（以关键点为中心的二维高斯分布，标准差为`1px`）。对于每个响应图，都比较其与ground truth的误差作为代价，使用`MSE`（均方误差）来作为`LOSS`函数。体现了中继监督`(intermediate supervision)`的思想。

在级联漏斗网络中加入了中间监督，就是中间结果和真值数据比较，提升网络的性能，这个有点类似人脸识别中的 FaceID系列中采用的方法。


**5.预测与实战**

预测时，输入任意一张人物图片（`256*256`),得到最后一个stack(网络级数)的输出热图（不是所有的）。算出每一个joint对应的热图中最大值元素所在的坐标，然后scale回去，作为最后预测到的关键点位置。

作者在github上工程里，默认是设置8个stack(8级网络），训练出来的模型在torch上达到205M，我后来对另一个tensorflow版本的代码进行修改，并按照与torch版本一样的方法进行训练，得到的模型达到301M，可以看出，该网络的参数是非常庞大的，肯定会有很多冗余的参数，如果想在移动端使用，需要进行进一步的修改和压缩。



### 3.消融实验

说明了性能的提升是来自堆叠沙漏带来的，而不是更大更深的网络引起的。

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p12.png" /> 
</div>

其次，见下图

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p13.png" /> 
</div>

从右图看出，2-stack、4-stack、8-stack中，8-stack的效果最好。


### 4.多人的情况

<div align=center>
<img src="zh-cn/img/anchorfree/shn/p1.png" /> 
</div>


本文研究的是单人姿态估计，遇到多人的情况，都是根据标注信息，选出target person。如果两个人很接近，如上图两个人中心点相差`26px`，那么两个人都会进行姿态估计。



### 5.细节整理

1. 重复的bottom-up（高分辨率到低分辨率）、top-down（低分辨率到高分辨率）和中间监督的使用
2. 对称结构
3. 单人姿态估计
4. 网络最低分辨率为4x4
5. bottom-up阶段：进行一系列卷积、池化（最大池化）操作，生成低分辨率的图片；top-down阶段：
进行nearest neighbor upsampling（最邻近插值）和跳跃连接。
6. 沙漏网络最后有两个`1x1`的卷积，生成heatmap（表示每个像素下，关节点存在的概率）作为输出。
7. 大于`3x3`的滤波器（filter）不被使用
8. 输入分辨率为`256x256`，最终输出分辨率为`64x64`
9. 预测了heatmap之后，就可以使用LOSS函数进行优化了。LOSS函数为MSE（均方误差），其中ground truth相同。ground truth heatmap是以关节点为中心的二维高斯分布（标准差为`1px`）
10. heatmap之后有一个`1x1`的卷积，用于将中间预测整合到特征空间（起维度匹配的作用）
11. 本文实现的是单人的姿态估计，遇到图片中有多个人，该怎么办？
12. 最终预测是heatmap的max activation location（最大值）


### Reference

[1].https://blog.csdn.net/qq_29631521/article/details/89254001

[2].https://blog.csdn.net/qq_38522972/article/details/82958077

[3].https://blog.csdn.net/shenxiaolu1984/article/details/51428392#commentBox

[4].https://blog.csdn.net/saturdaysunset/article/details/84204564

[5].https://blog.csdn.net/qq_36165459/article/details/78321529

[6].https://arxiv.org/abs/1603.06937

[7].https://github.com/wbenbihi/hourglasstensorlfow


## 2.CornerNet: Detecting Objects as Paired Keypoints

!> 论文地址：https://arxiv.org/abs/1808.01244

!> 代码链接：https://github.com/princeton-vl/CornerNet

这篇文章是ECCV2018的一篇目标检测论文，该论文的创新之处在于使用Keypoints代替原来的anchor思想进行目标检测，提出检测目标左上点和右下点来确定一个边界框，提出一个新的池化方法：corner pooling，在mscoco数据集上达到42.2%的AP，精度上是当时的单阶段目标检测器的SOTA，但是速度略慢，大约1fps（论文为Titan X 244ms/f），无法满足工程需求。

相对于基于anchor检测器创新意义有：

1. anchor数量巨大，造成训练正负样本不均衡（anchor机制解决方式为难例挖掘，比如ohem，focal loss）
2. anchor超参巨多，数量，大小，宽高比等等（比如yolo多尺度聚类anchor，ssd的多尺度aspect ratio）

### 1.模型结构

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p1.png" /> 
</div>

如上图Fig1，经过特征提取主干网络（主干网络为Hourglass-104）后分为两个分支（两个分支分别接前面提到的corner pooling，随后细谈），一个分支生成目标左上点热力图，一个分支生成目标右下点热力图，而此时两个热力图并没有建立联系，因此无法确定两点是够属于同一目标，因此两分支同时生成embeddings，通过判断两个embedding vector的相似性确定同一物体（距离小于某一阈值则划为同一目标）。

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p2.png" /> 
</div>

如上图Fig4，图片首先经过1个`7×7`的卷积层将输入图像尺寸缩小为原来的`1/4`（论文中输入图像大小是`511×511`，缩小后得到`128×128`大小的输出）。

然后经过`hourglass-104`提取特征，该网络通过串联多个hourglass模块组成（Fig4中的hourglass network由2个hourglass模块组成），每个hourglass 模块都是先通过一系列的降采样操作缩小输入的大小，然后通过上采样恢复到输入图像大小，因此该部分的输出特征图大小还是`128×128`，整个hourglass network的深度是`104`层。

hourglass模块后会有两个输出分支模块，分别表示左上角点预测分支和右下角点预测分支，每个分支模块包含一个corner pooling层和3个输出：heatmaps、embeddings和offsets。heatmaps是输出预测角点信息，可以用维度为`CHW`的特征图表示，其中`C`表示目标的类别（无背景类），每个点的预测值为0到1，表示该点是角点的分数；embeddings用来找到属于同一个目标的左上角角点和右下角角点；offsets用来对预测框做微调，与anchor机制中的offset有区别，前者用来微调特征图映射回原图后的取整量化误差，后者用来表示ground true与anchor的偏移。


### 2.Headmaps

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p3.png" /> 
</div>

CornerNet的第一个输出headmap用来预测角点的位置。公式（1）是针对角点预测（headmaps）的损失函数，是修改后的focal loss。

$p_{cij}$表示预测的heatmaps在第`c`个通道（类别`c`）的$(i,j)$位置的值，$y_{cij}$表示对应位置的ground truth，`N`表示目标的数量。$y_{cij=1}$时候的损失函数容易理解，就是focal loss，`α`参数用来控制难易分类样本的损失权重；$y_{cij}$等于其他值时表示$(i,j)$点不是类别`c`的目标角点，照理说此时$y_{cij}$应该是0（大部分算法都是这样处理的），但是这里$y_{cij}$不是0，而是用基于ground truth角点的高斯分布计算得到，因此距离ground truth比较近的$(i,j)$点的$y_{cij}$值接近1，这部分通过`β`参数控制权重，这是和focal loss的差别。因为靠近ground truth的误检角点组成的预测框仍会和ground truth有较大的重叠面积，如下图所示，红色实线框是ground truth；橘色圆圈是根据ground truth的左上角角点、右下角角点和设定的半径值画出来的，半径是根据圆圈内的角点组成的框和ground truth的IOU值大于0.7而设定的，圆圈内的点的数值是以圆心往外呈二维的高斯分布；白色虚线是一个预测框，可以看出这个预测框的两个角点和ground truth并不重合，但是该预测框基本框住了目标，因此是有用的预测框，所以要有一定权重的损失返回，这就是为什么要对不同负样本点的损失函数采取不同权重值的原因。
这里半径用$r$表示,半径内点的数值是以正样本角位置为中心的非标准化的2D高斯分布,其中$\sigma=\frac{1}{2}r$


<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p4.png" /> 
</div>


### 3.Embeddings

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p5.png" /> 
</div>

CornerNet的第二个输出是embeddings，对应文章中group corner的内容。前面介绍了关于角点的检测，在那部分中对角点的预测都是独立的，不涉及一个目标的一对角点的概念，因此如何找到一个目标的两个角点就是第二个输出embedding做的工作。这部分是受`associative embedding`那篇文章的启发，简而言之就是基于不同角点的embedding vector之间的距离找到每个目标的一对角点，如果一个左上角角点和一个右下角角点属于同一个目标，那么二者的embedding vector之间的距离应该很小。

embedding这部分的训练是通过两个损失函数实现的，$e_{tk}$表示属于`k`类目标的左上角角点的embedding vector，$e_{bk}$表示属于`k`类目标的右下角角点的embedding vector，$e_k$表示$e_{tk}$和$e_{bk}$的均值。公式（4）用来缩小属于同一个目标（`k`类目标）的两个角点的embedding vector（$e_{tk}$和$e_{bk}$）距离。公式（5）用来扩大不属于同一个目标的两个角点的embedding vector距离。

### 4.Offsets

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p6.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p7.png" /> 
</div>


CornerNet的第三个输出是offset，这个值和目标检测算法中预测的offset类似却完全不一样，说类似是因为都是偏置信息，说不一样是因为在目标检测算法中预测的offset是表示预测框和anchor之间的偏置，而这里的offset是表示在取整计算时丢失的精度信息，如上式（2），其中$(xk,yk)$表示第`k`个角点的原图坐标值，`n`代表下采样因子，$o_k$表示特征图缩放回原图后与原gt框的精度损失。然后通过公式（3）的smooth L1损失函数监督学习该参数，和常见的目标检测算法中的回归支路类似。

### 5.Corner pooling

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p8.png" /> 
</div>


CornerNet是预测左上角和右下角两个角点，但是这两个角点在不同目标上没有相同规律可循，如果采用普通池化操作，那么在训练预测角点支路时会比较困难。作者认为左上角角点的右边有目标顶端的特征信息（第一张图的头顶），左上角角点的下边有目标左侧的特征信息（第一张图的手），因此如果左上角角点经过池化操作后能有这两个信息，那么就有利于该点的预测。Fig3是针对左上角点做corner pooling的示意图，该层有2个输入特征图，特征图的宽高分别用W和H表示，假设接下来要对图中红色点（坐标假设是`(i,j)`）做corner pooling，那么就计算`(i,j)`到`(i,H)`的最大值（对应Fig3上面第二个图），类似于找到Fig2中第一张图的左侧手信息；同时计算`(i,j)`到`(W,j)`的最大值（对应Fig3下面第二个图），类似于找到Fig2中第一张图的头顶信息，**然后将这两个最大值相加得到`(i,j)`点的值（对应Fig3最后一个图的蓝色点）**。右下角点的corner pooling操作类似，只不过计算最大值变成从`(0,j)`到`(i,j)`和从`(i,0)`到`(i,j)`。

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p9.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p14.png" /> 
</div>

Fig6也是针对左上角点做corner pooling的示意图，是Fig3的具体数值计算例子，该图一共计算了4个点的corner pooling结果。


### 6.Prediction module

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p10.png" /> 
</div>

Fig7是Fig4中预测模块的详细结构,预测模型结构如下，模型的第一部分为修改的残差块，将第一个`3x3`的卷积替换为corner pooing,首先通过2个`3x3x128`的卷积核来处理来自backbone的feature map，后接一个corner pool层，将Pooled后的feature map送入`3x3x256`的conv-BN层中，同时增加了一个映射短链接，修正的残差块后接一个`3x3x256`的卷积及三个conv-BN-ReLU模块用于预测heatmaps,embedings,offsets。

### 7.Loss function

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p11.png" /> 
</div>

$L_{det}$为角点损失，$L_{pull}$、$L_{push}$为embedding损失，$L_{off}$为offset损失,其中`α`为`0.1`，`β`为`0.1`，`γ`为`1`，损失函数优化方式为`Adam`

### 8.Testing details

1. 在得到预测角点后，会对这些角点做`NMS`操作，选择前100个左上角角点和100个右下角角点。

2. 计算左上角和右下角角点的embedding vector的距离时采用`L1`范数，距离大于`0.5`或者两个点来自不同类别的目标的都不能构成一对，检测分数是两个角点的平均分数。

3. 测试图像采用`0`值填充方式得到指定大小作为网络的输入，而不是采用`resize`，另外同时测试图像的水平翻转图并融合二者的结果。

4. 最后通过`soft-nms`操作去除冗余框，只保留前`100`个预测框。

### 9.测试结果

在MS COCO数据集上的测试结果如下所示：

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p12.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/anchorfree/cornernet/p13.png" /> 
</div>


### 10.总结

本论文主要提出使用一对关键点（左上角点，右下角点）进行目标检测，并取得非常好的检测精度，由此掀起anchor-free热潮,后文我们还会介绍普林斯顿大学团队提出的cornernet-lite(https://arxiv.org/abs/1904.08900) ，该论文在速度和精度上均对cornernet进行提升，另一篇同期论文（2019.04）centernet（https://arxiv.org/abs/1904.08189) 提出Keypoint Triplets思想也对cornernet进行优化，达到目前单阶段目标检测器最高精度（47.0%）。接下来我们将对以上两篇论文进行总结，并有可能分析cornernet系列的源码实现细节。


### Reference

[1].https://www.w3xue.com/exp/article/20195/35392.html

[2].https://zhuanlan.zhihu.com/p/44553304

[3].https://blog.csdn.net/m_buddy/article/details/84920996

[4].https://arxiv.org/abs/1808.01244

[5].https://github.com/princeton-vl/CornerNet


## 3.CornerNet-Lite


## 4.ExtremeNet: Bottom-up Object Detection by Grouping Extreme and Center Points

!> 论文地址：https://arxiv.org/abs/1901.08043

!> 项目地址：https://github.com/xingyizhou/ExtremeNet

### 1.概述

ExtremeNet是今年（2019）1月23号挂在arxiv上的目标检测论文，是至今为止检测效果最好的单阶段目标检测算法。思想借鉴CornerNet，使用标准的关键点估计网络检测目标关键点进而构造目标的预测框。ExtremeNet预测四个extreme point（顶、左、底、右）以及目标的中心点，如果这五个点满足几何对齐关系，就将其组合成一个目标框。ExtremeNet在COCO test-dev上的AP达到43.2%。此外，根据extreme point还可以得到更加精细的八边形分割估计结果，在COCO Mask上的AP达到34.6%。

### 2.Preliminaries

**Extreme and center points**。常见做法会用左上和右下两点标注矩形框。由于两个角点经常位于对象外部，这会导致效果不准确，并且需要多次调整，整个过程平均花费时间较长。本文则直接用极值点代替矩形框。若四个极值点为$((x^t,y^t),(x^i,y^i),(x^b,y^b),(x^r,y^r))$，则中心点为:

$$(\frac{x^l+x^r}{2},\frac{y^t+y^b}{2})$$

**Keypoint detection**。目前效果最好的关键点估计网络是104层的HourglassNetwork，该网络采用完全卷积的方式进行训练。HourglassNetwork为每个输出通道回归一个宽为W、高为H的heatmap：

$$\hat{Y} \in (0,1)^{HW}$$

训练时的label为多峰高斯热图multi-peak Gaussian heatmap $Y$，其中每个关键点定义高斯核的平均值。标准差要么是固定的，要么与对象大小成正比。高斯热图在L2 loss的情况下会作为回归目标，在逻辑回归的情况下作为weight map来减少正样本点附近像素点的惩罚。

**CornetNet**。CornerNet使用HourglassNetwork作为检测器进行关键点估计，为矩形框的两个角点预测两套热图。为了平衡正样本点和负样本点，CornerNetwork在训练时使用了修正额focal loss：

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p1.png" /> 
</div>

其中`α`和`β`是超参数，训练期间固定`α=2、β=4`，`N`是图片中对象的数量。
为了极点的sub-pixel准确度，CornerNetwork为每个角点回归了类别未知的关键点偏移量`△(α)`。这个回归过程恢复了HourglassNetwork下采样过程中丢失的信息。The offset map is trained with smooth L1 Loss [11] SL1 on ground truth extreme point locations:

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p2.png" /> 
</div>

其中，`s`是下采样因子（HourglassNetwork的`s=4`），$\vec{x}$为关键点坐标。
接着，CornerNet使用关联特征将角点分组，本文沿用了CornerNet的网络架构和loss，但没有沿用关联特征。

**Deep Extreme Cut**。DeepExtremeCut (DEXTRE)是一种基于极值点的图像实例分割算法，该算法取了四个极值点并裁剪这四个极值点组成的矩形框的图片区域作为输入，利用语义分割网络对相应对象进行类别不确定的前景分割。DeepExtremeCut学习了匹配输入极点的分割掩码。


### 3.ExtremeNet介绍

ExtremeNet是一个自底向上的目标检测框架，检测目标的四个极值点（顶端、左端、底端、右端），使用SOTA的关键点估计框架产生每个类别的五个Heatmaps（四个极值点和一个中心点）。使用纯几何方法组合同一目标的极值点：四个极值点的几何中心与预测的中心点heatmap匹配且高于阈值（暴力枚举，时间复杂度$O(n^4)$，不过$n$一般非常小）。

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p3.png" /> 
</div>

上图展示了算法的大致流程。首先产生四个预测极值点的heatmap（图顶部）和一个预测中心点的heatmap（图左下），提取极值点heatmap的峰值（图中左），暴力枚举所有组合，计算几何中心（图中右），如果几何中心与中心heatmap高度匹配，则接受该组合，否则拒绝（图右下）。

该算法与CornerNet的区别在于关键点定义和组合。（1）CornerNet采用左上和右下角点，角点往往不在目标上，没有较强的外观特征；而ExtremeNet采用极值点，极值点在目标上，容易区分且具有一致的局部外观特征。（2）CornerNet点对组合是根据embedding vector的距离，而ExtremeNet则是根据几何中心点。ExtremeNet完全基于外观，没有任何的隐特征学习。


### 4.网络结构

ExtremeNet采用HourglassNetwork检测每个类别的5个关键点（四个极值点、一个中心点），沿用了CornerNet训练配置、loss和偏移量预测部分。偏移量预测部分是类别未知的，但极值点是类别明确的。偏移量预测是针对极值点的，中心点是没有偏移量预测的。 ExtremeNet网络输出`5xC`个heatmaps和`4x2`个偏移量maps（如下图），其中`C`是类别数量。一旦极值点明确了，网络就用几何的方式将他们分组到检测中。

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p4.png" /> 
</div>


### 5.Center Grouping(中心点分组)

极值点位于对象不同的边上，这使得组合变得比较复杂，比如关联特征没有足够的全局特征来分组关键点。本文通过利用极值点的扩散性，提出了一个不同的分组方法。

本文分组算法的输入是每个类别的五个heatmaps：一个中心heatmap和上下左右四个极值点。已知一个heatmap，本文通过检测所有的峰值点来抽取相应的关键点。一个峰值点是值大于$\tau_p$的任意像素点，且在周围的`3x3`像素窗口是局部最大值，这个过程叫做**ExtrectPeak**。

已知从热力图$\hat Y^{(t)},\hat Y^{(l)},\hat Y^{(b)},\hat Y^{(r)}$中抽取的四个极点`t, b, r, l`，可计算出几何中心$c=(\frac{l_x+r_x}{2}, \frac{t_y+b_y}{2})$.如果中心点`c`被预测在center map$\hat Y^{(c)}$中，且分数较高，则认为这四个极点是一个有效检测：$\hat Y_{c_x,c_y}^{(c)} \geq \tau_c$ 其中($\tau_c$为阈值)。接着，以暴力方式罗列出所有的关键点`t, b, r, l`，分别从每一类中抽取检测。如下图，实验中设置$\tau_p=0.1$, $\tau_c=0.1$

上述的暴力分组算法时间复杂度为$O(n^4)$，其中`n`为每个基础方向抽取的极点数量。补充的材料中提出一种比我们更快的时间复杂度为$O(n^2)$的算法，然而很难在GPU上加速，且当`n=40`时在MS COCO数据集上的效果比我们的算法要慢。

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p5.png" /> 
</div>

### 6.Ghost box suppression(Ghost box抑制)

中心点分组可能对相同大小的三个等间距共线对象给出高置信度的false-positive检测。这里位于中间的对象有两个选择，检测出正确的小框，或预测一个包含相邻对象极值点的比较大的框，这些false-positive检测被称作ghost boxes。在实验中，这些ghost boxes是不常见的，但却是模型分组中的唯一错误类型。

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p6.png" /> 
</div>


<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p7.png" /> 
</div>

模型提出了一种简单的后处理方法来相处ghost boxes。根据定义一个ghost box包含许多其他小的检测对象，采用soft-NMS解决这个问题。若某个box所包含的所有框的分数和大于其自身的3倍，则最终分数除以2。soft-NMS类似于标准的基于重叠的NMS，但惩罚的是ghost boxes，而不是重叠的boxes。我们将在第10节介绍soft-NMS。


### 7.Edge aggregation(边缘融合)

极值点不总是唯一定义的，若极值点来自物体的垂直或水平边（如汽车顶部），则沿边缘任何一点都可被认为是极值点。因此我们的网络在对象的任一对齐边上产生弱响应，而不是单一的强峰值响应。这种弱响应存在两个问题：①弱响应可能低于峰值选择阈值，将完全错过极值点；②即使检测到关键点，它的分值也低于一个轻微旋转对象的强峰值响应。

使用**边缘聚合**来解决这个问题。对于提取为局部最大值的每个极值点，在垂直方向(左右极点)或水平方向(顶部和底部极点)汇总其得分。然后对所有单调递减的分数进行聚合，并在聚合方向上的局部最小值处停止聚合。 特别地，若$m$为极值点，$N_i^{(m)}=\hat Y_{m_xi,m_y}$为该点的垂直或水平线段。若$i_0<0$, $i_1>0$,则两个最近的局部最小值为$N_{i_0}^{(m)}$和$N_{i_1}^{(m)}$，其中$N_{i_0-1}^{(m)}>N_{i_0}^{(m)}$,$N_{i_1}^{(m)}<N_{i_1+1}^{(m)}$。边缘聚合更新关键点的值为
$$\tilde Y_m=\hat Y_m+\lambda_aggr \sum_{i=i_0}^{i_1}N_i^{(m)}$$,其中$\lambda_aggr$是聚合权重。在我们的实验中，设置$\lambda_aggr=0.1$，如下图所示：

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p8.png" /> 
</div>


### 8.Extreme Instance Segmentation

极值点相比简单的边界框携带更多的关于对象的信息，其标注值至少是极值的两倍`(8 vs 4)`。我们提出一种使用极值点来近似对象掩码的简单方法，方法是创建一个以极值点为中心的八边形。具体地说，对于一个极值点，我们在其对应的边的两个方向上将其扩展到整个边长度的`1/4`的段。当它遇到一个角点时，线段被截断。然后我们把这四个部分的端点连接起来形成八边形。如下图所示：

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p9.png" /> 
</div>

为了进一步细化边界框分割，使用了`DeepExtremeCut[29]`，这是一个经过训练的深度网络，可以将手动提供的极值点转换为实例分割掩码。在本工作中，简单地将`DeepExtremeCut[29]`的手工输入替换为极值点预测，执行2个阶段的实例分割。具体地说，对于预测的每个边界框，需要裁剪边界框区域，使用预测的极值点绘制高斯图，然后将串联的图像提供给预先训练的DeepExtremeCut模型。DeepExtremeCut[29]是类别未知的，因此我们直接使用检测到的类别和ExtremeNet的得分，没有进一步的后处理。

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p10.png" /> 
</div>


### 9.Experiments

作者在MS COCO上做实验进行算法评估。在COCO数据集中没有直接的极值点标注，不过对于实例分割掩码有完整的标注，因此，可以在多边形掩码标注中找到作为极值点的点。如果多边形的一条边平行于轴，或者有小于$3^。$的误差，边缘的中点则为中心点。

**Training details**

本文是基于CornerNet实现的，沿用了CornerNet的超参数：

1）输入和输出分辨率分别为`511x511`、`128x128`；

2）数据增强使用了翻转、`0.6`到`1.3`的随机缩放、随机裁剪和随机颜色抖动；

3）使用`Adam`优化方式，学习率为`2.5e-4`；

4）CornerNet在10个GPU上训练了`500k`次迭代，相当于在一个`GPU`上训练`140`天。考虑到GPU资源的限制，作者在CornerNet预训练模型上微调网络，随机初始化head layers，在5个GPU上迭代`250k`次，`batch size`设为`24`，在`200k`次迭代时学习率缩降`10`倍。

**Testing details**

对于每个输入图像，网络为极点生成4个`c`通道的heatmaps，为中心点生成1个`c`通道heatmap，以及4个2通道的offset maps。本文将边缘聚合应用于每个极点的heatmap，并将中心点的heatmap乘以2，以修正整体尺度变化。然后将中心分组算法应用于heatmap，在ExtrectPeak中最多提取40个top点，以保持枚举效率。通过在offset maps的相应位置添加偏移量，以细化预测的边界框坐标。

与CornerNet一样，输入图片分辨率可以不同，不会resize到固定大小。测试时数据增强方式为图片翻转，在本文实验中，额外使用了`5`个多尺度`（0.5,0.75,1,1.25,1.5）`增强。最后，使用soft-NMS过滤检测结果。测试时，**一张图片耗时`322ms`,其中网络前向传播耗时`13ms`，分组耗时`150ms`，剩余时间消耗在NMS后处理上。**

下图为ExtremeNet和目前比较好的方法在COCO test-dev上的对比结果。
ExtremeNet在小目标和大小居中的目标上的效果比CornerNet要好，对于大目标，中心点响应图不够精确，这是因为几个像素的移动可能就会造成检测和false-negative之间的差异。

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p11.png" /> 
</div>

### 10.soft-NMS

soft-NMS来自于ICCV2017的文章，是NMS算法的改进(本教程R-CNN算法中有详细的关于NMS的介绍)，从论文题目可以看出，改进仅仅花了一行代码！首先NMS（Non maximum suppression）是在object detection算法中必备的后处理步骤，目的是用来去除重复框，也就是降低误检（false positives）。NMS算法的大致过程可以看原文这段话：First, it sorts all detection boxes on the basis of their scores. The detection box M with the maximum score is selected and all other detection boxes with a significant overlap (using a pre-defined threshold) with M are suppressed. This process is recursively applied on the remaining boxes.

那么传统的NMS算法存在什么问题呢？可以看下图。图中中，检测算法本来应该输出两个框，但是传统的NMS算法可能会把score较低的绿框过滤掉（如果绿框和红框的IOU大于设定的阈值就会被过滤掉），导致只检测出一个object（一个马），显然这样object的recall就比较低了。 

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p12.png" /> 
</div>

可以看出NMS算法是略显粗暴，因为NMS直接将和得分最大的box的IOU大于某个阈值的box的得分置零，那么有没有soft一点的方法呢？这就是本文提出Soft NMS。那么Soft-NMS算法到底是什么样呢？简单讲就是：An algorithm which decays the detection scores of all other objects as a continuous function of their overlap with M. 换句话说就是用稍低一点的分数来代替原有的分数，而不是直接置零。另外由于Soft NMS可以很方便地引入到object detection算法中，不需要重新训练原有的模型，因此这是该算法的一大优点。

下图是Soft-NMS算法的伪代码。首先是关于三个输入$B$、$S$、$N_t$，在下图中已经介绍很清楚了。$D$集合用来放最终的box，在boxes集合$B$非空的前提下，搜索score集合$S$中数值最大的数，假设其下标为$m$，那么$b_m$（也是M）就是对应的box。然后将$M$和$D$集合合并，并从$B$集合中去除$M$。再循环集合$B$中的每个box，这个时候就有差别了，如果是传统的NMS操作，那么当$B$中的box $b_i$和$M$的IOU值大于阈值Nt，那么就从$B$和$S$中去除该box；如果是Soft NMS，则对于$B$中的box $b_i$也是先计算其和$M$的IOU，然后该IOU值作为函数$f()$的输入，最后和box $b_i$的score $s_i$相乘作为最后该box $b_i$的score。就是这么简单！ 

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p13.png" /> 
</div>

接下来得重点就是如何确定函数$f()$了。 
首先NMS算法可以用下面的式子表示：

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p14.png" /> 
</div>

为了改变NMS这种hard threshold做法，并遵循IOU越大，得分越低的原则（IOU越大，越有可能是false positive），自然而然想到可以用下面这个公式来表示Soft-NMS： 

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p15.png" /> 
</div>

但是上面这个公式是不连续的，这样会导致box集合中的score出现断层，因此就有了下面这个Soft NMS式子（也是大部分实验中采用的式子）：

<div align=center>
<img src="zh-cn/img/anchorfree/extremenet/p16.png" /> 
</div>

这个式子满足了：A continuous penalty function should have no penalty when there is no overlap and very high penalty at a high overlap.



### 11.总结

优点：

延续CornerNet的检测新思路，将角点检测改为极值点检测，更加稳定，在muti-scale的测试下效果更好。

缺点：

1）无法规避的硬伤，ghost box 在实际场景中（并行的车辆）很可能出现。

2）从效果上看，并没有比CorNerNet有明显的提升。

3）速度很慢，主干网络计算量太大。


### Reference

[1].https://www.cnblogs.com/cieusy/p/10399960.html

[2].https://blog.csdn.net/chunfengyanyulove/article/details/99181340

[3].https://blog.csdn.net/weixin_39875161/article/details/93374834

[4].https://www.wandouip.com/t5i247334/

[5].https://zhuanlan.zhihu.com/p/57254154

[6].https://www.jianshu.com/p/68039cb1ba80

[7].https://blog.csdn.net/u014380165/article/details/79502197

[8].https://www.jianshu.com/p/8da5b4593a16

[9].https://arxiv.org/abs/1704.04503

[10].https://arxiv.org/abs/1901.08043

[11].https://github.com/xingyizhou/ExtremeNet