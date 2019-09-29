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