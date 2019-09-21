<div align=center>
<img src="zh-cn/img/anchorfree/p1.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/anchorfree/p2.png" /> 
</div>

## 1.Stacked Hourglass Networks for Human Pose Estimation

!> 论文地址：https://arxiv.org/abs/1603.06937

!> 引入改论文的目的是在`CornerNet`中我们会用到沙漏网络的backbone

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