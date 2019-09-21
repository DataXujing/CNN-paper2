## 神经风格转换

### 1.Visualizing and Understanding Convolutional NetWorks(ZFNet)

!> 期待更新

------

### 2.Deconvolutional Networks

!> 期待更新

------

### 3.VGG 16-19

!> 论文地址：https://arxiv.org/abs/1409.1556

2014年，牛津大学计算机视觉组(Visual Geometry Group)和Google DeepMind公司的研究员一起研发出了新的深度卷积神经网络：VGGNet，并取得了ILSVRC2014比赛分类项目的第二名(第一名是GoogLeNet，也是同年提出的)和定位项目的第一名。 

VGGNet探索了卷积神经网络的深度与其性能之间的关系，成功地构筑了16-19层深的卷积神经网络，证明了增加网络的深度能够在一定程度上影响网络最终的性能，使错误率大幅下降，同时拓展性又很强，迁移到其它图片数据上的泛化性也非常好。到目前为止，VGG仍然被用来提取图像特征。VGGNet可以看成是加深版本的AlexNet, 都是由卷积层、全连接层两大部分构成。

<div align=center>
<img src="zh-cn/img/chapter8/vgg/p1.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/chapter8/vgg/p2.png" /> 
</div>

以网络结构D（VGG16）为例，介绍其处理过程如下，请对比上面的表格和下方这张图，留意图中的数字变化，有助于理解VGG16的处理过程：


<div align=center>
<img src="zh-cn/img/chapter8/vgg/p3.png" /> 
</div>


1. 输入224x224x3的图片，经64个3x3的卷积核作两次卷积+ReLU，卷积后的尺寸变为224x224x64
2. 作max pooling（最大化池化），池化单元尺寸为2x2（效果为图像尺寸减半），池化后的尺寸变为112x112x64
3. 经128个3x3的卷积核作两次卷积+ReLU，尺寸变为112x112x128
4. 作2x2的max pooling池化，尺寸变为56x56x128
5. 经256个3x3的卷积核作三次卷积+ReLU，尺寸变为56x56x256
6. 作2x2的max pooling池化，尺寸变为28x28x256
7. 经512个3x3的卷积核作三次卷积+ReLU，尺寸变为28x28x512
8. 作2x2的max pooling池化，尺寸变为14x14x512
9. 经512个3x3的卷积核作三次卷积+ReLU，尺寸变为14x14x512
10. 作2x2的max pooling池化，尺寸变为7x7x512
11. 与两层1x1x4096，一层1x1x1000进行全连接+ReLU（共三层）
12. 通过softmax输出1000个预测结果

------

### 4.A Neural Algorithm of Artistic Style

!> 论文地址：https://arxiv.org/abs/1508.06576


卷积神经网络是深层神经网络中处理图像最强大的一个类别。卷积神经网络由一层层小的计算单元（神经元）组成，可以以前馈的方式分层地处理视觉上的信息（图1）。每一层中的计算单元（神经元）可以被理解为是对过滤图像信息的收集，也就是说，每一个神经元都会从输入的图像中抽取某个特征。因此，每层的输出是由所谓的feature map组成，它们是对输入的图像进行不同类型的过滤得到的。（也就是说每个神经元都会关注图像的某个特征）

当卷积神经网络被训练用于物体识别时，会生成一个图像的表征(representations) ，随着处理层级的上升，物体的信息越来越明确。因此，随着神经网络中的层级一级一级地被处理，输入的图像会被转换成一种表征，与图片的像素细节相比，这种表征会越来越关注图片的实际内容。通过对某一层的提取出来的feaure map的重塑，我们可以直接看到该层包含的图片信息。层级越高，那么获取的图像中物体内容就越高质量，并且没有确切的像素值的约束（层级越高，像素丢失越多）。相反，在低层级中重塑的话，其实像素丢失地很少。所以我们参考的是神经网络高层的特征，用它来作为图片内容的表征。（因为我们要得到更多内容，更少像素）–内容表征

为了获取输入图像的风格表征，我们用一个特征空间去捕获纹理的信息。这个特征空间建立在每层神经网络的过滤响应之上（也就是上面提到的feature map)。在feature map的空间范围上(也就是同一层上的feature map)，过滤响应各有不同（feature map关注的特征不同），而这个特征空间就是由这些差异构成。对每一层featute map两两求相关性，我们会获得一个静态的，多尺度的图像表征，它捕获了纹理的信息（但这纹理信息并非全局的）。–风格表征

上面三段简而言之就是讲了三句话： 
1. 每个卷基层是有多个神经元组成，每个神经元输出的是一个feature map。 
2. 神经网络较高层输出的一组feature map是内容表征。 
3. 神经网络某一层输出的一组feature map，使他们两两求相关性，这个相关性就是风格表征。

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p1.png" /> 
</div>

*图1：卷积神经网络（CNN）: 
一张输入的图片，会在卷积神经网的各层以一系列过滤后的图像表示。随着层级的一层一层处理，过滤后的图片会通过向下取样的方式不断减小（比如通过池化层）。这使得每层神经网的神经元数量会原来越小。（也就是层越深，因为经过了池化层，单个feature map会越来越小，于是每层中的神经元数量也会越来越少） 
内容重塑 :
在只知道该层的输出结果，通过重塑输入图像，我们可以看到CNN不同阶段的图像信息。我们在原始的VGG-Network上的5个层级:conv1_1,conv1_2,conv1_3,conv1_4,conv1_5上重塑了输入的图像。 
输入的图像是上图中的一排房子，5个层级分别是a,b,c,d,e。 
我们发现在较低层的图像重构（abc）非常完美；在较高层（de），详细的像素信息丢失了。也就是说，我们提取出了图片的内容，抛弃了像素。 风格重塑 : 在原始的CNN表征之上(feature map)，我们建立了一个新的特征空间(feature space)，这个特征空间捕获了输入图像的风格。风格的表征计算了在CNN的不同层级间不用特征之间的相似性。通过在CNN隐层的不同的子集上建立起来的风格的表征，我们重构输入图像的风格。如此，便创造了与输入图像一致的风格而丢弃了全局的内容。*


于是，同样，我们也可以在CNN的各层中利用风格特征空间所捕获的信息来重构图像。事实上，重塑风格特征就是通过捕获图片的颜色啊结构啊等等生产出输入的图像的纹理的版本。另外，随着层级的增加，图像结构的大小和复杂度也会增加。我们将这多尺度的表征称为风格表征。

本文关键的发现是对于内容和风格的表征在CNN中是可以分开的。我们可以独立地操作两个表征来产生新的，可感知意义的图像。为了展示这个发现，我们生成一个图像，这个图像混合了来自两个不同图像的内容和风格表征。确切的说，我们将著名艺术画“星空”的风格，和一张德国拍的照片的内容混合起来了。 
我们寻找这样一张图片，它同时符合照片的内容表征，和艺术画的风格表征。原始照片的整体布局被保留了，而颜色和局部的结构却由艺术画提供。如此一来，原来的那张风景照旧像极了艺术作品。

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p2.png" /> 
</div>

*图2： 图中描述的是同一张风景照的内容，融合来自不同的风景画的风格的图片。*


风格表征是一个多尺度的表征，包括了神经网络的多层。在图2中看到的图像，风格的表征包含了整个神经网络的层级。而风格也可以只包含一小部分较低的层级。（见下面的图3，第一行是卷基层1，第5行是卷基层5的输出）。若符合了较高层级中的风格表征，局部的图像结构会大规模地增加，从而使得图像在视觉上更平滑与连贯。因此，看起来美美的图片通常是来自于符合了较高层级的风格表征。

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p3.png" /> 
</div>

当然啦，图像的内容和风格并不能被完全地分解开。当风格与内容来自不同的两个图像时，这个被合成的新图像并不存在在同一时刻完美地符合了两个约束。但是，在图像合成中最小化的损失函数分别包括了内容与风格两者，它们被很好地分开了。所以，我们可以平滑地将重点既放在内容上又放在风格上（可以从图3的一列中看出）。将重点过多地放在风格上会导致图像符合艺术画的外观，有效地给出了画的纹理，但是几乎看不到照片的内容了。而将重点过多地放在内容上，我们可以清晰地看到照片，但是风格就不那么符合艺术画了。因此，我们要在内容与风格之间调整trade-off，这样才能创造出美美的画。

在之前的研究中，是通过评估复杂度小很多的感官输入来将内容与风格分离的。比如说通过不同的手写字，人脸图，或者指纹。 
而在我们的展示中，我们给出了一个有着著名艺术作品风格的照片。这个问题常常会更靠近与计算机视觉的一个分支–真实感渲染。理论上更接近于利用纹理转换来获取艺术风格的转换。但是，这些以前的方法主要依赖于非参数的技术并且直接对图像表征的像素进行操作。相反，通过在物体识别上训练深度神经网了，我们在特征空间上进行相关操作，从而明确地表征了图像的高质量内容。

神经网络在物体识别中产生的特征先前就已经被用来做风格识别，为的是根据艺术作品的创作时期来为作品分类。分类器是在原始的网络上被训练的，也就是我们现在叫的内容表征。我们猜测静态特征空间的转换，比如我们的风格表征也许可以在风格分类上有更好的表现。

通常来说，我们这种合成图像的方法提供了一个全新的迷人的工具用于学习艺术，风格和独立于内容的图像外观的感知与神经表征。总之，一个神经网络可以学习图像的表征，是的图像内容与风格的分离成为可能，是如此激动人心。若要给出解释的话，就是当学习物体识别到时候，神经网络对所有图像的变化都能保持不变从而保留了物体的特性。

**方法**

上文展示的结果是依赖于卷积神经网络–VGG神经网络模型产生的。我们使用由19层的VGG神经网络（16个卷积和5个池化层）提供的特征空间。并且这个神经网络中没有一个是全链接的。这个模型是可以被公开获取的，并且可以caffe这个深度学习的框架中被调用。对于图像合成，我们发现用均值池化层代替最大值池化层会提高梯度流，并且得到更加完美的结果。所以本案例中我们用的是均值池化层。–模型概述

每一层神经网络定义了一个非线性的过滤器（这里所说的过滤器就是神经元），这个过滤器的复杂度随着隐层的位置而增加。因此，给定一个输入的图像$x$, 在CNN的每层都会被过滤器编码。一个有$N_t$个不同的过滤器的隐层有$N_t$个feature map（每个神经元输出一个feature map)。每个feature map的大小是$M_t$，$M_t$是feature map高乘以宽的大小。所以一个层的输出可以存储为矩阵：
$$F^l \in R^{N_lXM_l}$$
$F_{ij}^l$表示在l层的位置j上的第i个过滤器的激活结果。为了可视化不同层级中的图像信息，我们在一个白噪声上使用梯度下降来找到另一个图像，它与原始图像的特征输出结果相符合(白噪声上的图像其实就是定义一个随机的新图，然后通过梯度下降不断迭代，不断更新这个新图）。所以让$p$和$x$作为原始图像和后来产生的图像，$P^l$和$F^l$是他们在l层各自的特征表征。然后我们定义两个特征表征之间的平方误差损失。 （也就是，输出的结果是`14*14*256`的矩阵，256是该层神经元的个数，`14*14`是feature map,将content image与新图都走一遍这个CNN, 他们各自会生成以上矩阵，也就是$P$和$F$，将这两个矩阵在对应的位置求平方误差和，就是内容上的损失函数。乘以1/2是为了求导方便）

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p4.png" /> 
</div>

这个损失函数的导数是：（针对F求导） 

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p5.png" /> 
</div>

以上公式中，图像$x$的梯度可以通过标准误差的后向计算传播。因此我们可以改变初始的随机图像$x$直到它产生了在CNN中与原始图像$p$一样的输出结果。在图1中的5个内容重构来自于原始VGG的

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p6.png" /> 
</div>

另外，我们通过计算不同过滤器输出结果之间的差异，来计算风格相似度。我们期望获得输入图片空间上的衍生。这些特征的相似性用Gram matrix $G^l \in R^{N_l\timesN_l}$给出。$G_{ij}^l$是来自于l层中矢量的feature mao i和j之间。(直译心好累，解释一下上面讲的，就是将艺术画也放进CNN中，比如输出也是`14*14*256`的一个矩阵，然后将256个`14*14`的feature map两两求相似性，这里是两两相乘，于是会得带`256*256`的一个特征空间矩阵，G就是这个特征空间）

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p7.png" /> 
</div>

为了生成符合给定艺术作品风格的纹理，我们对一个带有白噪声的图像（也就是我们定义的随机的新图）做梯度下降，从而去寻找另一个图像，使得这个图像符合艺术画的风格表征。而这个梯度下降的过程是通过使得原始图像（艺术画）的Gram矩阵和被生成的图像（新图）的Gram矩阵的距离的均方误差最小化得到的。因此, 令$a$和$x$分别作为原始艺术图像与被生成的图像，$A^l$和$G^l$分别作为l层的两个风格表征。l层对于总损失的贡献是： 

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p8.png" /> 
</div>

而总损失用公式表达为：

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p9.png" /> 
</div>

$w_t$表示每一层对于总损失的贡献的权重因子。$E_t$的导数可以这样计算：

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p10.png" /> 
</div>

$E_l$的在低层级的梯度可以很方便地计算出来，通过标准误差后向传播。在图1中5个风格的重塑可以通过满足一下这些层的风格表征来生成：
`‘conv1 1’ (a), ‘conv1 1’ and ‘conv2 1’
(b), ‘conv1 1’, ‘conv2 1’ and ‘conv3 1’ (c), ‘conv1 1’, ‘conv2 1’, ‘conv3 1’ and ‘conv4 1’ (d),
‘conv1 1’, ‘conv2 1’, ‘conv3 1’, ‘conv4 1’ and ‘conv5 1’ (e).`

为了生成混合了照片内容和艺术画风格的新图像，我们需要联合最小化风格损失与内容损失：

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p11.png" /> 
</div>

`α`和`β`分别是内容和风格在图像重构中的权重因子。`α`和`β`分别是内容和风格两个损失的权重。`α+β=1`.如果`α`比较大，那么输出后的新图会更多地倾向于内容上的吻合，如果`β`较大，那么输出的新图会更倾向于与风格的吻合。这两个参数是一个trade-off,可以根据自己需求去调整最好的平衡。论文的作者给出了它调整参数的不同结果，如下图，从左到右四列分别是`α/β = 10^-5, 10^-4,10^-3, 10^-2`.也就是`α`越来越大，的确图像也越来越清晰地呈现出了照片的内容。 


<div align=center>
<img src="zh-cn/img/chapter8/nst1/p12.png" /> 
</div>

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p13.png" /> 
</div>


**Pytorch Code**

```python
#build_model.py
import torch
import torch.nn as nn
import torchvision.models as models

import loss

vgg = models.vgg19(pretrained=True).features
if torch.cuda.is_available():
    vgg = vgg.cuda()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_loss(style_img,
                             content_img,
                             cnn=vgg,
                             style_weight=1000,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):

    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential()
    if torch.cuda.is_available():
        model = model.cuda()
    gram = loss.Gram()
    if torch.cuda.is_available():
        gram = gram.cuda()

    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            if name in content_layers_default:
                target = model(content_img)
                content_loss = loss.Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                style_loss = loss.Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    return model, style_loss_list, content_loss_list

```

```python
#loss.py
import torch.nn as nn
import torch


class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram


class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        G = self.gram(input) * self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
```

```python
# load_img.py
import PIL.Image as Image
import torchvision.transforms as transforms

img_size = 512


def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()
```

```python
#run_code.py

import torch.nn as nn
import torch.optim as optim

from build_model import get_style_model_and_loss


def get_input_param_optimier(input_img):
    """
    input_img is a Variable
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(
        style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)

    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:

        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)

    return input_param.data


```

```python
#demo.py
from torch.autograd import Variable
from torchvision import transforms
from run_code import run_style_transfer
from load_img import load_img, show_img
from torch.autograd import Variable

style_img = load_img('./picture/style.png')
style_img = Variable(style_img).cuda()
content_img = load_img('./picture/content.jpg')
content_img = Variable(content_img).cuda()

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)

show_img(out.cpu())

save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))

save_pic.save('./picture/saved_picture.png')
```

运行后的效果图展示：

<div align=center>
<img src="zh-cn/img/chapter8/nst1/p14.png" /> 
</div>


<div align=center>
<img src="zh-cn/img/chapter8/nst1/p15.png" /> 
</div>

------


### 5.Perceptual Losses for Real-Time Style Transfer and Super-Resolution (Fast Neural Style Transfer)

!> 基于感知损失函数的实时风格转换和超分辨率重建

!> 论文地址：https://arxiv.org/abs/1603.08155

#### 0. Abstract

我们考虑图像转换的问题，即将一个输入图像变换成一个输出图像。最近热门的图像转换的方法通常是训练前馈卷积神经网络，将输出图像与原本图像的逐像素差距作为损失函数。并行的工作表明，**高质量的图像可以通过用预训练好的网络提取高级特征、定义并优化感知损失函数来产生**。我们组合了一下这两种方法各自的优势，提出采用感知损失函数训练前馈网络进行图像转换的任务。本文给出了图像风格转化的结果，训练一个前馈网络去解决实时优化问题（Gatys等人提出的），和基于有优化的方法对比，**我们的网络产生质量相当的结果，却能做到三个数量级的提速**。我们还实验了单图的超分辨率重建，同样采用感知损失函数来代替求逐像素差距的损失函数

关键词：风格转换，超分辨率重建，深度学习

#### 1.Introduction

许多经典问题可以被分为图像转换任务，即一个系统接收到一些输入图像，将其转化成输出图像。用图像处理来举例，比如图像降噪，超分辨率重建，图像上色，这都是输入一个退化的图像（噪声，低分辨率，灰度），输出一个高质量的彩色图像。从计算机视觉来举例，包括语义分割，深度估计，其中的输入是一个彩色图像，输出是图像对场景的语义或几何信息进行了编码。

一个处理图像转换任务的方法是在有监督模式下训练一个前馈卷积神经网络，用逐像素差距作损失函数来衡量输出图像和输入图像的差距。这个方法被Dong等人用来做了超分辨率重建，被Cheng等人做了图像上色，被Long等人做了图像分割，被Eigen等人做了深度和表面预测。这个方法的优势在于在测试时，只需要一次前馈的通过已训练好的网络。

然而，**这些方法都用了逐像素求差的损失函数，这个损失函数无法抓住输入及输出图像在感知上的差距**。举个例子，考虑两张一模一样的图像，只有1像素偏移上的差距，尽管从感知上这俩图片一模一样，但用逐像素求差的方法来衡量的话，这俩图片会非常的不一样。

同时，**最近的一些工作证明，高质量的图像可以通过建立感知损失函数（不基于逐像素间的差距，取而代之的是从预训练好的CNN中提取高层次的图像特征来求差）图像通过使损失函数最小化来生成**，这个策略被应用到了特征倒置[6]（Mahendran等），特征可视化[7] (Simonyan等/Yosinski等)，纹理综合及图像风格化[9,10] (Gatys等)。这些方法能产生很高质量的图片，**不过很慢，因为需要漫长的迭代优化过程**。

在这篇论文中，我们结合了两类方法的优势。我们训练一个用于图像转换任务的前馈网络，且不用逐像素求差构造损失函数，转而使用感知损失函数，从预训练好的网络中提取高级特征。在训练的过程中，感知损失函数比逐像素损失函数更适合用来衡量图像之间的相似程度，在测试的过程中，生成器网络能做到实时转换。

我们实验了两个任务，图像风格转化和单图的超分辨率重建。这两种都有天生的缺陷：图像风格化没有唯一正确的输出，超分辨率重建的话，我们可以从一个低分辨率图像重建出很多高分辨率的图像。比较好的是，这两个任务都需要对输入的图像进行语义上的理解。图像风格化中，输出图片从语义维度来看必须跟输入图像比较接近，尽管颜色和纹理会发生质的变化。超分辨率重建任务中，必须从视觉上模糊的低分辨率输入来推断出新的细节。原则上，一个为任何任务训练的高质量的神经网络应该能隐式的学习输入图像的相关语义；然而在实践中我们不需要从头开始学习：使用感知损失函数，允许从损失网络直接转移语义信息到转换网络。

<div align=center>
<img src="zh-cn/img/ncs2/p1.png" /> 
</div>

*图1：我们的结果，第一行是风格化，第二行是4倍的超分辨率重建*

对于图像风格化，我们的前馈网络用来解决优化问题[10]；我们的结果跟[10]中很相似（无论是质量还是目标函数的值），但能达成3个数量级的速度飞升。对于超分辨率重建，我们证实：把逐像素求差损失函数改成感知损失函数，能带来视觉享受级的4倍和8倍超分辨率重建。

#### 2.Related Work

**前馈图像转换**：最近几年前馈图像转换任务应用十分广泛，很多转换任务都用了逐像素求差的方式来训练深度卷积神经网络。

语义分割的方法[3,5,12,13,14,15]产生了密集的场景标签，通过在在输入图像上以全卷积的方式运行网络，配上逐像素分类的损失函数。[15]跨越了逐像素求差，通过把CRF当作RNN，跟网络的其他部分相加训练。我们的转换网络的结构是受到[3]和[14]的启发，使用了网络中下采样来降低特征图谱的空间范围，其后紧跟一个网络中上采样来产生最终的输出图像。

最近的方法在深度估计[5,4,16]和表面法向量估计[5,17]上是相似的，它们把一张彩色输入图像转换成有几何意义的图像，是用前馈神经网络，用逐像素回归[4,5]或分类[17]的损失函数。一些方法把逐像素求差改换成了惩罚图像梯度或是用CRF损失层来强制促使输出图像具有一致性。[2]中一个前馈模型用逐像素求差的损失函数训练，用于将灰度图像上色。

**感知的优化：**有一定数量的论文用到了优化的方法来产生图像，它们的对象是具有感知性的，感知性取决于从CNN中提取到的高层次特征。图像可以被生成用于最大限度提升分类预测的分数[7,8]，或是个体的特征[8]用来理解训练网络时的函数编码。相似的优化技巧同样可以用于产生高可信度的迷惑图像[18,19]。

Mahendran和Vedaldi从卷积网络中反转特征，通过最小化特征重建损失函数，为了能理解保存在不同网络层中的图像信息；相似的方法也被用来反转局部二进制描述符[20]和HOG特征[21].

Dosovitskiy和Brox的工作是跟我们的最相关的，它们训练了一个前馈神经网络去倒置卷积特征，快速的逼近了[6]提出的优化问题的结局方案，然而他们的前馈网络是用的逐像素重建损失函数来训练，而我们的网络是直接用了[6]用的特征重建损失函数。

**风格转换：**Gatys等人展示艺术风格转换，结合了一张内容图和另一张风格图，通过最小化根据特征重建的代价函数，风格重建用的代价函数也是基于从预训练模型中提取的高级特征；一个相似的方法之前也被用于做纹理合成。他们的方法产出了很高质量的结果，不过计算代价非常的昂贵因为每一次迭代优化都需要经过前馈、反馈预训练好的整个网络。为了克服这样一个计算量的负担，我们训练了一个前馈神经网络去快速获得可行解。

**图像超分辨率重建**。图像超分辨率重建是一个经典的问题，很多人提出了非常广泛的技术手段来做图像超分辨率重建。Yang等人提供了一个对普通技术的详尽评价，在广泛采用CNN之前，它们把超分辨率重建技术归类成了一种基于预测的方法.(bilinear, bicubic, Lanczos, [24]), 基于边缘的方法[25,26] ，统计的方法[27,28,29]，基于块的方法[25,30,31,32,33] ，稀疏字典方法[37, 38]。最近在单图超分辨率放大方向取得成就的表现是用了三层卷积神经网络，用逐像素求差的方式算损失函数。其他一些有艺术感的方法在[39,40,41]

#### 3.Method

像*图2*中展示的那样，我们的系统由两部分组成：一个图片转换网络`fw `和一个损失网络` φ`（用来定义一系列损失函数$l_1,l_2,l_k$），图片转换网络是一个**深度残差网络**，参数是权重`W`，它把输入的图片`x`通过映射 `y=fw(x)`转换成输出图片$\hat{y}$，每一个损失函数计算一个标量值$l_i(\hat{y},y_i)$, 衡量输出的$\hat{y}$和目标图像$y_i$之间的差距。图片转换网络是用`SGD`训练，使得一系列损失函数的加权和保持下降。

<div align=center>
<img src="zh-cn/img/ncs2/p2.png" /> 
</div>

*图2：系统概览。左侧是Generator，右侧是预训练好的vgg16网络（一直固定）*

<div align=center>
<img src="zh-cn/img/ncs2/p3.png" /> 
</div>

为了明确逐像素损失函数的缺点，并确保我们的损失函数能更好的衡量图片感知及语义上的差距，我们从最近的优化迭代生成图片的系列工作中得到了灵感[6,7,8,9,10]，这些方法共同的关键点在于CNN是预先训练好用于图像分类的，这个CNN已经学会感知和语义信息编码，这正是我们希望在我们的损失函数中做的。所以我们用了一个预训练好用于图像分类的网络`φ`，来定义我们的损失函数。之后使用同样是深度卷积网络的损失函数来训练我们的深度卷积转换网络。

损失网络`φ`是能定义一个特征（内容）损失$l_{feat}^{\phi}$和一个风格损失$l_{style}^{\phi}$，分别衡量内容和风格上的差距。对于每一张输入的图片x我们有一个内容目标$y_c$一个风格目标$y_s$，对于风格转换，内容目标$y_c$是输入图像`x`，输出图像`y`，应该把风格$y_s$结合到内容$x=y_c$上。我们为每一个目标风格训练一个网络。对于单图超分辨率重建，输入图像`x`是一个低分辨率的输入，目标内容是一张真实的高分辨率图像，风格重建没有使用。我们为每一个超分辨率因子训练一个网络。

##### 3.1 图像转换网络

我们的图像转换网络结构大致上遵循Radford提出的指导方针[42]。我们不用任何的池化层，取而代之的是用步幅(strided)卷积或微步幅(fractionally strided)卷积（[http://www.jiqizhixin.com/article/1417](https://link.jianshu.com?t=http://www.jiqizhixin.com/article/1417)）做网络内的上采样或者下采样。我们的神经网络有五个残差块[42]组成，用了[44]说的结构。所有的非残差卷积层都跟着一个空间性的`batch-normalization`[45]，和`RELU`的非线性层，最末的输出层除外。最末层使用一个缩放的`Tanh`来确保输出图像的像素在[0,255]之间。除开第一个和最后一个层用`9x9`的`kernel`，其他所有卷积层都用`3x3`的`kernels`。

**输入和输出：**对于风格转换，输入和输出都是彩色图片，大小`3x256x256`。对于超分辨率重建，有一个上采样因子`f`，输出是一个高分辨率的图像`3x288x288`，输入是一个低分辨率图像 `3x288/fx288/f`，因为图像转换网络是全卷积，所以在测试过程中它可以被应用到任何分辨率的图像中。

**下采样和上采样**：对于超分辨率重建，有一个上采样因子`f`，我们用了几个残差块跟着$\log_2f$卷积网络（`stride=1/2`）。这个处理和[1]中不一样，[1]在把输入放进网络之前使用了双立方插值去上采样这个低分辨率输入。不依赖于任何一个固定的上采样插值函数，微步长卷积允许上采样函数和网络的其他部分一起被训练。

<div align=center>
<img src="zh-cn/img/ncs2/p4.png" /> 
</div>

*图3，和[6]相似，我们用了优化的方式去找一个图像y，能使得针对某些层的特征（内容）损失最小化，使用了预训练好的vgg16网络，在我们用较高层重建的时候，图像的内容和空间结构被保留了，但是颜色，纹理和精确的形状改变了。*

对于风格转换，我们的网络用了两个`stride=2`的卷积去下采样输入，紧跟着的是几个残差块，接下来是两个卷积层（`stride=1/2` 转置卷积）来做上采样。虽然输入和输出有着相同的大小，但是先下采样再上采样的过程还是有一些其他好处。

首当其冲的好处是计算复杂性。用一个简单的实现来举例，一个`3x3`的卷积有`C`个fiters，输入尺寸`C x H x W`需要$9HWC^2$ 的乘加，这个代价和`3x3`卷积有$DC$个`filter`，输入尺寸`DCxH/DxW/D`是一样的。在下采样之后，我们可以因此在相同计算代价下用一个更大的网络。

第二个好处是有效的感受野大小。高质量的风格转换需要一致的改变图片的一大块地方；因此这个优势就在于在输出中的每个像素都有输入中的大面积有效的感受野。除开下采样，每一个附加的`3x3`卷积层都能把感受野的大小增加`2`倍，在用因子`D`进行下采样后，每个`3x3`的卷积不是增加了感受野的大小到`2D`，给出了更大的感受野大小但有着相同数量的层。

残差连接：He[43]等人用了残差连接去训练非常深的网络用来做图像分类，它们证明了残差连接能让网络更容易的去学习确定的函数，这在图像转换网络中也是一个很有吸引力的研究，因为在大多数情况下，输出图像应该和输入图像共享结构。因此我们网络的大体由几个残差块组成，每个包含两个`3x3`的卷积层，我们用[44]中设计的残差块，在附录中有。

##### 3.2 感知损失函数

我们定义了两个感知损失函数，用来衡量两张图片之间高级的感知及语义差别。要用一个预训练好用于图像分类的网络模型。在我们的试验中这个模型是`VGG16`[46]，使用`Imagenet`的数据集来做的预训练。

<div align=center>
<img src="zh-cn/img/ncs2/p5.png" /> 
</div>


*图4: 和[10]一样，我们用了优化的方式去找到一张图y，最小化从VGG16的某几层取出来的风格损失。图像y只保存风格特征不保存空间结构。*

**特征（内容）损失：**我们不建议做逐像素对比，而是用VGG计算来高级特征（内容）表示，这个取法和那篇artistic style使用VGG-19提取风格特征是一样的，公式：

<div align=center>
<img src="zh-cn/img/ncs2/p6.png" /> 
</div>

如在[ 6 ]和在图3重现的，找到一个图像 `Y`使较低的层的特征损失最小，往往能产生在视觉上和`y`不太能区分的图像，如果用高层来重建，内容和全局结构会被保留，但是颜色纹理和精确的形状不复存在。用一个特征损失来训练我们的图像转换网络能让输出非常接近目标图像`y`，但并不是让他们做到完全的匹配。

**风格损失：** 特征（内容）损失惩罚了输出的图像（当它偏离了目标y时），所以我们也希望去惩罚风格上的偏离：颜色，纹理，共同的模式，等方面。为了达成这样的效果Gatys等人提出了以下风格重建的损失函数。

让$\phi_j(x)$代表网络`φ`的第`j`层，输入是`x`。特征图谱的形状就是$C_j x H_j x W_j$、定义矩阵$G_j(x)$为$C_j x C_j$矩阵（特征矩阵）其中的元素来自于：

<div align=center>
<img src="zh-cn/img/ncs2/p7.png" /> 
</div>



如果我们把$\phi_j(x)$理解成一个$C_j$维度的特征，每个特征的尺寸是$H_j x W_j$，那么上式左边$G_j(x)$就是与$C_j$维的非中心的协方差成比例。每一个网格位置都可以当做一个独立的样本。这因此能抓住是哪个特征能带动其他的信息。梯度矩阵可以很高效的被计算，通过调整$\phi_j(x)$的形状为一个矩阵`ψ`，形状为$C_j x H_jW_j$，然后$G_j(x)$就是$\PhiPhi^T/C_jH_jW_j$。

风格重建的损失是定义的很好的，甚至当输出和目标有不同的尺寸是，因为有了梯度矩阵，所以两者会被调整到相同的形状。

就像[10]中介绍的，如图5重建，能生成一张图片y使得风格损失最小，从而保存了风格上的特征，但是不保存空间上的结构特征。

为了表示从一个集合层的风格重建，而不是由单层重建，我们把$l_{style}^{\phi}(\hat{y},y)$定义成一个损失的集合（针对每一个层的损失求和）。

<div align=center>
<img src="zh-cn/img/ncs2/p8.png" /> 
</div>

##### 3.3简单损失函数

除了感知损失，我们还定义了两种简单损失函数，仅仅用了低维的像素信息

**像素损失：**像素损失是输出图和目标图之间标准化的差距。如果两者的形状都是`CxHxW`,那么像素损失就是$l_{pixel}(\hat{y},y)=|\hat{y}-y|^{2}/CHW$。这只能被用在我们有完全确定的目标，让这个网络去做完全匹配。

**全变差正则化：**为使得输出图像比较平滑，我们遵循了前人在特征反演上的研究[6,20]，超分辨率重建上的研究[48,49]并且使用了全变差正则化$l_{TV}(\hat{y})$。（全变差正则化一般用在信号去噪）

#### 4.Experiments

我们实验了两个图像变换任务：风格转换和单图超分辨率重建。风格转换中，前人使用优化来生成的图像，我们的前馈网络产生类似的定性结果，但速度快了三个数量级。单图像超分辨率中，用了卷积神经网络的都用的逐像素求差的损失，我们展示了令人振奋的的有质量的结果，通过改用感知损失。

##### 4.1风格转换

风格转换的目标是产生一张图片，既有着内容图的内容信息，又有着风格图的风格信息，我们为每一种风格训练了一个图像转换网络，这几种风格图都是我们手工挑选的。然后把我们的结果和基础Gatys的结果做了对比。

<div align=center>
<img src="zh-cn/img/ncs2/p9.png" /> 
</div>

**基线：**作为基线，我们重现了Gatys等人的方法，给出风格和内容目标$y_s$和$y_c$，层$j$和$J$表示特征和风格重建。$\hat{y}$通过解决下述问题来获得。

<div align=center>
<img src="zh-cn/img/ncs2/p10.png" /> 
</div>

`λ`开头的都是参数，`y`初始化为白噪声，用`L-BFGS`优化。我们发现，无约束的优化方程通常会导致输出图片的像素值跑到`[0,255]`之外，做一个更公平的比较，对基线，我们用`L-BFGS`投影，每次迭代都把图片y调整到`[0,255]`，在大多数情况下，运算优化在500次迭代之内收敛到满意的结果，这个方法比较慢因为每一个`L-BFGS`迭代需要前馈再反馈通过VGG16网络$\phi$。

**训练细节：** 我们的风格转换网络是用COCO数据集训练的，我们调整每一个图像到`256x256`，共`8`万张训练图，`batch-size=4`，迭代`40000`次，大约跑了两轮。用`Adam`优化，初始学习速率`0.001.`输出图被用了全变量正则化（`strength 在1e-6到1e-4之间`），通过交叉验证集选择。不用权重衰减或者dropout，因为模型在这两轮中没有过拟合。对所有的风格转换实验我们取`relu2_2`层做内容，`relu1_2，relu2_2，relu3_3和relu4_3`作为风格。VGG-16网络，我们的实验用了`Torch`和`cuDNN`，训练用了大约`4`个小时，在一个`GTX Titan X GPU`上。

**定性结果：** 在图6中我们展示了结果的对比，比较了我们的结果和那些基础方法，用了一些风格和内容图。所有的参数`λ`都是一样的，所有的训练集都是从MS-COCO2014验证集里抽选的。我们的方法能达到和基本方法一样的质量。

尽管我们的模型是用`256x256`的图片训练的，但在测试时候可以用在任何图像上面，在图7中我们展示了一些测试用例，用我们的模型训练`512`大小的图片

<div align=center>
<img src="zh-cn/img/ncs2/p11.png" /> 
</div>

*图6: 用我们的图像生成网络做图像风格转换。我们的结果和Gatys相似，但是更快（看表1）。所有生成图都是256x256的*

<div align=center>
<img src="zh-cn/img/ncs2/p12.png" /> 
</div>

*图7: 我们的网络在512x512图上的测试样例，模型用一个全卷积操作来达成高分辨率的图像（测试时），风格图和图6一样。*

通过这些结果可以明确的是，风格转换网络能意识到图像的语义内容。举个例子，在图7中的海滩图像，人们是很明显的被识别了出来，但背景被风格扭曲了；同样的，猫脸很明显的被识别了出来，但他的身体并没有被识别出来。一个解释是：VGG16网络是被训练用来分类的，所以对于图片的主体（人类和动物）的识别要比那些背景保留完整的多。

**定量结果：** 基本方法和我们的方法都是使公式5最小化。基本方法针对一张图进行明确的优化（针对要输出的图像）我们的方法训练一个解决方案（能在前馈中处理任意一张图片$y_c$）我们可以量化的比较这两种方法，通过衡量它们成功减少代价函数的幅度。（公式5）

我们用我们的方法和它们的方法一起训练了五十张图片（从MSCOCO验证集中得到）使用The Muse by Pablo Picasso当作一个风格图。对于基础方法我们记录了函数在每一个迭代过程的值。对我们的方法我们对每一张图片记录了公式5的值。我们还计算了公式5的值，当y和输出图像$y_c$相等时，结果显示在图5，我们看到内容图$y_c$达到了非常高的损失，和我们的方法在50-100之间差不多。

尽管我们的网络用`256x256的`尺寸训练的，但他们在`512,1024`的情况下都能成功的使代价函数最小化，结果展示在表图5中。我们可以看到哪怕在高分辨率下，和普通方法达成相同损失的时间也差不多。

**速度：** 在表1中我们比较了运行的时间（我们的方法和基础方法）对于基础方法，我们记录了时间，对所有的图像大小比对，我们可以看出我们方法的运行时间大致是基本方法迭代一次时间的一半。跟基本方法500次迭代的相比，我们的方法快了三个数量级。我们的方法在·20fps·里产生·512x512·的图片，让他可能应用在实时图像转换或者视频中。

<div align=center>
<img src="zh-cn/img/ncs2/p13.png" /> 
</div>

##### 4.2 单图超分辨率重建

在单图超分辨率重建中，任务是从一个低分辨率的输入，去产生一个高分辨率的输出图片。这是一个固有的病态问题，因为对一个低分辨率图像，有可能对应着很多种高分辨率的图像。当超分辨率因子变大时，这个不确定性会变得更大。对于更大的因子（`x4 x8`），高分辨率图像中的好的细节很可能只有一丁点或者根本没有出现在它的低分辨率版本中。

为了解决这个问题，我们训练了超分辨率重建网络，不使用过去使用的逐像素差损失函数，取而代之的是一个特征重建损失函数（`看section 3`）以保证语义信息可以从预训练好的损失网络中转移到超分辨率网络。我们重点关注`x4`和`x8`的超分辨率重建，因为更大的因子需要更多的语义信息。

传统的指标来衡量超分辨率的是`PSNR`和`SSIM`，两者都和人类的视觉质量没什么相关的[55,56,57,58,59].`PSNR`和`SSIM`仅仅依赖于像素间低层次的差别，并在高斯噪声的相乘下作用，这可能是无效的超分辨率。另外的，`PSNR`是相当于逐像素差的，所以用`PSNR`衡量的模型训练过程是让逐像素损失最小化。因此我们强调，这些实验的目标并不是实现先进的`PSNR`和`SSIM`结果，而是展示定性的质量差别（逐像素损失函数vs感知损失）

**模型细节：**我们训练模型来完成`x4`和`x8`的超分辨率重建，通过最小化特征损失（用vgg16在`relu2_2`层提取出），用了`288x288`的小块（`1`万张MSCOCO训练集），准备了低分辨率的输入，用高斯核模糊的（`σ=1.0`）下采样用了双立方插值。我们训练时`bacth-size=4`，训练了`20`万次，`Adam`，学习速率`0.001`，无权重衰减，无dropout。作为一个后续处理步骤，我们执行网络输出和低分辨率输入的直方图匹配。

**基础：**基本模型我们用的` SRCNN`[1] 为了它优秀的表现，`SRCNN`是一个三层的卷积网络，损失函数是逐像素求差，用的ILSVRC2013数据集中的`33x33`的图片。SRCNN没有训练到`x8`倍，所以我们只能评估`x4`时的差异。

SRCNN训练了超过1亿次迭代，这在我们的模型上是不可能实现的。考虑到二者的差异（SRCNN和我们的模型），在数据，训练，结构上的差异。我们训练图片转换网络`x4`,`x8`用了逐像素求差的损失函数，这些网络使用相同搞得数据，结构，训练网络去减少$l_{feat}$**评测：**我们评测了模型，在标准的集合5[60]，集合6[61]，BSD100[41]数据集，我们报告的PSNR和SSIM[54]，都只计算了在Y通道上的（当转换成YCbCr颜色空间后），跟[1,39]一样。

**结果：**我们展示了`x4`倍超分辨率重建的结果（图8），和其他的方法相比，我们的模型用特征重建训练的，得到了很好的结果，尤其是在锋锐的边缘和好的细节，比如图1的眼睫毛，图2帽子的细节元素。特征重建损失在放大下引起轻微的交叉阴影图案，和基础方法比起来更好。

<div align=center>
<img src="zh-cn/img/ncs2/p14.png" /> 
</div>

`x8`倍放大展示在图9中，我们又一次看到我们的模型在边缘和细节上的优秀。比如那个马的脚。$l_{feat}$模型不会无差别的锐化边缘；和$l_{pixel}$模型相比，$l_{feat}$模型锐化了马和骑士的边缘，但是北京的树并没被锐化。可能是因为$l_{feat}$模型更关注图像的语义信息。

<div align=center>
<img src="zh-cn/img/ncs2/p15.png" /> 
</div>

因为我们的$l_{pixel}$和$l_feat$模型有着相同的结构，数据，和训练过程，所以所有的差别都是因为$l_{pixel}和$l_{feat}$的不同导致的。$l_{pixel}$给出了更低的视觉效果，更高的PSNR值，而$l_{feat}$在重建细节上有着更好的表现，有着很好的视觉结果。

#### 5.结论

在这篇文章中，我们结合了前馈网络和基于优化的方法的好处，通过用感知损失函数来训练前馈网络。我们对风格转换应用了这个方法达到了很好的表现和速度。对超分辨率重建运用了这个方法，证明了用感知损失来训练，能带来更多好的细节和边缘。

未来的工作中，我们期望把感知损失函数用在更多其他的图像转换任务中，如上色或者语义检测。我们还打算研究不同损失网络用于不同的任务，或者更多种不同的语义信息的数据集



