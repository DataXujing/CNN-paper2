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

!> 期待更新