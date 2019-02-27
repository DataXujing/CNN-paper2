<img src="docs/_media/icon.svg" align="right" alt="logo" height="180" width="180" />

# 卷积神经网络教程

**徐静**

卷积神经网络基础学习: XuJing'Home [https://dataxujing.github.io/](https://dataxujing.github.io/)

关于CNN的基础知识及相关理论推导可以参考：<https://dataxujing.github.io/深度学习之CNN/>

----

<div align=center>
<img src="docs/zh-cn/img/index/cnn_hist.png" />
</div>

常用图像分类CNN结构：

+ ConvNet：卷积神经网络名称

+ ImageNet top1 acc：该网络在ImageNet上Top1 最佳准确率

+ ImageNet top5 acc：该网络在ImageNet上Top5 最佳准确率

+ Published In：发表源（期刊/会议/arXiv）


|         ConvNet            | ImageNet top1 acc | ImageNet top5 acc |   Published In     |
|:--------------------------:|:-----------------:|:-----------------:|:------------------:|
|           Vgg              |      76.3         |       93.2        |      ICLR2015      |   
|        GoogleNet           |       -           |       93.33       |      CVPR2015      |   
|        PReLU-nets          |       -           |       95.06       |      ICCV2015      |   
|          ResNet            |       -           |       96.43       |      CVPR2015      |   
|       PreActResNet         |      79.9         |       95.2        |      CVPR2016      |   
|       Inceptionv3          |      82.8         |       96.42       |      CVPR2016      |   
|       Inceptionv4          |      82.3         |       96.2        |      AAAI2016      |   
|    Inception-ResNet-v2     |      82.4         |       96.3        |      AAAI2016      |   
|Inceptionv4 + Inception-ResNet-v2|      83.5         |       96.92       |      AAAI2016      |   
|           RiR              |       -           |         -         |  ICLR Workshop2016 |   
|  Stochastic Depth ResNet   |      78.02        |         -         |      ECCV2016      |   
|           WRN              |      78.1         |       94.21       |      BMVC2016      |   
|       SqueezeNet           |      60.4         |       82.5        |      arXiv2017([rejected by ICLR2017](https://openreview.net/forum?id=S1xh5sYgx))     |   
|          GeNet             |      72.13        |       90.26       |      ICCV2017      |   
|         MetaQNN            |       -           |         -         |      ICLR2017      |   
|        PyramidNet          |      80.8         |       95.3        |      CVPR2017      |   
|         DenseNet           |      79.2         |       94.71       |      ECCV2017      |   
|        FractalNet          |      75.8         |       92.61       |      ICLR2017      |   
|         ResNext            |       -           |       96.97       |      CVPR2017      |   
|         IGCV1              |      73.05        |       91.08       |      ICCV2017      |   
| Residual Attention Network |      80.5         |       95.2        |      CVPR2017      |   
|        Xception            |       79          |       94.5        |      CVPR2017      |   
|        MobileNet           |      70.6         |         -         |      arXiv2017     |   
|         PolyNet            |      82.64        |       96.55       |      CVPR2017      |   
|           DPN              |       79          |       94.5        |      NIPS2017      |   
|        Block-QNN           |      77.4         |       93.54       |      CVPR2018      |   
|         CRU-Net            |      79.7         |       94.7        |      IJCAI2018     |   
|       ShuffleNet           |      75.3         |         -         |      CVPR2018      |   
|       CondenseNet          |      73.8         |       91.7        |      CVPR2018      |   
|          NasNet            |      82.7         |       96.2        |      CVPR2018      |   
|       MobileNetV2          |      74.7         |         -         |      CVPR2018      |   
|         IGCV2              |      70.07        |         -         |      CVPR2018      |   
|          hier              |      79.7         |       94.8        |      ICLR2018      |   
|         PNasNet            |      82.9         |       96.2        |      ECCV2018      |   
|        AmoebaNet           |      83.9         |       96.6        |      arXiv2018     |   
|          SENet             |       -           |       97.749      |      CVPR2018      |   
|       ShuffleNetV2         |      81.44        |         -         |      ECCV2018      |   
|          IGCV3             |      72.2         |         -         |      BMVC2018      |   
|         MnasNet            |      76.13        |       92.85       |      arXiv2018     |   

from: <https://github.com/weiaicunzai/awesome-image-classification>

关于LeNet-5,AlexNet,VGG16,VGG19这些网络结构我们在<https://dataxujing.github.io/深度学习之CNN/>中已经详细的解释，并且本教程中涉及的网路结构像ResNet,NIN,Inception,YOLO等也做了详细解释。本教程是对这些网络结构更详细的讨论。

----

+ ResNet
+ Google Inception
+ DensenNet
+ R-CNN, Selective Search, SPP-net
+ Fast R-CNN
+ Faster R-CNN
+ SSD
+ Mask R-CNN
+ YOLO
+ 从MobileNet到ShuffleNet
+ 神经风格转换
+ 人脸识别
+ 图像分割
+ N种卷积
