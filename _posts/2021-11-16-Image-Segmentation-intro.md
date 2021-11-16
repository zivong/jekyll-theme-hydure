---
layout: post
title: Image Segmentation Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Image Segmentation includes Semantics Segmentation, Instance Segmentation, Video Object Segmentation, Panopitc Segmentation.


## Semantic Segmentation (意義分割）
以物件偵測的發展目標來看，我們可以知道大概有以下幾種目標:
![](https://miro.medium.com/max/700/1*OdEIh5K6qkHSzrPFk5Oa1g.jpeg)

### FCN - Fully Convolutional Networks

**Paper:** [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) <br />
**Code:** [https://github.com/hayoung-kim/tf-semantic-segmentation-FCN-VGG16](https://github.com/hayoung-kim/tf-semantic-segmentation-FCN-VGG16)<br />
**Ref.** [FCN for Semantic Segmentation簡介](https://ivan-eng-murmur.medium.com/%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-s7-fcn-for-semantic-segmentation%E7%B0%A1%E4%BB%8B-29814b07f96a)<br />

**FCN Architecture**
![](https://miro.medium.com/max/1288/1*W8LidqqqK7mClVx8BDjH1g.png)
**FCN-8 Architecture**
![](https://www.researchgate.net/publication/324435688/figure/fig3/AS:614087490809857@1523421275334/Model-architecture-We-adopt-the-full-convolutional-neural-network-FCN-model-for.png)
**Conv & DeConv**
![](https://miro.medium.com/max/700/1*I_01NsG2-wjq10xni91Zow.png)
![](https://miro.medium.com/max/700/1*-jIYmKVMMu_V-VY3A13gTQ.gif)
![](https://miro.medium.com/max/2000/1*PzvqB-3Q_11SVYbz_sJ-Mg.png)
<font size="3">
上圖為作者在論文中給出的融合組合。第一列的FCN-32是指將conv7層直接放大32倍的網路；而FCN-16則是將conv7層放大兩倍之後，和pool4做結合再放大16倍的網路，以此類推。<br />
</font>
![](https://ars.els-cdn.com/content/image/1-s2.0-S0950705120303464-gr4.jpg)<br />
<font size="3">
這些網路對應到的成果圖如下圖。可以發現，考慮越多不同尺度的feature map所得到的最終prediction map之精細度也越高，越接近ground-truth。<br/>
</font>
![](https://miro.medium.com/max/700/1*NpZlUwx4ogKf5B5Bj1XNwA.png)<br />
### Mask-RCNN
> **Paper:** [arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)<br />
> **Ref.** [理解Mask R-CNN的工作原理](https://www.jiqizhixin.com/articles/Mask_RCNN-tree-master-samples-balloon)
<font size="3">
Mask R-CNN 是個兩階段的架構，第一階段掃描圖像並生成proposals(即有可能包含一個目標的區域），第二階段分類提議並生成邊界框和Mask
</font>
![](https://image.jiqizhixin.com/uploads/editor/04c95a7a-8bc1-406c-8777-acb04578284c/1521687745369.jpg)<br />

### Image Segmentation Survey
> **Paper:** [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)<br />

![](https://miro.medium.com/max/1838/1*yqYWF5UcgImFGKtA7dCjMw.png)

> **Paper:** [Evolution of Image Segmentation using Deep Convolutional Neural Network: A Survey](https://arxiv.org/abs/2001.04074)<br />

![](https://ars.els-cdn.com/content/image/1-s2.0-S0950705120303464-gr2.jpg)

## Instance Segmentation (實例分割）
### YOLOACT
> **Paper:** [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)<br />
> &emsp;&emsp;&emsp;[YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)<br />
> **Code:** [https://github.com/dbolya/yolact](https://github.com/dbolya/yolact)<br />
> &emsp;&emsp;&emsp;[https://www.kaggle.com/rkuo2000/yolact](https://www.kaggle.com/rkuo2000/yolact)<br />

![](https://neurohive.io/wp-content/uploads/2019/12/Screenshot_7-scaled.png)
### U-Net
> **Paper:** [arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)<br />
> **Code:** [U-Net Keras](https://github.com/zhixuhao/unet)<br />

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-07_at_9.08.00_PM_rpNArED.png)
### 3D U-Net
> **Paper:** [arxiv.org/abs/1606.06650](https://arxiv.org/abs/1606.06650)<br />

![](https://miro.medium.com/max/2000/1*ovEGmOI3bcCeauu8jEBzsg.png)
### Brain Tumor Segmentation
> **Dataset:** Brain Tumor Segmentation(BraTS2020)<br />
> **Code:** [https://www.kaggle.com/polomarco/brats20-3dunet-3dautoencoder](https://www.kaggle.com/polomarco/brats20-3dunet-3dautoencoder)<br />

![](https://www.med.upenn.edu/cbica/assets/user-content/images/BraTS/brats-tumor-subregions.jpg)
### 3D MRI BraTS using AutoEncoder
> **Paper:** [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/abs/1810.11654)<br />

![](https://media.arxiv-vanity.com/render-output/5138047/x1.png)
### BraTS with 3D U-Net
> **Paper:**[Brain tumor segmentation with self-ensembled, deeply-supervised 3D U-net neural networks: a BraTS 2020 challenge solution](https://arxiv.org/abs/2011.01045)<br />

![](https://www.researchgate.net/publication/345261283/figure/fig1/AS:953986987356173@1604459630416/Neural-Network-Architecture-3D-Unet-with-minor-modifications.png)
### SegNet - A Deep Convolutional Encoder-Decoder Architecture
> **Paper:** [arxiv.org/abs/1511.00561](https://arxiv.org/abs/1511.00561)<br />
> **Code:** [github.com/yassouali/pytorch_segmentation](https://github.com/yassouali/pytorch_segmentation)<br />

![](https://production-media.paperswithcode.com/methods/segnet_Vorazx7.png)
### PsPNet - Pyramid Scene Parsing Network
> **Paper:** [arxiv.org/abs/1612.01105](https://arxiv.org/abs/1612.01105)<br />
> **Code:** [github.com/hszhao/semseg](https://github.com/hszhao/semseg) **(PSPNet, PSANet in PyTorch)**<br />

![](https://hszhao.github.io/projects/pspnet/figures/pspnet.png)
### DeepLab V3+
> **Paper:** [arxiv.org/abs/1802.02611](https://arxiv.org/abs/1802.02611)<br />
> **Code:** [github.com/bonlime/keras-deeplab-v3-plus](https://github.com/bonlime/keras-deeplab-v3-plus)<br />

![](https://discuss.pytorch.org/uploads/default/original/2X/5/56ab90914d256b1e3c8b1dd467f88357513fda1e.png)<br />
### Semantic Segmentation on MIT ADE20K
> **Code:** [github.com/CSAILVision/semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)<br />
> **Dataset:** [MIT ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/), Models: PSPNet, UPerNet, HRNet<br />

![](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/teaser/ADE_val_00000278.png?raw=true)
![](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/teaser/ADE_val_00001519.png?raw=true)
### Semantic Segmentation in PyTorch
> **Code:** [github.com/yassouali/pytorch_segmentation](https://github.com/yassouali/pytorch_segmentation)<br />
> **Datasets:** Pascal VOC, CityScapes, ADE20K, COCO Stuff<br />
> **Models:** [Deeplab V3+],[GCN],[UperNet],[DUC,HDC],[PSPNet],[ENet], [UNet],[SegNet],[FCN]<br />

## Video Object Datasets (影像物件資料集)
### DAVIS - Densely Annotated VIdeo Segmentation
**[DAVIS dataset](https://davischallenge.org/)**<br />
![](https://miro.medium.com/max/855/1*1QaTrWr5TjMHcp2y6gh92Q.jpeg)
### YTVOS - YouTube Video Object Segmentation
**[Video Object Segmentation](https://youtube-vos.org/)**<br />
![](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/161407653395985841614076533320.png)
<font size='3'>
2019 version <br />
&ensp;Training: 3471 videos, 65 categories and 6459 unique object instances. <br />
&ensp;Validation: 507 videos, 65 training categories, 26 unseen categories and 1063 unique object instances. <br />
&ensp;Test: 541 videos, 65 training categories, 29 unseen categories and 1092 unique object instances.
</font>

### YTVIS - YouTube Video Instance Segmentation
**[Video Instance Segmentation](https://youtube-vos.org/dataset/vis/)**<br />
![](https://production-media.paperswithcode.com/tasks/YouTube-VIS_wOOeQeN.png)

<font size='3'>
2021 version <br />
&ensp;3,859 high-resolution YouTube videos, 2,985 training videos, 421 validation videos and 453 test videos.<br />
&ensp;An improved 40-category label set <br />
&ensp;8,171 unique video instances <br />
&ensp;232k high-quality manual annotations
</font>
### UVO - Unidentified Video Objects
> **Paper:** [Unidentified Video Objects: A Benchmark for Dense, Open-World Segmentation](https://arxiv.org/abs/2104.04691)<br />
> **Website:** [Unidentified Video Objects](https://sites.google.com/view/unidentified-video-object/home)<br />

![](https://images.deepai.org/converted-papers/2104.04691/x2.png)

### Anomaly Video Datasets
> **Paper:** [A survey of video datasets for anomaly detection in automated surveillance](https://www.researchgate.net/publication/318412614_A_survey_of_video_datasets_for_anomaly_detection_in_automated_surveillance)<br />

![](https://d3i71xaburhd42.cloudfront.net/638d50100fe392ae705a0e5f7d9b47ca1dad3eea/5-TableI-1.png)<br />

## Video Object Segmentation (影像物件分割)

## Semantic Segmentation Datasets for Autonomous Driving (自動駕駛用意義分割資料集)

## Panoptic Segmentation (全景分割)


This demo site was last updated {{ site.time | date: "%B %d, %Y" }}.

