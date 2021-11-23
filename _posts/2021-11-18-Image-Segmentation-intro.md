---
layout: post
title: Image Segmentation Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Image Segmentation includes Image Mitting, Semantics Segmentation, Human Part Segmentation, Instance Segmentation, Video Object Segmentation, Panopitc Segmentation.

![](https://github.com/mcahny/vps/blob/master/image/landscape.png?raw=true)

---
## Image Mitting
[Image Matting](https://paperswithcode.com/task/image-matting) is the process of accurately estimating the foreground object in images and videos.
<p align="center"><img src="https://production-media.paperswithcode.com/thumbnails/task/task-0000001397-92abcd60.jpg"></p>

### Deep Image Matting
**Paper:** [arxiv.org/abs/1703.03872](https://arxiv.org/abs/1703.03872)<br />

![](https://lh3.googleusercontent.com/GPJJYIN30gAuNbkFfqqNR2e1_ZBKFvVUUI5xL5e2FEhE_v8mVwAdTYU43-UHrKJHD0l5cdgeYJaoimkVIXwSCwNeWEh27X5s0JLVNpGy6jY=w1280)
![](https://lh4.googleusercontent.com/32qjRA2hs8DEtWc9fMCaQzHHbdY-mueYK0IdZ088g2NXXUSGLGUOr4bBnumDCBl6FmxfMOTWMpyZIRrm01jsnk9jxoM5EA0g4HR6xfucEcY=w1280)

### Semantic Image Matting
**Paper:** [arxiv.org/abs/2104.08201](https://arxiv.org/abs/2104.08201)<br />
**Code:** [nowsyn/SIM](https://github.com/nowsyn/SIM)<br />

![](https://github.com/nowsyn/SIM/blob/master/figures/framework.jpg?raw=True)
<table>
  <tr>
  <td><img src="https://github.com/nowsyn/SIM/raw/master/figures/example1.png"></td>
  <td><img src="https://github.com/nowsyn/SIM/raw/master/figures/example2.png"></td>
  </tr>
</table>

### MODNet: Trimap-Free Portrait Matting in Real Time
**Paper:** [arxiv.org/abs/2011.11961](https://arxiv.org/abs/2011.11961)<br />
**Code:** [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)<br />

![](https://github.com/ZHKKKe/MODNet/raw/master/doc/gif/homepage_demo.gif)

---
## Semantic Segmentation (意義分割）

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

### Image Segmentation Survey
> **Paper:** [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)<br />

![](https://miro.medium.com/max/1838/1*yqYWF5UcgImFGKtA7dCjMw.png)

> **Paper:** [Evolution of Image Segmentation using Deep Convolutional Neural Network: A Survey](https://arxiv.org/abs/2001.04074)<br />
<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0950705120303464-gr2.jpg" />
</p>

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
### PSPNet - Pyramid Scene Parsing Network
> **Paper:** [arxiv.org/abs/1612.01105](https://arxiv.org/abs/1612.01105)<br />
> **Code:** [github.com/hszhao/semseg](https://github.com/hszhao/semseg) **(PSPNet, PSANet in PyTorch)**<br />

![](https://hszhao.github.io/projects/pspnet/figures/pspnet.png)
### DeepLab V3+
> **Paper:** [arxiv.org/abs/1802.02611](https://arxiv.org/abs/1802.02611)<br />
> **Code:** [github.com/bonlime/keras-deeplab-v3-plus](https://github.com/bonlime/keras-deeplab-v3-plus)<br />

![](https://discuss.pytorch.org/uploads/default/original/2X/5/56ab90914d256b1e3c8b1dd467f88357513fda1e.png)<br />

---
### Semantic Segmentation on MIT ADE20K
> **Code:** [github.com/CSAILVision/semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)<br />
> **Dataset:** [MIT ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/), Models: PSPNet, UPerNet, HRNet<br />

![](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/teaser/ADE_val_00000278.png?raw=true)
![](https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/teaser/ADE_val_00001519.png?raw=true)

### Semantic Segmentation in PyTorch
> **Code:** [github.com/yassouali/pytorch_segmentation](https://github.com/yassouali/pytorch_segmentation)<br />
> **Datasets:** Pascal VOC, CityScapes, ADE20K, COCO Stuff<br />
> **Models:** [Deeplab V3+],[GCN],[UperNet],[DUC,HDC],[PSPNet],[ENet], [UNet],[SegNet],[FCN]<br />

---
## Human Part Segmentation
[https://paperswithcode.com/task/human-part-segmentation](https://paperswithcode.com/task/human-part-segmentation)

### [Look Into Person Challenge 2020](https://vuhcs.github.io) [[LIP](http://sysu-hcp.net/lip/index.php)]
* LIP is the largest single person human parsing dataset with 50000+ images. This dataset focus more on the complicated real scenarios. LIP has 20 labels, including 'Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'.

### [HumanParsing-Dataset](https://github.com/lemondan/HumanParsing-Dataset) [[ATR](http://pan.baidu.com/s/1qY8bToS)] (passwd：kjgk)
* **Paper: [Human Parsing with Contextualized Convolutional Neural Network](https://openaccess.thecvf.com/content_iccv_2015/papers/Liang_Human_Parsing_With_ICCV_2015_paper.pdf)**<br />
* ATR is a large single person human parsing dataset with 17000+ images. This dataset focus more on fashion AI. ATR has 18 labels, including 'Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'.

### [PASCAL-Part Dataset](http://roozbehm.info/pascal-parts/pascal-parts.html) [[PASCAL](http://roozbehm.info/pascal-parts/trainval.tar.gz)]
![](http://roozbehm.info/pascal-parts/pascal_part_annotation.png)
* Pascal Person Part is a tiny single person human parsing dataset with 3000+ images. This dataset focus more on body parts segmentation. Pascal Person Part has 7 labels, including 'Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'.

---
### Self Correction Human Parsing
> **Blog:** [HumanPartSegmentation : A Machine Learning Model for Segmenting Human Parts](https://medium.com/axinc-ai/humanpartsegmentation-a-machine-learning-model-for-segmenting-human-parts-cd7e39480714)<br />
> **Paper:** [arxiv.org/abs/1910.09777](https://arxiv.org/abs/1910.09777)<br />
> **Code:** [PeikeLi/Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing)<br />

![](https://miro.medium.com/max/700/1*iLCXQ9nZTecAClSStzdlxw.png)
![](https://github.com/PeikeLi/Self-Correction-Human-Parsing/raw/master/demo/lip-visualization.jpg)

<iframe width="665" height="382" src="https://www.youtube.com/embed/-bYS9TJYmzI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br />

### Cross-Domain Complementary Learning Using Pose for Multi-Person Part Segmentation
> **Paper:** [arxiv.org/abs/1907.05193](https://arxiv.org/abs/1907.05193)<br />
> **Code:** [kevinlin311tw/CDCL-human-part-segmentation](https://github.com/kevinlin311tw/CDCL-human-part-segmentation)<br />

![](https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/cdcl_teaser.jpg)
<iframe width="685" height="514" src="https://www.youtube.com/embed/8QaGfdHwH48" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Instance Segmentation (實例分割）
### A Survey on Instance Segmentation
> **Paper:** [arxiv.org/abs/2007.00047](https://arxiv.org/abs/2007.00047)<br />

![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs13735-020-00195-x/MediaObjects/13735_2020_195_Fig1_HTML.png)
![]()

### Mask-RCNN
> **Paper:** [arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)<br />
> **Ref.** [理解Mask R-CNN的工作原理](https://www.jiqizhixin.com/articles/Mask_RCNN-tree-master-samples-balloon)<br />
<font size="3">
Mask R-CNN 是個兩階段的架構，第一階段掃描圖像並生成proposals(即有可能包含一個目標的區域），第二階段分類提議並生成邊界框和Mask
</font>
![](https://image.jiqizhixin.com/uploads/editor/04c95a7a-8bc1-406c-8777-acb04578284c/1521687745369.jpg)

### TensorMask - A Foundation for Dense Object Segmentation
> **Paper:** [arxiv.org/abs/1903.12174](https://arxiv.org/abs/1903.12174)<br />
> **Code:** [TensorMask in Detectron2](https://github.com/facebookresearch/detectron2/blob/main/projects/TensorMask/README.md)<br />

![](https://camo.githubusercontent.com/45c54f74b67b2912f4bd768000f64be02b8841af6f9a9b9afc44b8d9a0852a25/687474703a2f2f78696e6c6569632e78797a2f696d616765732f746d61736b2e706e67)

### YOLACT - Real-Time Instance Segmentation
> **Paper:** [arxiv.org/abs/1904.02689](https://arxiv.org/abs/1904.02689)<br />
> &emsp;&emsp;&emsp;[YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)<br />
> **Code:** [https://github.com/dbolya/yolact](https://github.com/dbolya/yolact)<br />
> &emsp;&emsp;&emsp;[https://www.kaggle.com/rkuo2000/yolact](https://www.kaggle.com/rkuo2000/yolact)<br />

![](https://neurohive.io/wp-content/uploads/2019/12/Screenshot_7-scaled.png)

### INSTA YOLO
> **Paper:** [arxiv.org/abs/2102.06777](https://arxiv.org/abs/2102.06777)<br />

![](https://www.researchgate.net/publication/349335219/figure/fig2/AS:991674289377283@1613444983309/Insta-YOLO-architecture-which-is-inspired-by-YOLO-the-right-part-illustrate-our.ppm)

---
## Video Object Datasets (影像物件資料集)
### DAVIS - Densely Annotated VIdeo Segmentation
**[DAVIS dataset](https://davischallenge.org/)**<br />
![](https://miro.medium.com/max/855/1*1QaTrWr5TjMHcp2y6gh92Q.jpeg)

**[DAVIS 2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)**<br />
`!wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip`<br />
`!unzip -q DAVIS-2017-trainval-480p.zip`<br />

### YTVOS - YouTube Video Object Segmentation
**[Video Object Segmentation](https://youtube-vos.org/)**<br />
![](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/161407653395985841614076533320.png)

* 4000+ high-resolution YouTube videos
* 90+ semantic categories
* 7800+ unique objects
* 190k+ high-quality manual annotations
* 340+ minutes duration

**[YTVOS 2019](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz)**<br />
**[YTVOS 2018](https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f)**
* train.zip
* train_all_frames.zip
* valid.zip
* valid_all_frames.zip
* test.zip
* test_all_frames.zip

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

---
## Video Object Segmentation (影像物件分割)

![](https://pic1.zhimg.com/v2-59e3503956443609125aef18f8fbc7ed_1440w.jpg?source=172ae18b)

### FlowNet 2.0
> **Paper:** [arxiv.org/abs/1612.01925](https://arxiv.org/abs/1612.01925)<br />
> **Code:** [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch)<br />

![](https://www.researchgate.net/profile/Mehdi-Elahi-2/publication/338873930/figure/fig2/AS:857134044565505@1581368089880/FlowNet-20-architecture-37-which-consists-of-FlowNetS-and-FlowNetC-stacked.ppm)

### Learning What to Learn for VOS
> **Paper:** [arxiv.org/abs/2003.11540](https://arxiv.org/abs/2003.11540)<br />
> **Blog:** [Learning What to Learn for Video Object Seg](https://zhuanlan.zhihu.com/p/197559268)<br />

![](https://pic2.zhimg.com/v2-59e3503956443609125aef18f8fbc7ed_1440w.jpg)

### FRTM-VOS
> **Paper:** [arxiv.org/abs/2003.00908](https://arxiv.org/abs/2003.00908)<br />
> **Code:** [andr345/frtm-vos](https://github.com/andr345/frtm-vos)<br />

![](https://img-blog.csdnimg.cn/2020071300041668.png)

### State-Aware Tracker for Real-Time VOS
> **Paper:** [arxiv.org/abs/2003.00482](https://arxiv.org/abs/2003.00482)<br />
> **Code:** [MegviiDetection/video_analyst](https://github.com/MegviiDetection/video_analyst)<br />
<table>
  <tr>
  <td><img src="https://github.com/MegviiDetection/video_analyst/blob/master/docs/resources/siamfcpp_ice2.gif?raw=true" /></td>
  <td><img src="https://github.com/MegviiDetection/video_analyst/blob/master/docs/resources/sat_runman.gif?raw=true" /></td>
  </tr>
</table>
### Motion Estimation with Optical Flow
> **Blog:** [Introduction to Motion Estimation with Optical Flow](https://nanonets.com/blog/optical-flow/)<br />

![](https://nanonets.com/blog/content/images/2019/04/sparse-vs-dense.gif)

### LiteFlowNet3
> **Paper:** [arxiv.org/abs/2007.09319](https://arxiv.org/abs/2007.09319)<br />
> **Code:** [twhui/LiteFlowNet3](https://github.com/twhui/LiteFlowNet3)<br />

![](https://github.com/twhui/LiteFlowNet3/raw/master/figures/LiteFlowNet3.png?raw=True)
#### Cost Volume Modulation (CM)
![](https://github.com/twhui/LiteFlowNet3/raw/master/figures/cost_volume_modulation.png?raw=True)
#### Flow Field Deformation (FD)
![](https://github.com/twhui/LiteFlowNet3/raw/master/figures/flow_field_deformation.png?raw=True)

---
## Semantic Segmentation Datasets for Autonomous Driving 
<h3>自動駕駛用意義分割資料集</h3>
### [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
![](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/pr/DBOverview1_1_huff_0000964.jpg)

### [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
<iframe width="920" height="520" src="https://www.youtube.com/embed/KXpZ6B1YB_k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### [CityScapes Dataset](https://www.cityscapes-dataset.com/)
[Cityscapes Examples](https://www.cityscapes-dataset.com/examples/)<br />
[Cityscapes 3D Benchmark Online](https://www.cityscapes-dataset.com/cityscapes-3d-benchmark-online/)<br />
<table>
  <tr>
  <td><img src="https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/zuerich00-300x149.png" /></td>
  <td><img src="https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/jena00-300x150.png" /></td>
  </tr>
</table>
<table>
  <tr>
  <td><iframe src="https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/videos/labelExamples.mp4?id=1" frameborder="0" allowfullscreen></iframe></td>
  <td><iframe src="https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/videos/gpsMotionMeta.mp4?id=3" frameborder="0" allowfullscreen></iframe></td>
  </tr>
</table>

### [Mapillary Vitas Dataset](https://www.mapillary.com/dataset/vistas)
* 25,000 high-resolution images
* 124 semantic object categories
* 100 instance-specifically annotated categories
* Global reach, covering 6 continents
* Variety of weather, season, time of day, camera, and viewpoint

![](https://pbs.twimg.com/media/C-_OXVoU0AAPc6h?format=jpg&name=small)

---
## Panoptic Segmentation (全景分割)
### YOLOP
> **Paper:** [arxiv.org/abs/2108.11250](https://arxiv.org/abs/2108.11250)<br />
> **Code:** [hustvl/YOLOP](https://github.com/hustvl/YOLOP)<br />

![](https://github.com/hustvl/YOLOP/blob/main/pictures/yolop.png?raw=true)

### VPSNet for  Video Panoptic Segmentation
> **Paper:** [arxiv.org/abs/2006.11339](https://arxiv.org/abs/2006.11339)<br />
> **Code:** [mcahny/vps](https://github.com/mcahny/vps)<br />

![](https://github.com/mcahny/vps/blob/master/image/panoptic_pair_240.gif?raw=true)

---
### PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation
> **Paper:** [arxiv.org/abs/2101.06175](https://arxiv.org/abs/2101.06175)<br />
> **Code:** [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)<br />

PaddleSeg提供了语义分割、交互式分割、全景分割、Matting四大图像分割能力
![](https://user-images.githubusercontent.com/53808988/130562440-1ea5cbf5-4caf-424c-a9a7-55d56b7d7776.gif?raw=True)
PaddleSeg被广泛地应用在自动驾驶、医疗、质检、巡检、娱乐等场景。
![](https://user-images.githubusercontent.com/53808988/130562530-ae45c2cd-5dd7-48f0-a080-c0e843eea49d.gif?raw=True) 
<br />

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

