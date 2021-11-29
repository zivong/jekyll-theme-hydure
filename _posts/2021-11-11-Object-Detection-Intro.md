---
layout: post
title: Object Detection Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Object Detection includes Image Datasets, Object Detection and Object Tracking.

---
## Image Datasets
### [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
**VOC2007:** 20 classes. The train/val/test 9,963 images containiing 24,640 annotated objects<br/>
* [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* [VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

**VOC2012:** 20 classes. The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations <br/>
* [VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
* [benchmark.tgz](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

![](https://paperswithcode.github.io/torchbench/img/pascalvoc2012.png)

* [TorchBench PASCAL VOC2012](https://paperswithcode.github.io/torchbench/pascalvoc/)

```
from torchbench.semantic_segmentation import PASCALVOC
from torchvision.models.segmentation import fcn_resnet101
model = fcn_resnet101(num_classes=21, pretrained=True)

PASCALVOC.benchmark(model=model,
    paper_model_name='FCN ResNet-101',
    paper_arxiv_id='1605.06211')
```
    
### [COCO Dataset](https://cocodataset.org)
330K images (>200K labels), 1.5million object instances,80 object categories

![](https://cocodataset.org/images/coco-examples.jpg)

### [ImageNet](https://image-net.org/)
1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images.
![](https://miro.medium.com/max/1400/1*IlzW43-NtJrwqtt5Xy3ISA.jpeg)

### [Open Images Dateset](https://storage.googleapis.com/openimages/web/index.html)
Open Images Dataset V6+: 
These annotation files cover the 600 boxable object classes, and span the 1,743,042 training images where we annotated bounding boxes, object segmentations, visual relationships, and localized narratives; as well as the full validation (41,620 images) and test (125,436 images) sets.
![](https://1.bp.blogspot.com/-yuodfZa6gyM/XlbQfiAzbzI/AAAAAAAAFYA/QSTnuZksQII2PaRON2mqHntZBHL-saniACLcBGAsYHQ/s640/Figure1.png)

---
### Object Detection Milestones
![](https://www.researchgate.net/profile/Zhengxia-Zou-2/publication/333077580/figure/fig2/AS:758306230501380@1557805702766/A-road-map-of-object-detection-Milestone-detectors-in-this-figure-VJ-Det-10-11-HOG.ppm)
![](https://www.researchgate.net/profile/Zhengxia-Zou-2/publication/333077580/figure/fig3/AS:758306234724352@1557805703089/The-accuracy-improvements-of-object-detection-on-VOC07-VOC12-and-MS-COCO-datasets.ppm)

---
### R-CNN, Fast R-CNN, Faster R-CNN
**Blog:** [目標檢測](https://www.twblogs.net/a/5cb52483bd9eee0f00a1ad24)

![](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdnimg.cn/20190415130546284.png)

* **R-CNN**首先使用Selective search提取region proposals（候選框）；然後用Deep Net（Conv layers）進行特徵提取；最後對候選框類別分別採用SVM進行類別分類，採用迴歸對bounding box進行調整。其中每一步都是獨立的。
* **Fast R-CNN**在R-CNN的基礎上，提出了多任務損失(Multi-task Loss), 將分類和bounding box迴歸作爲一個整體任務進行學習；另外，通過ROI Projection可以將Selective Search提取出的ROI區域（即：候選框Region Proposals）映射到原始圖像對應的Feature Map上，減少了計算量和存儲量，極大的提高了訓練速度和測試速度。
* **Faster R-CNN**則是在Fast R-CNN的基礎上，提出了RPN網絡用來生成Region Proposals。通過網絡共享將提取候選框與目標檢測結合成一個整體進行訓練，替換了Fast R-CNN中使用Selective Search進行提取候選框的方法，提高了測試過程的速度。

---
### R-CNN
**Paper:** [arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)<br/>
![](https://miro.medium.com/max/700/1*REPHY47zAyzgbNKC6zlvBQ.png)
![](https://miro.medium.com/max/500/1*E-8oQW8ZO-hHgTf6laWhhQ.png)

---
### Fast R-CNN
**Paper:** [arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083)<br/>
**Github:** [faster-rcnn](https://github.com/rbgirshick/fast-rcnn)<br/>
![](https://miro.medium.com/max/700/1*0pMP3aY8blSpva5tvWbnKA.png)

---
### Faster R-CNN
**Paper:** [arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)<br/>
**Github:** [faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn), [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
![](https://miro.medium.com/max/842/1*ndYVI-YCEGCoyRst1ytHjA.png)

---
**Blog:** [[物件偵測] S3: Faster R-CNN 簡介](https://ivan-eng-murmur.medium.com/object-detection-s3-faster-rcnn-%E7%B0%A1%E4%BB%8B-5f37b13ccdd2)<br />
* RPN是一個要提出proposals的小model，而這個小model需要我們先訂出不同尺度、比例的proposal的邊界匡的雛形。而這些雛形就叫做anchor。

<p align="center"><img width="50%" height="50%" src="https://miro.medium.com/max/700/1*X36ZRFab42L4Rwn22j8d6Q.png"></p>
![](https://miro.medium.com/max/2000/1*ddngAD0M9ovnPcg9YaZu9g.png)

* RPN的上路是負責判斷anchor之中有無包含物體的機率，因此，1×1的卷積深度就是9種anchor，乘上有無2種情況，得18。而下路則是負責判斷anchor的x, y, w, h與ground truth的偏差量(offsets)，因此9種anchor，乘上4個偏差量(dx, dy, dw, dh)，得卷積深度為36。

![](https://miro.medium.com/max/1400/1*Fg7DVdvF449PfX5Fd6oOYA.png)

---
### Mask R-CNN
**Paper:** [arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)<br/>
**Code:** [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)<br/>
![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-23_at_7.44.34_PM.png)

**Blog:** [[物件偵測] S9: Mask R-CNN 簡介](https://ivan-eng-murmur.medium.com/%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-s9-mask-r-cnn-%E7%B0%A1%E4%BB%8B-99370c98de28)<br/>
![](https://miro.medium.com/max/1400/0*IDBowO6956w5RGVw)
![](https://miro.medium.com/max/2000/0*RTcInnhfoh0m9ItI)
![](https://miro.medium.com/max/2000/0*-tQsWmjcPhVfwRZ4)
![](https://github.com/matterport/Mask_RCNN/blob/master/assets/street.png?raw=true)
![](https://github.com/matterport/Mask_RCNN/blob/master/assets/images_to_osm.png?raw=true)
![](https://github.com/matterport/Mask_RCNN/blob/master/assets/nucleus_segmentation.png?raw=true)

---
### SSD: Single Shot MultiBox Detector
**Paper:** [arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)<br/>
**Code:** [pierluigiferrari/ssd_keras](https://github.com/pierluigiferrari/ssd_keras)<br/>
**Blog:** [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)<br/>
![](https://miro.medium.com/max/700/1*JuhjYUWXgfxMMoa4SIKLkA.png)
使用神經網絡（VGG-16）提取feature map後進行分類和回歸來檢測目標物體。
![](https://miro.medium.com/max/700/1*51joMGlhxvftTxGtA4lA7Q.png)
![](https://miro.medium.com/max/480/1*IZf0wajQ75DPsoBkWjwlsA.gif)

---
### RetinaNet
**Paper:** [arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)<br/>
**Code:** [keras-retinanet](https://github.com/fizyr/keras-retinanet)<br/>
**Blog:** [RetinaNet 介紹](https://gino6178.medium.com/%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-retinanet-%E4%BB%8B%E7%B4%B9-dda4100673bb)
![](https://miro.medium.com/max/622/0*ksQqcCYF0iQN_oX2.png)
從左到右分別用上了<br/>
* 殘差網路(Residual Network ResNet)
* 特徵金字塔(Feature Pyramid Network FPN)
* 類別子網路(Class Subnet)
* 框子網路(Box Subnet)
* 以及Anchors

**Blog:** [Review: RetinaNet — Focal Loss](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)
![](https://miro.medium.com/max/600/0*E30eIZ5aCGjhCz9E.gif)
<iframe width="665" height="382" src="https://www.youtube.com/embed/lZxMklxzm2Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### PointRend
**Paper:** [arxiv.org/abs/1912.08193](https://arxiv.org/abs/1912.08193)<br/>
**Code:** [keras-retinanet](https://github.com/fizyr/keras-retinanet)<br/>
**Blog:** [Facebook PointRend: Rendering Image Segmentation](https://medium.com/syncedreview/facebook-pointrend-rendering-image-segmentation-f3936d50e7f1)
![](https://miro.medium.com/max/606/0*XPUeNroEr8WATWGG.png)
![](https://miro.medium.com/max/674/0*7ed5Nc7fVtSEByt3.png)
![](https://miro.medium.com/max/700/0*P0Mp6DeRC-fHYd8g.png)

---
### EfficientDet
**Paper:** [arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)<br/>
**Code:** [google efficientdet](https://github.com/google/automl/tree/master/efficientdet)<br/>
**Kaggle:** [rkuo2000/efficientdet-gwd](https://www.kaggle.com/rkuo2000/efficientdet-gwd)<br/>
![](https://github.com/google/automl/raw/master/efficientdet/g3doc/network.png)
![](https://camo.githubusercontent.com/f0c80711512aacc0c1423a64e1036608a773f61c0bb6480ec0d57614ce3e7ccc/68747470733a2f2f696d6775722e636f6d2f3579554a4350562e6a7067)
<table>
<tr>
<td><img src="https://github.com/google/automl/blob/master/efficientdet/g3doc/flops.png?raw=true"></td>
<td><img src="https://github.com/google/automl/blob/master/efficientdet/g3doc/params.png?raw=true"></td>
</tr>
</table>

### [YOLO- You Only Look Once](https://pjreddie.com/darknet/yolo/)
**Code:** [pjreddie/darknet](https://github.com/pjreddie/darknet)<br/>
![](https://pyimagesearch.com/wp-content/uploads/2018/11/yolo_design.jpg)

**[YOLOv1](https://arxiv.org/abs/1506.02640)** : mapping bounding box<br/>
![](https://manalelaidouni.github.io/assets/img/pexels/YOLO_arch.png)

**[YOLOv2](https://arxiv.org/abs/1612.08242)** : anchor box proportional to K-means<br/>
![](https://2.bp.blogspot.com/-_R-w_tWHdzc/WzJPsol7qFI/AAAAAAABbgg/Jsf-AO3qH0A9oiCeU0LQxN-wdirlOz4WgCLcBGAs/s400/%25E8%259E%25A2%25E5%25B9%2595%25E5%25BF%25AB%25E7%2585%25A7%2B2018-06-26%2B%25E4%25B8%258B%25E5%258D%258810.36.51.png)

**[YOLOv3](https://arxiv.org/abs/1804.02767)** : Darknet-53 + FPN<br/>
![](https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41598-021-81216-5/MediaObjects/41598_2021_81216_Fig1_HTML.png)
![](https://miro.medium.com/max/2000/1*d4Eg17IVJ0L41e7CTWLLSg.png)

---
### YOLObile
**Paper:** [arxiv.org/abs/2009.05697](https://arxiv.org/abs/2009.05697)<br/>
**Code:** [nightsnack/YOLObile](https://github.com/nightsnack/YOLObile)<br/>
**Blog:** [YOLObile：移動設備上的實時目標檢測](https://twgreatdaily.com/zh-hk/jRnp3HQBd8y1i3sJ_wK9.html)<br/>
![](https://images.twgreatdaily.com/images/elastic/vhk/vhk33XQBd8y1i3sJeRJg.jpg)
![](https://github.com/nightsnack/YOLObile/raw/master/figure/yolo_demo.jpg)

---
### YOLOv4 : YOLOv3 + CSPDarknet53 + SPP + PAN + BoF + BoS
**Paper:** [arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)<br/>
**Code:** [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)<br/>
**Code:** [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)<br/>
![](https://blog.roboflow.com/content/images/2020/06/image-10.png?raw=true)

![](https://www.researchgate.net/profile/Jishu-Miao/publication/349381918/figure/fig4/AS:994706955722753@1614168027527/Normal-YOLOv4-network-architecture.ppm)

**CSP**<br/>
![](https://blog.roboflow.com/content/images/2020/06/image-15.png)
**PANet**<br/>
![](https://blog.roboflow.com/content/images/2020/06/image-17.png)

---
### YOLOv5
**Code:** [ultralytics/yolov5/](https://github.com/ultralytics/yolov5/)<br/>
![](https://user-images.githubusercontent.com/4210061/107134685-4b249480-692f-11eb-93b1-619708d95441.png)
![](https://user-images.githubusercontent.com/26833433/127574988-6a558aa1-d268-44b9-bf6b-62d4c605cc72.jpg)
![](https://user-images.githubusercontent.com/26833433/136901921-abcfcd9d-f978-4942-9b97-0e3f202907df.png)

---
### Scaled-YOLOv4
**Paper:** [arxiv.org/abs/2011.08036](https://arxiv.org/abs/2011.08036)<br/>
**Code:** [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)<br/>

![](https://miro.medium.com/max/1838/1*OE4SO1U87DHcAClSZFGlMg.png)

---
### YOLOR : You Only Learn One Representation
**Paper:** [arxiv.org/abs/2105.04206](https://arxiv.org/abs/2105.04206)<br/>
**Code:** [WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)<br/>
![](https://github.com/WongKinYiu/yolor/raw/main/figure/unifued_network.png?raw=true)
![](https://github.com/WongKinYiu/yolor/blob/main/inference/output/horses.jpg?raw=true)
![](https://github.com/WongKinYiu/yolor/raw/main/figure/performance.png?raw=true)

---
### YOLOX
**Paper:** [arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)<br/>
**Code:** [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)<br/>
![](https://miro.medium.com/max/915/1*ihnRFgPMgatEtrlTtOM2Bg.png)
![](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/assets/demo.png?raw=true)
![](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/assets/git_fig.png?raw=true)

<br/>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

