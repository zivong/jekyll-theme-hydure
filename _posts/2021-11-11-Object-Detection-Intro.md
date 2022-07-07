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
**VOC2007:** 20 classes. The train/val/test 9,963 images containiing 24,640 annotated objects<br>
* [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* [VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

**VOC2012:** 20 classes. The train/val data has 11,530 images containing 27,450 ROI annotated objects and 6,929 segmentations <br>
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
## Object Detection
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
**Paper:** [arxiv.org/abs/1311.2524](https://arxiv.org/abs/1311.2524)<br>
![](https://miro.medium.com/max/700/1*REPHY47zAyzgbNKC6zlvBQ.png)
![](https://miro.medium.com/max/500/1*E-8oQW8ZO-hHgTf6laWhhQ.png)

---
### Fast R-CNN
**Paper:** [arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083)<br>
**Github:** [faster-rcnn](https://github.com/rbgirshick/fast-rcnn)<br>
![](https://miro.medium.com/max/700/1*0pMP3aY8blSpva5tvWbnKA.png)

---
### Faster R-CNN
**Paper:** [arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497)<br>
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
**Paper:** [arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)<br>
![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-23_at_7.44.34_PM.png)
![](https://miro.medium.com/max/2000/0*-tQsWmjcPhVfwRZ4)
**Blog:** [[物件偵測] S9: Mask R-CNN 簡介](https://ivan-eng-murmur.medium.com/%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-s9-mask-r-cnn-%E7%B0%A1%E4%BB%8B-99370c98de28)<br>
![](https://miro.medium.com/max/1400/0*IDBowO6956w5RGVw)
![](https://miro.medium.com/max/2000/0*RTcInnhfoh0m9ItI)
**Code:** [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)<br>
![](https://github.com/matterport/Mask_RCNN/blob/master/assets/street.png?raw=true)
![](https://github.com/matterport/Mask_RCNN/blob/master/assets/images_to_osm.png?raw=true)
![](https://github.com/matterport/Mask_RCNN/blob/master/assets/nucleus_segmentation.png?raw=true)

---
### SSD: Single Shot MultiBox Detector
**Paper:** [arxiv.org/abs/1512.02325](https://arxiv.org/abs/1512.02325)<br>
**Blog:** [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)<br>
![](https://miro.medium.com/max/700/1*JuhjYUWXgfxMMoa4SIKLkA.png)
使用神經網絡（VGG-16）提取feature map後進行分類和回歸來檢測目標物體。
![](https://miro.medium.com/max/700/1*51joMGlhxvftTxGtA4lA7Q.png)
![](https://miro.medium.com/max/480/1*IZf0wajQ75DPsoBkWjwlsA.gif)
**Code:** [pierluigiferrari/ssd_keras](https://github.com/pierluigiferrari/ssd_keras)<br>
<table>
<tr>
<td><img src="https://github.com/pierluigiferrari/ssd_keras/blob/master/examples/trained_ssd300_pascalVOC2007_test_pred_05_no_gt.png?raw=true"></td>
<td><img src="https://github.com/pierluigiferrari/ssd_keras/blob/master/examples/trained_ssd300_pascalVOC2007_test_pred_04_no_gt.png?raw=true"></td>
</tr>
<tr>
<td><img src="https://github.com/pierluigiferrari/ssd_keras/blob/master/examples/trained_ssd300_pascalVOC2007_test_pred_01_no_gt.png?raw=true"></td>
<td><img src="https://github.com/pierluigiferrari/ssd_keras/blob/master/examples/ssd7_udacity_traffic_pred_02.png?raw=true"></td>
</tr>
</table>

---
### RetinaNet
**Paper:** [arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)<br>
**Blog:** [RetinaNet 介紹](https://gino6178.medium.com/%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-retinanet-%E4%BB%8B%E7%B4%B9-dda4100673bb)
![](https://miro.medium.com/max/622/0*ksQqcCYF0iQN_oX2.png)
從左到右分別用上了<br>
* 殘差網路(Residual Network ResNet)
* 特徵金字塔(Feature Pyramid Network FPN)
* 類別子網路(Class Subnet)
* 框子網路(Box Subnet)
* 以及Anchors

**Blog:** [Review: RetinaNet — Focal Loss](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)
![](https://miro.medium.com/max/600/0*E30eIZ5aCGjhCz9E.gif)
<iframe width="665" height="382" src="https://www.youtube.com/embed/lZxMklxzm2Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
**Code:** [keras-retinanet](https://github.com/fizyr/keras-retinanet)<br>

---
### PointRend
**Paper:** [arxiv.org/abs/1912.08193](https://arxiv.org/abs/1912.08193)<br>
**Blog:** [Facebook PointRend: Rendering Image Segmentation](https://medium.com/syncedreview/facebook-pointrend-rendering-image-segmentation-f3936d50e7f1)
![](https://miro.medium.com/max/606/0*XPUeNroEr8WATWGG.png)
![](https://miro.medium.com/max/674/0*7ed5Nc7fVtSEByt3.png)
![](https://miro.medium.com/max/700/0*P0Mp6DeRC-fHYd8g.png)
**Code:** [keras-retinanet](https://github.com/fizyr/keras-retinanet)<br>

---
### EfficientDet
**Paper:** [arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)<br>
![](https://github.com/google/automl/raw/master/efficientdet/g3doc/network.png)
**Code:** [google efficientdet](https://github.com/google/automl/tree/master/efficientdet)<br>
![](https://github.com/google/automl/blob/master/efficientdet/g3doc/street.jpg?raw=true)
<table>
<tr>
<td><img src="https://github.com/google/automl/blob/master/efficientdet/g3doc/flops.png?raw=true"></td>
<td><img src="https://github.com/google/automl/blob/master/efficientdet/g3doc/params.png?raw=true"></td>
</tr>
</table>
**Kaggle:** [rkuo2000/efficientdet-gwd](https://www.kaggle.com/rkuo2000/efficientdet-gwd)<br>
![](https://camo.githubusercontent.com/f0c80711512aacc0c1423a64e1036608a773f61c0bb6480ec0d57614ce3e7ccc/68747470733a2f2f696d6775722e636f6d2f3579554a4350562e6a7067)

---
### [YOLO- You Only Look Once](https://pjreddie.com/darknet/yolo/)
**Code:** [pjreddie/darknet](https://github.com/pjreddie/darknet)<br>
![](https://pyimagesearch.com/wp-content/uploads/2018/11/yolo_design.jpg)

**[YOLOv1](https://arxiv.org/abs/1506.02640)** : mapping bounding box<br>
![](https://manalelaidouni.github.io/assets/img/pexels/YOLO_arch.png)
**[YOLOv2](https://arxiv.org/abs/1612.08242)** : anchor box proportional to K-means<br>
![](https://2.bp.blogspot.com/-_R-w_tWHdzc/WzJPsol7qFI/AAAAAAABbgg/Jsf-AO3qH0A9oiCeU0LQxN-wdirlOz4WgCLcBGAs/s400/%25E8%259E%25A2%25E5%25B9%2595%25E5%25BF%25AB%25E7%2585%25A7%2B2018-06-26%2B%25E4%25B8%258B%25E5%258D%258810.36.51.png)
**[YOLOv3](https://arxiv.org/abs/1804.02767)** : Darknet-53 + FPN<br>
![](https://media.springernature.com/m685/springer-static/image/art%3A10.1038%2Fs41598-021-81216-5/MediaObjects/41598_2021_81216_Fig1_HTML.png)
![](https://miro.medium.com/max/2000/1*d4Eg17IVJ0L41e7CTWLLSg.png)

---
### YOLObile
**Paper:** [arxiv.org/abs/2009.05697](https://arxiv.org/abs/2009.05697)<br>
**Blog:** [YOLObile：移動設備上的實時目標檢測](https://twgreatdaily.com/zh-hk/jRnp3HQBd8y1i3sJ_wK9.html)<br>
![](https://images.twgreatdaily.com/images/elastic/vhk/vhk33XQBd8y1i3sJeRJg.jpg)
**Code:** [nightsnack/YOLObile](https://github.com/nightsnack/YOLObile)<br>
![](https://github.com/nightsnack/YOLObile/raw/master/figure/yolo_demo.jpg)

---
### YOLOv4
**Paper:** [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)<br>
* YOLOv4 = YOLOv3 + CSPDarknet53 + SPP + PAN + BoF + BoS<br>
![](https://blog.roboflow.com/content/images/2020/06/image-10.png?raw=true)
![](https://www.researchgate.net/profile/Jishu-Miao/publication/349381918/figure/fig4/AS:994706955722753@1614168027527/Normal-YOLOv4-network-architecture.ppm)
* CSP
![](https://blog.roboflow.com/content/images/2020/06/image-15.png)
* PANet
![](https://blog.roboflow.com/content/images/2020/06/image-17.png)

**Code:** [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)<br>
**Code:** [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)<br>

---
### YOLOv5
![](https://user-images.githubusercontent.com/4210061/107134685-4b249480-692f-11eb-93b1-619708d95441.png)
**Code:** [ultralytics/yolov5/](https://github.com/ultralytics/yolov5/)<br>
![](https://user-images.githubusercontent.com/26833433/127574988-6a558aa1-d268-44b9-bf6b-62d4c605cc72.jpg)
![](https://user-images.githubusercontent.com/26833433/136901921-abcfcd9d-f978-4942-9b97-0e3f202907df.png)

---
### Scaled-YOLOv4
**Paper:** [arxiv.org/abs/2011.08036](https://arxiv.org/abs/2011.08036)<br>
![](https://miro.medium.com/max/1838/1*OE4SO1U87DHcAClSZFGlMg.png)
**Code:** [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)<br>

---
### YOLOR : You Only Learn One Representation
**Paper:** [arxiv.org/abs/2105.04206](https://arxiv.org/abs/2105.04206)<br>
![](https://github.com/WongKinYiu/yolor/raw/main/figure/unifued_network.png?raw=true)
**Code:** [WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)<br>
![](https://github.com/WongKinYiu/yolor/blob/main/inference/output/horses.jpg?raw=true)
![](https://github.com/WongKinYiu/yolor/raw/main/figure/performance.png?raw=true)

---
### YOLOX
**Paper:** [arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)<br>
![](https://miro.medium.com/max/915/1*ihnRFgPMgatEtrlTtOM2Bg.png)
**Code:** [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)<br>
![](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/assets/demo.png?raw=true)
![](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/assets/git_fig.png?raw=true)

---
### YOLOv5 vs YOLOX
**Paper:** [Evaluation of YOLO Models with Sliced Inference for Small Object Detection](https://arxiv.org/abs/2203.04799)<br>

![](https://www.researchgate.net/publication/359129591/figure/tbl2/AS:1131979458248722@1646896342669/AP50-scores-for-each-bounding-boxes-size-wise.png)

---
### CSL-YOLO
**Paper:** [arxiv.org/abs/2107.04829](https://arxiv.org/abs/2107.04829)<br>
![](https://www.researchgate.net/publication/353208773/figure/fig1/AS:1044955464224768@1626148205216/Overall-architecture-of-CSL-YOLO-the-convolution-1x1-is-weights-sharing.ppm)
**Code:** [D0352276/CSL-YOLO](https://github.com/D0352276/CSL-YOLO)<br>
![](https://github.com/D0352276/CSL-YOLO/blob/main/demo/result_img_1.png?raw=true)
**Camera Demo**<br>
![](https://github.com/D0352276/CSL-YOLO/blob/main/demo/camera_demo.gif?raw=true)

---
### PP-YOLOE
**Paper:** [PP-YOLOE: An evolved version of YOLO](https://arxiv.org/abs/2203.16250)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PP-YOLOE.png?raw=true)
**Code:**  [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PP-YOLOE_MS_COCO.png?raw=true)
**Kaggle:** [rkuo2000/pp-yoloe](https://www.kaggle.com/code/rkuo2000/pp-yoloe)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PP-YOLOE_demo.jpg?raw=true)

---
### YOLOv6
**Blog:** [YOLOv6：又快又准的目标检测框架开源啦](https://tech.meituan.com/2022/06/23/yolov6-a-fast-and-accurate-target-detection-framework-is-opening-source.html)<br>
* RegVGG是一種簡單又强力的CNN結構，在訓練時使用了性能高的多分支模型，而在推理時使用了速度快、省内存的單路模型，也是更具備速度和精度的均衡。
![](https://p0.meituan.net/travelcube/9f7878c7872787f9b8706b28e5e7c611237315.png)
* EfficientRep將在backbone中stride=2的卷積層换成了stride=2的RepConv層。並且也將CSP-Block修改為RepBlock
![](https://p0.meituan.net/travelcube/8ec8337d37c2545b8fcf355625854802145939.png)
* 同樣為了降低在硬體上的延遲，在Neck上的特徵融合結構中也引入了Rep結構。在Neck中使用的是Rep-PAN。
![](https://p0.meituan.net/travelcube/c37c23c37fd094e05e8cab924659a9d9199592.png)
* 和YOLOX一樣，YOLOv6也對檢測頭近行了解耦，分開了邊框回歸與類别分類的過程。
![](https://pic4.zhimg.com/80/v2-3e5868d7f1aadc76d9b112e4cb79719b_720w.jpg)

**Code:** [meituan/YOLOv6](https://github.com/meituan/YOLOv6)<br>
![](https://github.com/meituan/YOLOv6/raw/main/assets/picture.png)
![](https://p0.meituan.net/travelcube/bc0e60516ae0bcad1c111d7c0c5c3b9e335568.png)
<iframe width="864" height="486" src="https://www.youtube.com/embed/5GXWvoDzpDU" title="YOLOv6 versus YOLOv5 Nano Object Detection models in 4K" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### YOLOv7
**Paper:** [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)<br>
* Extended efficient layer aggregation networks
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv7_E-ELAN.png?raw=true)
* Model scaling for concatenation-based models
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv7_model_scaling.png?raw=true)
* Planned re-parameterized convolution
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv7_planned_reparameterized_model.png?raw=true)
* Coarse for auxiliary and fine for lead head label assigner
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv7_coarse_to_fine_lead_guided_assigner.png?raw=true)

**Code:** [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)<br>
![](https://github.com/WongKinYiu/yolov7/raw/main/figure/performance.png)

---
## Applications
### Localize and Classify Wastes on the Streets
**Paper:** [arxiv.org/abs/1710.11374](https://arxiv.org/abs/1710.11374)<br>
**Model:** GoogLeNet<br>
![](https://d3i71xaburhd42.cloudfront.net/5e409a99833470206dac6cf79a4f857d5436dd4a/2-Figure1-1.png)

---
### Street Litter Detection
**Code:** [isaychris/litter-detection-tensorflow](https://github.com/isaychris/litter-detection-tensorflow)<br>
![](https://camo.githubusercontent.com/ab04d9b6af8e7885d44eb001f38c82a9682c8132a44648f6224eaa393cfba080/68747470733a2f2f692e696d6775722e636f6d2f456671716f536d2e706e67)

---
### [TACO: Trash Annotations in Context](http://tacodataset.org/)
**Paper:** [arxiv.org/abs/2003.06875](https://arxiv.org/abs/2003.06875)<br>
**Code:** [pedropro/TACO](https://github.com/pedropro/TACO)<br>
**Model:** Mask R-CNN
![](https://raw.githubusercontent.com/wiki/pedropro/TACO/images/teaser.gif)

---
### Marine Litter Detection
**Paper:** [arxiv.org/abs/1804.01079](https://arxiv.org/abs/1804.01079)<br>
**Dataset:** [Deep-sea Debris Database](http://www.godac.jamstec.go.jp/catalog/dsdebris/e/)<br>
![](https://d3i71xaburhd42.cloudfront.net/aa9ca01584600207773814660d8ba20a8a830772/6-Figure3-1.png)

---
### Marine Debris Detection
**Ref.** [Detect Marine Debris from Aerial Imagery](https://medium.com/@yhoso/mapping-marine-debris-with-keras-part-1-f485dedf2073)<br>
**Code:** [yhoztak/object_detection](https://github.com/yhoztak/object_detection)<br>
**Model:** RetinaNet
![](https://miro.medium.com/max/700/1*EtGCA8Bux9xJcUaHgs63IA.png)
![](https://miro.medium.com/max/700/1*8hi2MeOFBCNA4B_33I7VaA.png)

---
### UDD dataset
**Paper:** [A New Dataset, Poisson GAN and AquaNet for Underwater Object Grabbing](https://arxiv.org/abs/2003.01446)<br>
**Dataset:** [UDD_Official](https://github.com/chongweiliu/UDD_Official)<br>
Concretely, UDD consists of 3 categories (seacucumber, seaurchin, and scallop) with 2,227 images
![](https://github.com/chongweiliu/UDD_Official/raw/main/results.jpg?raw=true)
![](https://d3i71xaburhd42.cloudfront.net/7edd63a0668014c825a702a156e8aea4e527d57a/2-Figure2-1.png)
![](https://d3i71xaburhd42.cloudfront.net/7edd63a0668014c825a702a156e8aea4e527d57a/4-Figure4-1.png)

---
### Detecting Underwater Objects (DUO)
**Paper:** [A Dataset And Benchmark Of Underwater Object Detection For Robot Picking](https://arxiv.org/abs/2106.05681)<br>
**Dataset:** [DUO](https://drive.google.com/file/d/1w-bWevH7jFs7A1bIBlAOvXOxe2OFSHHs/view)<br>
![](https://d3i71xaburhd42.cloudfront.net/5951ed58d17cc510dd32da3db47c4f0fed08b80e/2-Figure1-1.png)

---
### OpenCV-Python play GTA5
**Ref.** [Reading game frames in Python with OpenCV - Python Plays GTA V](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)<br>
**Code:** [Sentdex/pygta5](https://github.com/Sentdex/pygta5)<br>
<iframe width="670" height="377" src="https://www.youtube.com/embed/VRsmPvu0xj0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### T-CNN : Tubelets with CNN
**Paper:** [arxiv.org/abs/1604.02532](https://arxiv.org/abs/1604.02532)<br>
**Blog:** [人工智慧在太空的應用](https://www.narlabs.org.tw/xcscience/cont?xsmsid=0I148638629329404252&qcat=0I164512713411182211&sid=0J295566068384018349)<br>
![](https://www.narlabs.org.tw/files/file_pool/1/0J295568342633298375/%E5%9C%962.png)
![](https://www.narlabs.org.tw/files/file_pool/1/0J295570208893834419/%E5%9C%963.png)
![](https://www.narlabs.org.tw/files/file_pool/1/0J295570971338287463/%E5%9C%965.png)

---
## Object Tracking Datasets
**Paper:** [Deep Learning in Video Multi-Object Tracking: A Survey](https://arxiv.org/abs/1907.12740)<br>

### [Multiple Object Tracking (MOT)](https://motchallenge.net/)
**[MOT-16](https://motchallenge.net/data/MOT16/)**<br>
![](https://d3i71xaburhd42.cloudfront.net/ac0d88ca5f75a4a80da90365c28fa26f1a26d4c4/3-Figure1-1.png)

---
### Under-water Ojbect Tracking (UOT)
**Paper:** [Underwater Object Tracking Benchmark and Dataset](http://www.hstabstractbook.org/index_htm_files/c-PID6132325.pdf)<br>
**[UOT32](https://www.kaggle.com/landrykezebou/uot32-underwater-object-tracking-dataset)**<br>
**[UOT100](https://www.kaggle.com/landrykezebou/uot100-underwater-object-tracking-dataset)**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/UOT32.png?raw=true)

---
### Re3 : Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects
**Paper:** [arxiv.org/abs/1705.06368](https://arxiv.org/abs/1705-06368)<br>
**Code:** [moorejee/Re3](https://github.com/moorejee/Re3)<br>
![](https://github.com/moorejee/Re3/blob/master/demo/output.gif?raw=true)

---
### Deep SORT
**Paper:** [arxiv.org/abs/1703.07402](https://arxiv.org/abs/1703.07402)<br>
**Code:** [nwojke/deep_sort](https://github.com/nwojke/deep_sort)<br>
**Blog:** [Deep SORT多目标跟踪算法代码解析(上)](https://zhuanlan.zhihu.com/p/133678626)<br>
* **Kalman Filter** to create “Track”,  associate track_i with incoming detection_k
* A distance metric (**squared Mahalanobis distance**) to quantify the association 
* an efficient algorithm (**standard Hungarian algorithm**) to associate the data

---
### YOLOv5 + DeepSort
**Code:** [mikel-brostrom/Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)<br>
![](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/raw/master/MOT16_eval/track_pedestrians.gif?raw=true)
![](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/raw/master/MOT16_eval/track_all.gif?raw=true)

---
### SiamBAN
**Paper:** [arxiv.org/abs/2003.06761](https://arxiv.org/abs/2003.06761)<br>
**Code:** [hqucv/siamban](https://github.com/hqucv/siamban)<br>
**Blog:** [[CVPR2020][SiamBAN] Siamese Box Adaptive Network for Visual Tracking](https://www.bilibili.com/read/cv7541809)
![](https://i0.hdslb.com/bfs/article/357345f94693ef09cd71406530f42c590a756336.png@942w_444h_progressive.webp)
![](https://github.com/hqucv/siamban/blob/master/demo/output/12.gif?raw=true)
![](https://github.com/hqucv/siamban/blob/master/demo/output/34.gif?raw=true)

---
### SiamCAR
**Paper:** [arxiv.org/abs/1911.07241](https://arxiv.org/abs/1911.07241)<br>
**Code:** [ohhhyeahhh/SiamCAR](https://github.com/ohhhyeahhh/SiamCAR)<br>
![](https://media.arxiv-vanity.com/render-output/5247410/x2.png)
![](https://media.arxiv-vanity.com/render-output/5247410/x1.png)

---
### 3D-ZeF
**Paper:** [arxiv.org/abs/2006.08466](https://arxiv.org/abs/2006.08466)<br>
**Code:** [mapeAAU/3D-ZeF](https://github.com/mapeAAU/3D-ZeF)<br>
![](https://vap.aau.dk/wp-content/uploads/2020/05/setup-300x182.png)
![](https://vap.aau.dk/wp-content/uploads/2020/05/bitmap.png)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


