---
layout: post
title: Object Detection
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Introduction to Image Datasets, Object Detection, Object Tracking, and its Applications.

---
## Datasets

### [COCO Dataset](https://cocodataset.org/)
![](https://cocodataset.org/images/coco-examples.jpg)
* Object segmentation
* Recognition in context
* Superpixel stuff segmentation
* 330K images (>200K labeled)
* 1.5 million object instances
* **80** object categories
* 91 stuff categories
* 5 captions per image
* 250,000 people with keypoints 

---
###  [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
* 15,851,536 boxes on **600** categories
* 2,785,498 instance segmentations on **350** categories
* 3,284,280 relationship annotations on 1,466 relationships
* 66,391,027 point-level annotations on 5,827 classes
* 61,404,966 image-level labels on 20,638 classes
* Extension - 478,000 crowdsourced images with 6,000+ categories

[Download](https://storage.googleapis.com/openimages/web/download.html)<br>
Download and Visualize using [FiftyOne](https://voxel51.com/docs/fiftyone/)<br>
![](https://storage.googleapis.com/openimages/web/images/fiftyone.png)

As with any other dataset in the [FiftyOne Dataset Zoo](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/index.html), downloading it is as easy as calling:<br>
`dataset = fiftyone.zoo.load_zoo_dataset("open-images-v6", split="validation")`<br>

```
import fiftyone as fo
import fiftyone.zoo as foz

# List available zoo datasets
print(foz.list_zoo_datasets())

#
# Load the COCO-2017 validation split into a FiftyOne dataset
#
# This will download the dataset from the web, if necessary
#
dataset = foz.load_zoo_dataset("coco-2017", split="validation")

# Give the dataset a new name, and make it persistent so that you can
# work with it in future sessions
dataset.name = "coco-2017-validation-example"
dataset.persistent = True

# Visualize the in the App
session = fo.launch_app(dataset)
```

---
## Object Detection

### Object Detection Milestones
![](https://www.researchgate.net/profile/Zhengxia-Zou-2/publication/333077580/figure/fig2/AS:758306230501380@1557805702766/A-road-map-of-object-detection-Milestone-detectors-in-this-figure-VJ-Det-10-11-HOG.ppm)
![](https://www.researchgate.net/profile/Zhengxia-Zou-2/publication/333077580/figure/fig3/AS:758306234724352@1557805703089/The-accuracy-improvements-of-object-detection-on-VOC07-VOC12-and-MS-COCO-datasets.ppm)

---
### Object Detection Landscape
**Blog:** [The Object Detection Landscape: Accuracy vs Runtime](https://deci.ai/blog/object-detection-landscape-accuracy-vs-runtime/)<br>
![](https://blog.roboflow.com/content/images/2020/06/image-10.png?raw=true)

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
**Paper:** [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)<br>
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
### CornerNet
**Paper:** [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244)<br>
![](https://timy90022.github.io/2019/08/09/CornerNet-Detecting-Objects-as-Paired-Keypoints/1.png)
**Code:** [princeton-vl/CornerNet](https://github.com/princeton-vl/CornerNet)<br>

---
### CenterNet
**Paper:** [CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189)<br>
![](https://deci.ai/wp-content/uploads/2021/05/7-Architecture-of-CenterNet.png)
**Code:** [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)<br>

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
![](https://www.researchgate.net/profile/Jishu-Miao/publication/349381918/figure/fig4/AS:994706955722753@1614168027527/Normal-YOLOv4-network-architecture.ppm)
* CSP
![](https://blog.roboflow.com/content/images/2020/06/image-15.png)
* PANet
![](https://blog.roboflow.com/content/images/2020/06/image-17.png)

**Code:** [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)<br>
**Code:** [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)<br>

---
### [YOLOv5](https://docs.ultralytics.com/yolov5/)
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
![](https://github.com/rkuo2000/AI-course/blob/main/images/PP-YOLOE.png?raw=true)
**Code:**  [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/PP-YOLOE_MS_COCO.png?raw=true)
**Kaggle:** [rkuo2000/pp-yoloe](https://www.kaggle.com/code/rkuo2000/pp-yoloe)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/PP-YOLOE_demo.jpg?raw=true)

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
![](https://github.com/rkuo2000/AI-course/blob/main/images/YOLOv7_E-ELAN.png?raw=true)
* Model scaling for concatenation-based models
![](https://github.com/rkuo2000/AI-course/blob/main/images/YOLOv7_model_scaling.png?raw=true)
* Planned re-parameterized convolution
![](https://github.com/rkuo2000/AI-course/blob/main/images/YOLOv7_planned_reparameterized_model.png?raw=true)
* Coarse for auxiliary and fine for lead head label assigner
![](https://github.com/rkuo2000/AI-course/blob/main/images/YOLOv7_coarse_to_fine_lead_guided_assigner.png?raw=true)

**Code:** [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)<br>
![](https://github.com/WongKinYiu/yolov7/raw/main/figure/performance.png)

---
### YOLOv8
[Ultralytics YOLOv8](https://www.ultralytics.com/) is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

**Blog:** [Dive into YOLOv8](https://openmmlab.medium.com/dive-into-yolov8-how-does-this-state-of-the-art-model-work-10f18f74bab1)<br>
![](https://miro.medium.com/v2/resize:fit:720/0*lCMq5uVmQkhV3UY_)

**Code:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)<br>

**Kaggle:** <br>
* [https://www.kaggle.com/code/rkuo2000/yolov8](https://www.kaggle.com/code/rkuo2000/yolov8)
* [https://www.kaggle.com/code/rkuo2000/yolov8-pothole-detection](https://www.kaggle.com/code/rkuo2000/yolov8-pothole-detection)

**Paper:** [Real-Time Flying Object Detection with YOLOv8](https://arxiv.org/abs/2305.09972)<br>
![](https://user-images.githubusercontent.com/27466624/239739723-57391d0f-1848-4388-9f30-88c2fb79233f.jpg)

**Dataset:** [flying_object_dataset](https://universe.roboflow.com/new-workspace-0k81p/flying_object_dataset)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/flying_object_dataset.png?raw=true)

**Paper:** [UAV-YOLOv8: A Small-Object-Detection Model Based on Improved YOLOv8 for UAV Aerial Photography Scenarios](https://www.mdpi.com/1424-8220/23/16/7190)<br>
![](https://www.mdpi.com/sensors/sensors-23-07190/article_deploy/html/images/sensors-23-07190-g001.png)

---
## Trash Detection
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
![](https://github.com/rkuo2000/AI-course/blob/main/images/UOT32.png?raw=true)

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
### SiamCAR
**Paper:** [arxiv.org/abs/1911.07241](https://arxiv.org/abs/1911.07241)<br>
**Code:** [ohhhyeahhh/SiamCAR](https://github.com/ohhhyeahhh/SiamCAR)<br>
![](https://media.arxiv-vanity.com/render-output/5247410/x2.png)
![](https://media.arxiv-vanity.com/render-output/5247410/x1.png)

---
### YOLOv5 + DeepSort
**Code:** [HowieMa/DeepSORT_YOLOv5_Pytorch](https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch)<br>
![](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/raw/master/MOT16_eval/track_pedestrians.gif?raw=true)

---
### Yolov5 + StrongSORT with OSNet
**Code:** [Yolov5_StrongSORT_OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)<br>
<table>
<tr>
<td><img src="https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/raw/master/strong_sort/results/output_04.gif"></td>
<td><img src="https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/raw/master/strong_sort/results/output_th025.gif"></td>
</tr>
</table>

---
### SiamBAN
**Paper:** [arxiv.org/abs/2003.06761](https://arxiv.org/abs/2003.06761)<br>
**Code:** [hqucv/siamban](https://github.com/hqucv/siamban)<br>
**Blog:** [[CVPR2020][SiamBAN] Siamese Box Adaptive Network for Visual Tracking](https://www.bilibili.com/read/cv7541809)
![](https://i0.hdslb.com/bfs/article/357345f94693ef09cd71406530f42c590a756336.png@942w_444h_progressive.webp)
![](https://github.com/hqucv/siamban/blob/master/demo/output/12.gif?raw=true)
![](https://github.com/hqucv/siamban/blob/master/demo/output/34.gif?raw=true)

---
### FairMOT
**Paper:** [FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888)<br>
**Code:** [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)<br>
![](https://github.com/ifzhang/FairMOT/blob/master/assets/pipeline.png?raw=true)

---
### 3D-ZeF
**Paper:** [arxiv.org/abs/2006.08466](https://arxiv.org/abs/2006.08466)<br>
**Code:** [mapeAAU/3D-ZeF](https://github.com/mapeAAU/3D-ZeF)<br>
![](https://vap.aau.dk/wp-content/uploads/2020/05/setup-300x182.png)
![](https://vap.aau.dk/wp-content/uploads/2020/05/bitmap.png)

---
## UAV-based Object Detection and Tracking
**Paper:** [Deep Learning for UAV-based Object Detection and Tracking: A Survey](https://arxiv.org/abs/2110.12638)<br>

---
### Efficient Object Detection Model for Real-Time UAV Applications
**Paper:** [Efficient Object Detection Model for Real-Time UAV Applications](https://arxiv.org/abs/1906.00786)<br>

---
## [Satellite Image Deep Learning](https://github.com/robmarkcole/satellite-image-deep-learning)

### T-CNN : Tubelets with CNN
**Paper:** [arxiv.org/abs/1604.02532](https://arxiv.org/abs/1604.02532)<br>
**Blog:** [人工智慧在太空的應用](https://www.narlabs.org.tw/xcscience/cont?xsmsid=0I148638629329404252&qcat=0I164512713411182211&sid=0J295566068384018349)<br>
![](https://www.narlabs.org.tw/files/file_pool/1/0J295568342633298375/%E5%9C%962.png)
![](https://www.narlabs.org.tw/files/file_pool/1/0J295570208893834419/%E5%9C%963.png)
![](https://www.narlabs.org.tw/files/file_pool/1/0J295570971338287463/%E5%9C%965.png)

---
### Swimming Pool Detection
**Dataset:** [Aerial images of swimming pools](https://www.kaggle.com/datasets/alexj21/swimming-pool-512x512)<br>
**Kaggle:** [Evaluation Efficientdet - Swimming Pool Detection](https://www.kaggle.com/code/alexj21/evaluation-efficientdet-swimming-pool-detection)<br>
![](https://cdn-images-1.medium.com/max/1200/0*kbWFs2dOILw7abke.png)

---
### Pool-Detection 
**Code:** [https://github.com/yacine-benbaccar/Pool-Detection](https://github.com/yacine-benbaccar/Pool-Detection)<br>
![](https://github.com/yacine-benbaccar/Pool-Detection/raw/6f0163171b6b9ed429a60f43fa7b861a9b459d20//README/pooldetection_th%3D0.75_zone18.jpg)

---
### Identify Military Vehicles in Satellite Imagery
**Blog:** [Identify Military Vehicles in Satellite Imagery with TensorFlow](https://python.plainenglish.io/identifying-military-vehicles-in-satellite-imagery-with-tensorflow-96015634129d)<br>
**Dataset:** [Moving and Stationary Target Acquisition and Recognition (MSTAR) Dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar)<br>
![](https://github.com/NateDiR/sar_target_recognition_deep_learning/raw/main/images/mstar_example.png)
The MSTAR dataset was prdocued between 1995-1997 in a collaboration between DARPA/Air Force Research Laboratories. It contains roughly 1000 SAR images of 8 different vehicle types:<br>
1. <u>2S1 Gvozdika</u> — Self-propelled artillery
2. <u>ZSU-23–4 Shilka</u> — Self-propelled anti-aircraft
3. <u>BRDM-2</u> — Amphibious armored scout car
4. <u>BTR-60</u> — Armored personnel carrier
5. <u>D7</u> — Caterpillar Bulldozer
6. <u>ZIL-131</u> — Military cargo truck
7. <u>T-62</u> — Main battle tank
8. <u>T-72</u> — main battle tank (2nd Gen)
9. <u>SLICY</u> — Structure acting as a ‘ground truth’ (not a vehicle)

**Code:** [Target Recognition in Sythentic Aperture Radar Imagery Using Deep Learning](https://github.com/NateDiR/sar_target_recognition_deep_learning)<br>
[script.ipynb](https://github.com/NateDiR/sar_target_recognition_deep_learning/blob/main/script.ipynb)<br>

---
## Steel Defect Detection
**Dataset:** [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)<br>
![](https://diyago.github.io/images/kaggle-severstal/input_data.png)

### Steel Defect Detection using UNet
**Kaggle:** [https://www.kaggle.com/code/jaysmit/u-net (Keras UNet)](https://www.kaggle.com/code/jaysmit/u-net)<br>
**Kaggle:** [https://www.kaggle.com/code/myominhtet/steel-defection (pytorch UNet](https://www.kaggle.com/code/myominhtet/steel-defection)<br>

### Steel-Defect Detection Using CNN
**Code:** [https://github.com/himasha0421/Steel-Defect-Detection](https://github.com/himasha0421/Steel-Defect-Detection)<br>
![](https://github.com/himasha0421/Steel-Defect-Detection/raw/main/images/unet_img.png)
![](https://github.com/himasha0421/Steel-Defect-Detection/raw/main/images/img_1.png)

### MSFT-YOLO
**Paper:** [MSFT-YOLO: Improved YOLOv5 Based on Transformer for Detecting Defects of Steel Surface](https://www.mdpi.com/1424-8220/22/9/3467)<br>
![](https://www.mdpi.com/sensors/sensors-22-03467/article_deploy/html/images/sensors-22-03467-g002-550.jpg)
![](https://www.mdpi.com/sensors/sensors-22-03467/article_deploy/html/images/sensors-22-03467-g006-550.jpg)

---
## PCB Defect Detection
### PCB Datasets
* [DeepPCB](https://github.com/tangsanli5201/DeepPCB)<br>
![](https://github.com/tangsanli5201/DeepPCB/raw/master/fig/test.jpg)
* [HRIPCB](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/joe.2019.1183)<br>
![](https://ietresearch.onlinelibrary.wiley.com/cms/asset/3130c389-d389-49c9-9ce8-61e624c5a794/tje2bf02982-fig-0003-m.png)

---
### PCB Defect Detection
**Paper:** [PCB Defect Detection Using Denoising Convolutional Autoencoders](https://arxiv.org/abs/2008.12589)<br>
![](https://www.researchgate.net/publication/343986672/figure/fig1/AS:930433990672384@1598844158824/The-overview-of-the-system-a-salt-and-pepper-noise-is-added-to-defective-PCBs-b.ppm)
![](https://www.researchgate.net/publication/343986672/figure/fig4/AS:930433994878976@1598844159541/examples-of-the-output-of-the-proposed-model.ppm)

---
### PCB Defect Classification
**Dataset:** [HRIPCB dataset (dropbox)](https://www.dropbox.com/s/h0f39nyotddibsb/VOC_PCB.zip?dl=0)<br>
印刷电路板（PCB）瑕疵数据集。它是一个公共合成PCB数据集，包含1386张图像，具有6种缺陷（漏孔、鼠咬、开路、短路、杂散、杂铜），用于图像检测、分类和配准任务。<br>
**Paper:** [End-to-end deep learning framework for printed circuit board manufacturing defect classification](https://www.nature.com/articles/s41598-022-16302-3)<br>
![](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-022-16302-3/MediaObjects/41598_2022_16302_Fig2_HTML.png?as=webp)
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-022-16302-3/MediaObjects/41598_2022_16302_Fig4_HTML.png?as=webp)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


