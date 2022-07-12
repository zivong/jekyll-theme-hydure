---
layout: post
title: Object Detection Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes Image Annotation, YOLOv4, YOLOv5, YOLOR, YOLOX, CSL-YOLO, YOLOv6, YOLOv7, Mask RCNN, SSD MobileNet, YOLOv5+DeepSort, Objectron.

---
## Image Annotation
### [labelme](https://github.com/wkentaro/labelme)
![](https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/.readme/annotation.jpg?raw=true)
`$pip install labelme`<br>

---
### [LabelImg](https://github.com/tzutalin/labelImg)
![](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)
`$pip install labelImg`<br>

`$labelImg`<br>
`$labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]`<br>
---
### VOC .xml convert to YOLO .txt
`$cd ~/tf/raccoon/annotations`
`$python ~/tf/xml2yolo.py`

---
### YOLOv4
**Kaggle:** [rkuo2000/yolov4](https://kaggle.com/rkuo2000/yolov4)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv4_PyTorch_horses.jpg?raw=true)

### YOLOv5
**Kaggle:** [rkuo2000/yolov5](https://kaggle.com/rkuo2000/yolov5)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_horses.jpg?raw=true)

### Scaled YOLOv4
**Kaggle:** [rkuo2000/scaled-yolov4](https://kaggle.com/rkuo2000/scaled-yolov4)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Scaled_YOLOv4_horses.jpg?raw=true)

### YOLOR
**Kaggle:** [rkuo2000/yolor](https://kaggle.com/rkuo2000/yolor)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOR_horses.jpg?raw=true)

### YOLOX
**Kaggle:** [rkuo2000/yolox](https://www.kaggle.com/code/rkuo2000/yolox)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOX_horses.jpg?raw=true)

### CSL-YOLO
**Kaggle:** [rkuo2000/csl-yolo](https://kaggle.com/rkuo2000/csl-yolo)
![](https://github.com/D0352276/CSL-YOLO/blob/main/dataset/coco/pred/000000000001.jpg?raw=true)

### PP-YOLOE
**Kaggle:** [rkuo2000/pp-yoloe](https://www.kaggle.com/code/rkuo2000/pp-yoloe)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PP-YOLOE_demo.jpg?raw=true)

### YOLOv6
**Kaggle:** [rkuo2000/yolov6](https://www.kaggle.com/code/rkuo2000/yolov6)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6_image2.png?raw=true)

### YOLOv7
**Kaggle:** [rkuo2000/yolov7](https://www.kaggle.com/code/rkuo2000/yolov7)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv7_horses.jpg?raw=true)

---
### YOLOv5 applications
**[YOLOv5 Detect](https://kaggle.com/rkuo2000/yolov5-detect)**<br>
detect image / video
<iframe width="498" height="280" src="https://www.youtube.com/embed/IL9GdRQrI-8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
**[YOLOv5 Elephant](https://kaggle.com/rkuo2000/yolov5-elephant)**<br>
train YOLOv5 for detecting elephant (dataset from OpenImage V6)
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_elephant.jpg?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_elephants.jpg?raw=true"></td>
</tr>
</table>

**[YOLOv5 BCCD](https://kaggle.com/rkuo2000/yolov5-bccd)**<br>
train YOLOv5 for Blood Cells Detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_BCCD.jpg?raw=true)

---
**[YOLOv5 Helmet](https://kaggle.com/rkuo2000/yolov5-helmet)**<br>
train YOLOv5 for Helmet detection
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Helmet.jpg?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Helmet_SafeZone.jpg?raw=true"></td>
</tr>
</table>

**[YOLOv5 Facemask](https://kaggle.com/rkuo2000/yolov5-facemask)**<br>
train YOLOv5 for facemask detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Facemask.jpg?raw=true)

**[YOLOv5 Traffic Analysis](https://kaggle.com/rkuo2000/yolov5-traffic-analysis)**<br>
use YOLOv5 to detect car/truck per frame, then analyze vehicle counts per lane and the estimated speed
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_traffic_analysis.jpg?raw=true)

**[YOLOv5 Global Wheat Detection](https://www.kaggle.com/rkuo2000/yolov5-global-wheat-detection)**<br>
train YOLOv5 for wheat detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_GWD.jpg?raw=true)

**[EfficientDet Global Wheat Detection](https://www.kaggle.com/rkuo2000/efficientdet-gwd)**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EfficientDet_GWD.png?raw=true)

---
## Mask R-CNN
**Kaggle:** [rkuo2000/mask-rcnn](https://www.kaggle.com/rkuo2000/mask-rcnn)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Mask_RCNN_TF2.png?raw=true)

### Mask R-CNN transfer learning
**Kaggle:** [Mask RCNN transfer learning](https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Mask_RCNN_transfer_learning.png?raw=true)

---
### YOLOv5 + DeepSort
**[YOLOv5 DeepSort](https://kaggle.com/rkuo2000/yolov5-deepsort)**<br>
<iframe width="574" height="323" src="https://www.youtube.com/embed/-NHq7yUAY7U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="498" height="280" src="https://www.youtube.com/embed/RKVrtJs1ry8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Objectron
**Kaggle:** [rkuo2000/mediapipe-objectron](https://www.kaggle.com/rkuo2000/mediapipe-objectron)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Objectron_shoes.png?raw=true)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

