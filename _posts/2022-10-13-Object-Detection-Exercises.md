---
layout: post
title: Object Detection Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes Image Annotation, YOLOv4, YOLOv5, YOLOR, YOLOX, CSL-YOLO, YOLOv6, YOLOv7, Mask RCNN, SSD MobileNet, YOLOv5+DeepSort, Objectron, Steel Defect Detection, PCB Defect Detection, Identify Military Vehicles in Satellite Imagery, Pothole Detection, Car Breaking Detection.

---
## Image Annotation
### [FiftyOne](https://voxel51.com/docs/fiftyone/)
[Annotating Datasets with LabelBox](https://voxel51.com/docs/fiftyone/tutorials/labelbox_annotation.html)<br>
To get started, you need to [install FiftyOne](https://voxel51.com/docs/fiftyone/getting_started/install.html) and [the Labelbox Python client](https://github.com/Labelbox/labelbox-python):<br>
`!pip install fiftyone labelbox`<br>
![](https://voxel51.com/docs/fiftyone/_images/labelbox_detection.png)

---
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
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5s_horses.jpg?raw=true)

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
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6s_image1.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6s_image2.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6s_horses.png?raw=true)

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

---
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

---
**[YOLOv5 Facemask](https://kaggle.com/rkuo2000/yolov5-facemask)**<br>
train YOLOv5 for facemask detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Facemask.jpg?raw=true)

---
**[YOLOv5 Traffic Analysis](https://kaggle.com/rkuo2000/yolov5-traffic-analysis)**<br>
use YOLOv5 to detect car/truck per frame, then analyze vehicle counts per lane and the estimated speed
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_traffic_analysis.jpg?raw=true)

---
**[YOLOv5 Global Wheat Detection](https://www.kaggle.com/rkuo2000/yolov5-global-wheat-detection)**<br>
train YOLOv5 for wheat detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_GWD.jpg?raw=true)

---
**[EfficientDet Global Wheat Detection](https://www.kaggle.com/rkuo2000/efficientdet-gwd)**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EfficientDet_GWD.png?raw=true)

---
## Mask R-CNN
**Kaggle:** [rkuo2000/mask-rcnn](https://www.kaggle.com/rkuo2000/mask-rcnn)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Mask_RCNN_TF2.png?raw=true)

---
### Mask R-CNN transfer learning
**Kaggle:** [Mask RCNN transfer learning](https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Mask_RCNN_transfer_learning.png?raw=true)

---
### YOLOv5 + DeepSort
**Kaggle:** [YOLOv5 DeepSort](https://kaggle.com/rkuo2000/yolov5-deepsort)<br>
<iframe width="574" height="323" src="https://www.youtube.com/embed/-NHq7yUAY7U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="498" height="280" src="https://www.youtube.com/embed/RKVrtJs1ry8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Objectron
**Kaggle:** [rkuo2000/mediapipe-objectron](https://www.kaggle.com/rkuo2000/mediapipe-objectron)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Objectron_shoes.png?raw=true)

---
### OpenCV-Python play GTA5
**Ref.** [Reading game frames in Python with OpenCV - Python Plays GTA V](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)<br>
**Code:** [Sentdex/pygta5](https://github.com/Sentdex/pygta5)<br>
<iframe width="670" height="377" src="https://www.youtube.com/embed/VRsmPvu0xj0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Steel Defect Detection
**Dataset:** [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)<br>
![](https://diyago.github.io/images/kaggle-severstal/input_data.png)
**Kaggle:** [https://www.kaggle.com/code/jaysmit/u-net (Keras UNet)](https://www.kaggle.com/code/jaysmit/u-net)<br>

---
### PCB Defect Detection
**Dataset:** [HRIPCB dataset (dropbox)](https://www.dropbox.com/s/h0f39nyotddibsb/VOC_PCB.zip?dl=0)<br>
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-022-16302-3/MediaObjects/41598_2022_16302_Fig4_HTML.png?as=webp)

---
### Identify Military Vehicles in Satellite Imagery
**Blog:** [Identify Military Vehicles in Satellite Imagery with TensorFlow](https://python.plainenglish.io/identifying-military-vehicles-in-satellite-imagery-with-tensorflow-96015634129d)<br>
**Dataset:** [Moving and Stationary Target Acquisition and Recognition (MSTAR) Dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar)<br>
![](https://github.com/NateDiR/sar_target_recognition_deep_learning/raw/main/images/mstar_example.png)

---
### Pothole Detection
**Blog:** [Pothole Detection using YOLOv4](https://learnopencv.com/pothole-detection-using-yolov4-and-darknet/?ck_subscriber_id=638701084)<br>
**Code:** [yolov4_pothole_detection.ipynb](https://github.com/spmallick/learnopencv/blob/master/Pothole-Detection-using-YOLOv4-and-Darknet/jupyter_notebook/yolov4_pothole_detection.ipynb)<br>
**Kaggle:** [YOLOv7 Pothole Detection](https://www.kaggle.com/code/rkuo2000/yolov7-pothole-detection)
![](https://learnopencv.com/wp-content/uploads/2022/07/Pothole-Detection-using-YOLOv4-and-Darknet.gif)

---
### Car Breaking Detection
**Code**: [YOLOv7 Braking Detection](https://github.com/ArmaanSinghSandhu/YOLOv7-Braking-Detection)<br>
![](https://github.com/ArmaanSinghSandhu/YOLOv7-Braking-Detection/raw/main/results/Detection.gif)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

