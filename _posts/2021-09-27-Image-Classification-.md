---
layout: post
title: Image Classification
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Image Classification models and applications

---
## Datasets

### [PASCAL VOC (Visual Ojbect Classes)](http://host.robots.ox.ac.uk/pascal/VOC/)
VOC2007 train/val/test 9,963張標註圖片，有24,640個標註物件<br> 
VOC2012 train/val/test11,530張標註圖片，有27,450個ROI 標註物件<br>
![](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/segexamples/images/006585_object.png)
![](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/layoutexamples/images/08_parts.jpg)
20 classes:
* Person: person
* Animal: bird, cat, cow, dog, horse, sheep
* Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
* Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

---
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

### [ImageNet](http://www.image-net.org/)
![](https://miro.medium.com/max/700/1*IlzW43-NtJrwqtt5Xy3ISA.jpeg)
![](https://devopedia.org/images/article/172/7316.1561043304.png)
14,197,122 images, 21841 synsets indexed <br>
[Download](http://image-net.org/download-imageurls)<br>

---
### [Open Images V6+](https://storage.googleapis.com/openimages/web/index.html)
* Blog: [Open Images V6 — Now Featuring Localized Narratives](https://ai.googleblog.com/2020/02/open-images-v6-now-featuring-localized.html)
These annotation files cover the 600 boxable object classes, and span the 1,743,042 training images where we annotated bounding boxes, object segmentations, visual relationships, and localized narratives; as well as the full validation (41,620 images) and test (125,436 images) sets.<br>
[Download](Download: https://storage.googleapis.com/openimages/web/download.html)<br>
![](https://1.bp.blogspot.com/-yuodfZa6gyM/XlbQfiAzbzI/AAAAAAAAFYA/QSTnuZksQII2PaRON2mqHntZBHL-saniACLcBGAsYHQ/s640/Figure1.png)

---
## Applications

### Speech Commands Recognition (語音命令辨識)

---
### Urban Sound Classification (城市聲音分類)

---
### Traffic Sign Classifier (交通號誌辨識)

---
### Emotion Detection (情緒偵測)

---
### Pneumonia Detection (肺炎偵測)

---
### COVID19 Detection (武漢肺炎偵測)

---
### Skin Lesion Classification (皮膚病變分類)

---
### Garbage Classification (垃圾分類)

---
### Food Classification  (食物分類)

---
### Mango Classification (芒果分類)

---
### FaceMask Classification (人臉口罩辨識)


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

