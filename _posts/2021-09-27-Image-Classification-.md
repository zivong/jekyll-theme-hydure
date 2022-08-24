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

---
### [ImageNet](http://www.image-net.org/)
![](https://miro.medium.com/max/700/1*IlzW43-NtJrwqtt5Xy3ISA.jpeg)
![](https://devopedia.org/images/article/172/7316.1561043304.png)
14,197,122 images, 21841 synsets indexed <br>
[Download](http://image-net.org/download-imageurls)<br>

---
### [Open Images V6+](https://storage.googleapis.com/openimages/web/index.html)
* Blog: [Open Images V6 — Now Featuring Localized Narratives](https://ai.googleblog.com/2020/02/open-images-v6-now-featuring-localized.html)
These annotation files cover the 600 boxable object classes, and span the 1,743,042 training images where we annotated bounding boxes, object segmentations, visual relationships, and localized narratives; as well as the full validation (41,620 images) and test (125,436 images) sets.<br>
[Download](https://storage.googleapis.com/openimages/web/download.html)<br>
![](https://1.bp.blogspot.com/-yuodfZa6gyM/XlbQfiAzbzI/AAAAAAAAFYA/QSTnuZksQII2PaRON2mqHntZBHL-saniACLcBGAsYHQ/s640/Figure1.png)

---
## Applications

### CIFAR-10
Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CIFAR-10.png?raw=true)
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
Kaggle: [https://www.kaggle.com/rkuo2000/cifar10-cnn](https://www.kaggle.com/rkuo2000/cifar10-cnn)<br>

---
### Traffic Sign Classifier (交通號誌辨識)
Dataset: [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html)<br>
![](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/61e9ce225148f6519be6c034_GTSRB-0000000633-9ce3c5f6_Dki5Rsf.jpeg)
34 traffic signs, 39209 training images, 12630 test images
Kaggle: [https://www.kaggle.com/rkuo2000/gtsrb-cnn](https://www.kaggle.com/rkuo2000/gtsrb-cnn)<br>

---
### Emotion Detection (情緒偵測)
Dataset:[FER-2013 (Facial Expression Recognition)](https://www.kaggle.com/datasets/msambare/fer2013)<br>
![](https://production-media.paperswithcode.com/datasets/FER2013-0000001434-01251bb8_415HDzL.jpg)
7 facial expression, 28709 training images, 7178 test images<br>
labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]<br>
Kaggle: [https://www.kaggle.com/rkuo2000/fer2013-cnn](https://www.kaggle.com/rkuo2000/fer2013-cnn)<br>

---
### Pneumonia Detection (肺炎偵測)
Dataset: [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)<br>
![](https://raw.githubusercontent.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning/master/demo/sample/sample.png)
Kaggle: [https://www.kaggle.com/rkuo2000/pneumonia-cnn](https://www.kaggle.com/rkuo2000/pneumonia-cnn)<br>

---
### COVID19 Detection (新冠肺炎偵測)
Dataset: [https://www.kaggle.com/bachrr/covid-chest-xray](https://www.kaggle.com/bachrr/covid-chest-xray)<br>
![](https://i.imgur.com/jZqpV51.png)
Kaggle: [https://www.kaggle.com/rkuo2000/covid19-vgg16](https://www.kaggle.com/rkuo2000/covid19-vgg16)<br>

---
### Skin Lesion Classification (皮膚病變分類)
Dataset : [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ham10000_dataset.png?raw=true)
7 types of lesions : (picture = 600x450)<br>
* Actinic Keratoses (光化角化病)
* Basal Cell Carcinoma (基底細胞癌)
* Benign Keratosis (良性角化病)
* Dermatofibroma (皮膚纖維瘤)
* Malignant Melanoma (惡性黑色素瘤)
* Melanocytic Nevi (黑素細胞痣)
* Vascular Lesions (血管病變)

Kaggle: [https://www.kaggle.com/rkuo2000/skin-lesion-cnn](https://www.kaggle.com/rkuo2000/skin-lesion-cnn)<br>

---
### Garbage Classification (垃圾分類)
Dataset: https://www.kaggle.com/asdasdasasdas/garbage-classification (42MB)<br>
<img widtih="50%" height="50%" src="https://miro.medium.com/max/2920/1*mJipx8yxeI_JW36jDAuM9A.png">
6 categories : cardboard(403), glass(501), metal(410), paper (594), plastic(482), trash(137)<br>

Kaggle: [https://www.kaggle.com/rkuo2000/garbage-cnn](https://www.kaggle.com/rkuo2000/garbage-cnn)<br>

---
### Food Classification  (食物分類)
Dataset: [Food-11](https://mmspg.epfl.ch/downloads/food-image-datasets/)<br>
![](https://929687.smushcdn.com/2633864/wp-content/uploads/2019/06/fine_tuning_keras_food11.jpg?lossy=1&strip=1&webp=1)
The dataset consists of 16,643 images belonging to 11 major food categories:<br>
* Bread (1724 images)
* Dairy product (721 images)
* Dessert (2,500 images)
* Egg (1,648 images)
* Fried food (1,461images)
* Meat (2,206 images)
* Noodles/pasta (734 images)
* Rice (472 images)
* Seafood (1,505 images)
* Soup (2,500 images)
* Vegetable/fruit (1,172 images)

Kaggle: [https://www.kaggle.com/rkuo2000/food11-classification](https://www.kaggle.com/rkuo2000/food11-classification)<br>

---
### Mango Classification (芒果分類)
Dataset: [台灣高經濟作物 - 愛文芒果影像辨識正式賽](https://aidea-web.tw/aicup_mango)<br>
Kaggle: <br>
* [https://www.kaggle.com/rkuo2000/mango-classification](https://www.kaggle.com/rkuo2000/mango-classification)
* [https://www.kaggle.com/rkuo2000/mango-efficientnet](https://www.kaggle.com/rkuo2000/mango-efficientnet)

---
### FaceMask Classification (人臉口罩辨識)
Dataset: [Face Mask ~12K Images dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/facemask_12k_dataset.png?raw=true)
Kaggle: [https://www.kaggle.com/rkuo2000/facemask-cnn](https://www.kaggle.com/rkuo2000/facemask-cnn)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

