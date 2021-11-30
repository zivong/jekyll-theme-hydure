---
layout: post
title: Face Recognition Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Face Datasets, Face Detection, Face Alignment, Face Landmark and Face Recognition.

---
![](https://machinelearningmastery.com/wp-content/uploads/2019/03/Overview-of-the-Steps-in-a-Face-Recognition-Process-1024x362.png)

---
## Face Datasets

### [VGGFace](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)<br>
[vgg_face_dataset.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz)<br>
The VGG Face dataset is face identity recognition dataset that consists of 2,622 identities. It contains over 2.6 million images

---
### [VGGFace2](https://github.com/ox-vgg/vgg_face2)
**Paper:** [VGGFace2: A dataset for recognising faces across pose and age](https://arxiv.org/abs/1710.08092)<br>

![](https://github.com/ox-vgg/vgg_face2/blob/master/web_page_img.png?raw=true)

---
### CelebA
**[Large-scale CelebFaces Attributes dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**<br>
**Paper:** [Deep Learning Face Attributes in the Wild](https://arxiv.org/abs/1411.7766)<br>

![](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png)

---
### LFW 
**[Labeled Faces in the Wild Home](http://vis-www.cs.umass.edu/lfw)** [(lfw.tgz)](http://vis-www.cs.umass.edu/lfw/lfw.tgz)<br>
**Paper:** [Labeled Faces in the Wild: A Database for Studying
Face Recognition in Unconstrained Environments](http://vis-www.cs.umass.edu/lfw/lfw.pdf)<br>
<br>
The LFW dataset contains 13,233 images of faces collected from the web. This dataset consists of the 5749 identities with 1680 people with two or more images. In the standard LFW evaluation protocol the verification accuracies are reported on 6000 face pairs.
![](https://production-media.paperswithcode.com/datasets/LFW-0000000022-7647ef6f_M2DdqYg.jpg)

---
## Face Detection

### MTCNN
**Paper:** [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)<br>
**Code:** [ipazc/mtcnn](https://github.com/ipazc/mtcnn)<br>
`$pip install mtcnn`<br>
![](https://miro.medium.com/max/490/1*mH7AABb6oha9g6v9jB4gjw.png)

---
### DeepFace
**Code:** [RiweiChen/DeepFace](https://github.com/RiweiChen/DeepFace)<br>
![](https://github.com/RiweiChen/DeepFace/raw/master/FaceAlignment/figures/deepid.png)
5 key points detection using DeepID architecture<br>
![](https://github.com/RiweiChen/DeepFace/raw/master/FaceAlignment/result/1.png)
![](https://github.com/RiweiChen/DeepFace/blob/master/FaceDetection/result/1.jpeg?raw=true)

---
## Face Alignment

### Face-Alignment
**Code:** [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)<br>

![](https://github.com/1adrianb/face-alignment/blob/master/docs/images/face-alignment-adrian.gif?raw=true)
```
import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('../test/assets/aflw-test.jpg')
preds = fa.get_landmarks(input)
```

---
### OpenFace
**Code:** [cmusatyalab/openface](https://github.com/cmusatyalab/openface)<br>
**Ref.** [Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)<br>

![](https://miro.medium.com/max/700/1*6xgev0r-qn4oR88FrW6fiA.png)
![](https://miro.medium.com/max/1400/1*woPojJbd6lT7CFZ9lHRVDw.gif)

---
## Face Landmark

### Face Landmark Estimation
**Paper:** [One Millisecond Face Alignment with an Ensemble of Regression Trees](https://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf)<br>
The basic idea is we will come up with 68 specific points (called landmarks) that exist on every face 
![](https://miro.medium.com/max/414/1*AbEg31EgkbXSQehuNJBlWg.png)

---
### 3D Face Reconstruction & Alignment
**Paper:** [Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network](https://arxiv.org/abs/1707.05653)<br>
**Code:** [YadiraF/PRNet](https://github.com/YadiraF/PRNet)<br>

![](https://github.com/YadiraF/PRNet/blob/master/Docs/images/prnet.gif?raw=true)
**3D Head Pose Estimation**<br>
![](https://github.com/YadiraF/PRNet/blob/master/Docs/images/pose.jpg?raw=true)

---
### EfficientFAN
**Paper:** [EfficientFAN: Deep Knowledge Transfer for Face Alignment](https://dl.acm.org/doi/abs/10.1145/3372278.3390692)<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EfficentFAN.png?raw=true)

---
## Face Recognition

### DeepID
**Paper:**[DeepID3: Face Recognition with Very Deep Neural Networks](https://arxiv.org/abs/1502.00873)<br>
**Code:** [hqli/face_recognition](https://github.com/hqli/face_recognition)<br>
**Blog:** [DeepID人臉識別算法之三代](https://read01.com/zh-tw/J8BJ8L.html#.YaW0m5FBxH7)<br>
**Ref.** [DeepID3 face recognition](https://www.twblogs.net/a/5b82951f2b717766a1e8f563)<br>
![](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdn.net/20160807223022319)
DeepID3在LFW上的face verification準確率爲99.53％，性能上並沒有比DeepID2+的99.47％提升多少。而且LFW數據集裏面有三對人臉被錯誤地標記了，在更正這些錯誤的label後，兩者準確率均爲99.52％。

---
### FaceNet
**Paper:** [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)<br>
**Code:** [davidsandberg/facenet](https://github.com/davidsandberg/facenet)<br>
&emsp;&emsp;&emsp;[sainimohit23/FaceNet-Real-Time-face-recognition](https://github.com/sainimohit23/FaceNet-Real-Time-face-recognition)<br>
&emsp;&emsp;&emsp;[timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)<br>

**Blog:** [人臉辨識模型 Google Facenet 介紹與使用](https://makerpro.cc/2018/12/introduction-to-face-recognition-model-google-facenet/)
![](https://makerpro.cc/wp-content/uploads/2018/12/6012_ucz-a_bkxq.jpeg)
![](https://makerpro.cc/wp-content/uploads/2018/12/6012_0kptwoxxog.png)
使用不同的 network model 的辨識成績差異
![](https://makerpro.cc/wp-content/uploads/2018/12/6012_bw8bj0ns3a.png)
MTCCN Face Tracking
![](https://github.com/timesler/facenet-pytorch/raw/master/examples/tracked.gif)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

