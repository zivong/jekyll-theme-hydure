---
layout: post
title: Face Recognition
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Face Datasets, Face Detection, Face Alignment, Face Landmark, Face Recognition and exercises.

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

---
### ShuffleNet V2
**Paper:** [https://arxiv.org/abs/1807.11164](https://arxiv.org/abs/1807.11164)<br>
**Code:** [https://github.com/ZhuYun97/ShuffleNet-v2-Pytorch](https://github.com/ZhuYun97/ShuffleNet-v2-Pytorch)<br>
[shufflenetv2.py](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/shufflenetv2.py)<br>

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-21_at_10.52.49_PM_181N53D.png)

---
### SqueezeFaceNet

### VGFaceNet

### GhostFaceNets

---
### MixFaceNets
**Paper:** [MixFaceNets: Extremely Efficient Face Recognition Networks](https://arxiv.org/abs/2107.13046)<br>
**Code:** [https://github.com/fdbtrs/mixfacenets](https://github.com/fdbtrs/mixfacenets)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/MixFaceNet-S-architecture.png?raw=true)

---
### EfficientFace
**Paper:** [EfficientFace: An Efficient Deep Network with Feature Enhancement for Accurate Face Detection](https://arxiv.org/abs/2302.11816)<br>
**Code:** [https://github.com/zengqunzhao/EfficientFace](https://github.com/zengqunzhao/EfficientFace)<br>
![](https://www.researchgate.net/publication/372367822/figure/fig1/AS:11431281188974854@1694829456888/The-complete-architecture-of-the-proposed-EfficientFace-includes-the-feature-extraction.png)

---
### EFaR 2023: Efficient Face Recognition Competition
**Paper:** [https://arxiv.org/abs/2308.04168](https://arxiv.org/abs/2308.04168)<br>

---
## Exercises

### Facial Emotion Recognition (表情辨識)
Using MTCNN to detect face, then use CNN model to recognize emotion<br>
Step 0. sync up Tensorflow version between Kaggle and PC<br>
Step 1. Train a CNN model for facial expression recognition (FER2013 dataset)<br>
Step 2. load CNN model to combine with MTCNN face detection<br>

#### Step 0. To sync up Tensorflow version between Kaggle and PC

<u>On Kaggle:</u><br> 
`import tensorflow as tf`<br>
`print(tf.__version__)`<br>
2.6.1

<u>On PC:</u><br>
`$python -c 'import tensorflow as tf; print(tf.__version__)' `<br>
2.6.1

*Make sure to use same version of Tensorflow on Kaggle and PC for model file compatibilty !*

#### Step 1. Train Ｍodel on Kaggle
![](https://static-01.hindawi.com/articles/cin/volume-2018/7208794/figures/7208794.fig.003.svgz)
**Kaggle:** [rkuo2000/fer2013-cnn](https://kaggle.com/rkuo2000/fer2013-cnn)<br>
**Dataset:** [FER-2013](https://www.kaggle.com/msambare/fer2013)<br>
1. [Edit & Copy] FER2013-CNN 
2. [Run All] to Train model
3. Download model file `fer2013_cnn.h5`

#### Step 2. Run Model on PC
1. Clone sample codes<br>
`$git clone https://github.com/rkuo2000/tf`<br>
2. Copy **fer2013_cnn.h5** to ~/tf/models<br>
3. To detect using image file<br>
`$python mtcnn_emotion_detection.py images/facemask1.jpg`<br>
4. To detect using webcam<br>
`$python mtcnn_emotion_detection_cam.py`<br>

[mtcnn_emotion_detection_cam.py](https://github.com/rkuo2000/tf/blob/master/mtcnn_emotion_detection_cam.py)

---
### [Homework]: Facial Emotion Recognition
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/facial_emotion_detection.png?raw=true)

---
### Face Mask Detection (口罩偵測)
Using MTCNN to detect face, then use CNN model to detect facemask<br>

Step 0. sync up Tensorflow version between Kaggle and PC<br>
Step 1. Train a CNN model for facemask detection<br>
Step 2. load CNN model to combine with MTCNN face detection<br>

#### Step 0. To sync up Tensorflow version between Kaggle and PC

<u>On Kaggle:</u><br> 
`import tensorflow as tf`<br>
`print(tf.__version__)`<br>
2.6.1

<u>On PC:</u><br>
`$python -c 'import tensorflow as tf; print(tf.__version__)' `<br>
2.6.1

*Make sure to use same version of Tensorflow on Kaggle and PC for model file compatibilty !*

#### Step 1. Train Ｍodel on Kaggle
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/facemask_detection_12k_images_dataset.png?raw=true)
**Kaggle:** [rkuo2000/faskmask-cnn](https://kaggle.com/rkuo2000/facemask-cnn)<br>
**Dataset:** [Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset<br>
1. [Edit & Copy] FaceMask-CNN 
2. [Run All] to Train model
3. Download model file `facemask_cnn.h5`

#### Step 2. Run Model on PC
1. Clone sample codes<br>
`$git clone https://github.com/rkuo2000/tf`<br>
2. Copy **facemask_cnn.h5** to ~/tf/models<br>
3. To detect using image file<br>
`$python mtcnn_facemask_detection.py images/facemask1.jpg`<br>
4. To detect using webcam<br>
`$python mtcnn_facemask_detection_cam.py`<br>

[mtcnn_facemask_detection_cam.py](https://github.com/rkuo2000/tf/blob/master/mtcnn_facemask_detection_cam.py)

---
### [Homework]: Face Mask Detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/face_mask_detection.png?raw=true)

---
## Face Identification (人臉辨識)
### FaceNet-PyTorch
**Kaggle:** [FaceNet-PyTorch](https://kaggle.com/rkuo2000/FaceNet-PyTorch)<br>

**[test_images](https://github.com/timesler/facenet-pytorch/tree/master/data/test_images)**<br>
<table>
<tr>
<td><img src="https://github.com/timesler/facenet-pytorch/blob/master/data/test_images/angelina_jolie/1.jpg?raw=true"></td>
<td><img src="https://github.com/timesler/facenet-pytorch/blob/master/data/test_images/bradley_cooper/1.jpg?raw=true"></td>
<td><img src="https://github.com/timesler/facenet-pytorch/blob/master/data/test_images/kate_siegel/1.jpg?raw=true"></td>
<td><img src="https://github.com/timesler/facenet-pytorch/blob/master/data/test_images/paul_rudd/1.jpg?raw=true"></td>
<td><img src="https://github.com/timesler/facenet-pytorch/blob/master/data/test_images/shea_whigham/1.jpg?raw=true"></td>
</tr>
</table>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/FaceNet_PyTorch_embeddings.png?raw=true)
```
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in test_embeddings]
for dist in dists:
    if any(e<1 for e in dist):
        print(names[np.argmin(dist)])
    else:
        print('unknown')
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/FaceNet_PyTorch_result.png?raw=true)

---
### [Homework]: Face Identification
**Github:** [facenet-pytorch](https://github.com/rkuo2000/facenet-pytorch)<br>
**Kaggle:** [FaceNet-PyTorch](https://kaggle.com/rkuo2000/FaceNet-PyTorch)<br>

Steps:<br>
1. **[Fork]** https://github.com/rkuo2000/facenet-pytorch
2. **[Edit&Copy]** https://kaggle.com/rkuo2000/facenet-pytorch<br>
* face detection : models/mtcnn.py<br>
* facenet model : models/inception_resnet_v1.py<br>
* example: examples/infer.ipynb<br>
3. upload 1.jpg for each name to **https://github.com/your_name/facenet-pytoch**  data/test_images
4. upload yourself photo 1.jpg to **https://github.com/your_name/facenet-pytoch** data/test_images/your_name
5. Open **https://kaggle.com/your_name/facenet-pytorch**
6. modify git clone path<br>
`!git clone https://github.com/your_name/facenet-pytorch`<br>
7. modify test image path<br>
`img = plt.imread("facenet_pytorch/data/test_images/your_name/1.jpg")`<br>
8. **[Run-All]**

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

