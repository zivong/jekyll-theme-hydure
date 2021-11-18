---
layout: post
title: Pose Estimation Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Pose Estimation includes Motion Datasets, Body Pose , Hand Pose , Object Pose.

---
## Datasets

### Moving MNIST
[Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy)[782Mb] contains 10,000 sequences each of length 20 showing 2 digits moving in a 64 x 64 frame.
<table>
  <tr>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000000.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000001.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000002.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000003.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000004.gif"></td>
  </tr>
  <tr>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000005.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000006.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000007.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000008.gif"></td>
  <td><img src="http://www.cs.toronto.edu/~nitish/unsupervised_video/images/000009.gif"></td>
  </tr>
</table>

### UCF-101 : Action Recognition Data Set 
[UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://www.crcv.ucf.edu/papers/UCF101_CRCV-TR-12-01.pdf)<br />
![](https://www.crcv.ucf.edu/data/UCF101/UCF101.jpg)
The 5 action categories :1) Human-Object Interaction 2) Body-Motion Only 3) Human-Human Interaction 4) Playing Musical Instruments 5) Sports
Train : 13320 trimmed videos
Background Data: 2980 untrimmed videos
Validation : 2104 untrimmed videos
Test : 5613 untrimmed videos

### HMDB-51 : A large human motion database
https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
**HMDB** - about 2GB for a total of 7,000 clips distributed in 51 action classes

---
## Body Pose
**Blog:** [A 2019 Guide to Huamn Pose Estimatioin](https://heartbeat.comet.ml/a-2019-guide-to-human-pose-estimation-c10b79b64b73#7c7f)<br />

### BodyPix - Person Segmentation in the Browser
**Code:** [tfjs-models/body-pix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)<br />
`pip install tf_bodypix`
[Live Demo](https://storage.googleapis.com/tfjs-models/demos/body-pix/index.html)

![](https://github.com/tensorflow/tfjs-models/raw/master/body-pix/images/body-pix-2.0.gif)<br />

### OpenPose
> **Paper:** [arxiv.org/abs/1812.08008](https://arxiv.org/abs/1812.08008)<br />
> **Code:** [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)<br />

![](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/.github/media/pose_face_hands.gif?raw=true)

### Pose Recognition
> **Code:** [burningion/dab-and-tpose-controlled-lights](https://github.com/burningion/dab-and-tpose-controlled-lights)
Based on OpenPose
![](https://github.com/burningion/dab-and-tpose-controlled-lights/raw/master/images/dab-tpose.gif)
![](https://raw.githubusercontent.com/burningion/dab-and-tpose-controlled-lights/master/images/neural1.png)

### MMPose
**Code:** [open-mmlab](https://github.com/open-mmlab/mmpose)
<iframe width="920" height="520" src="https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4" title="MMPos Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
**[Model Zoo](https://github.com/open-mmlab/mmpose#model-zoo)** <br />

### DensePose RCNN
> **Paper:** [arxiv.org/abs/1802.00434](https://arxiv.org/abs/1802.00434)<br />
> **Code:** [facebookresearch/DensePose](https://github.com/facebookresearch/Densepose)<br />

![](https://camo.githubusercontent.com/225186f89a5aca46035b68e09fded46de693fb40d5abc9663c7b9cc0bc42c25f/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d316b3448746f5870624456394d68757968615663784472586e79505f4e58383936)

**Region-based DensePose architecture**<br />
![](https://miro.medium.com/max/700/1*i4GLy3FNl7SSl2j3uI8mVg.png)

**Multi-task cascaded architectures**<br />
![](https://miro.medium.com/max/700/1*FKJRPgm7RUZdnvdqTTMe8g.png)

### Multi-Person Part Segmentation
**Paper:** [arxiv.org/abs/1907.05193](https://arxiv.org/abs/1907.05193)<br />
**Code:** [kevinlin311tw/CDCL-human-part-segmentation](https://github.com/kevinlin311tw/CDCL-human-part-segmentation)

![](https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/cdcl_teaser.jpg?raw=true)

---
## Hand Pose
[Papers](https://codechina.csdn.net/mirrors/xinghaochen/awesome-hand-pose-estimation)

### Hand3D
**Paper:** [arxiv.org/abs/1705.01389](https://arxiv.org/abs/1705.01389)<br />
**Code:** [lmb-freiburg/hand3d](https://github.com/lmb-freiburg/hand3d)<br />

![](https://github.com/lmb-freiburg/hand3d/blob/master/teaser.png?raw=true)

### DeeHPS
**Paper:** [arxiv.org/abs/1808.09208](https://arxiv.org/abs/1808.09208)<br />

![](https://www.researchgate.net/profile/Alexis-Heloir/publication/328310189/figure/fig1/AS:692903412240387@1542212455475/a-An-overview-of-our-method-for-simultaneous-3D-hand-pose-and-surface-estimation-A.ppm)

### GraphPoseGAN
**Paper:** [arxiv.org/abs/1912.01875](https://arxiv.org/abs/1912.01875)<br />

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/9479a278f3c63b57b26eb31e44744a238b11feee/1-Figure1-1.png)

### 3D Hand Shape
**Paper:** [arxiv.org/abs/1903.00812](https://arxiv.org/abs/1903.00812)<br />
**Code:** [https://github.com/3d-hand-shape/hand-graph-cnn](https://github.com/3d-hand-shape/hand-graph-cnn)<br />

---
## Object Pose

### [Benchmark for 6D Object Pose ](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/)
**Core Datasets:**
LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V.
**Other datasets:**
LM, RU-APC, IC-MI, TYO-L

<table>
  <tr>
  <td><img src="https://bop.felk.cvut.cz/media/lm_thumb_gt_wzRGtjs.jpg"></td>
  <td><p>LM (Linemod)</p>
      <p>Hinterstoisser et al.: Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes, ACCV 2012</p><br/>
      <p>15 texture-less household objects with discriminative color, shape and size. Each object is associated with a test image set showing one annotated object instance with significant clutter but only mild occlusion.</p></td>
  </tr>
</table>

### PoseCNN
**Paper:** [arxiv.org/abs/1711.00199](https://arxiv.org/abs/1711.00199)<br />
**Code:** [yuxng/PoseCNN](https://github.com/yuxng/PoseCNN)<br />

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

### DeepIM
**Paper:** [arxiv.org/abs/1804.00175](https://arxiv.org/abs/1804.00175)<br />
**Code:** [liyi14/mx-DeepIM](https://github.com/liyi14/mx-DeepIM)<br />

![](https://github.com/liyi14/mx-DeepIM/blob/master/assets/net_structure.png?raw=tru)

### Single Shot Pose
**Paper:** [arxiv.org/abs/1711.08848](https://arxiv.org/abs/1711.08848)<br />
**Code:** [microsoft/singleshotpose](https://github.com/microsoft/singleshotpose)<br />

![](https://camo.githubusercontent.com/803dd24670ed987bc9477d7bf7b63dd54509da2fe945de35e123a82c90006d6a/68747470733a2f2f6274656b696e2e6769746875622e696f2f73696e676c655f73686f745f706f73652e706e67)

### Segmentation-driven Pose
**Paper:** [arxiv.org/abs/1812.02541](https://arxiv.org/abs/1812.02541)<br />
**Code:** [cvlab-epfl/segmentation-driven-pose](https://github.com/cvlab-epfl/segmentation-driven-pose)<br />

![](https://github.com/cvlab-epfl/segmentation-driven-pose/blob/master/images/fig1.jpg?raw=true)

### DPOD
**Paper:** [arxiv.org/abs/1902.11020](https://arxiv.org/abs/1902.11020)<br />
**Code:** [yashs97/DPOD](https://github.com/yashs97/DPOD)<br />
<table>
  <tr>
  <td><img src="https://github.com/yashs97/DPOD/blob/master/demo_results/demo1.png?raw=true"></td>
  <td><img src="https://github.com/yashs97/DPOD/blob/master/demo_results/demo2.png?raw=true"></td>
</table>
![](https://d3i71xaburhd42.cloudfront.net/efae64551ff0b38fb6ac938727a001a9892be67f/4-Figure2-1.png)


---


*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

