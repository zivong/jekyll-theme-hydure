---
layout: post
title: Pose Estimation Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Pose Estimation includes Applications, Body Pose, Head Pose, Hand Pose , Object Pose.

---
### Pose Estimation Applications
* **[健身鏡](https://johnsonfitnesslive.com/?action=mirror_pro_intro)**<br />
![](https://johnsonfitnesslive.com/images/mirrorPro-parallax-bg2-img03.gif)

* **[AI健身教練](https://fc.bnext.com.tw/articles/view/1226)**<br />
健身新創 Peloton 在這波居家健身浪潮下，以販售主力產品飛輪、跑步機搭配線上課程，並將健身教練打造成「網紅」，用心拍攝運動影片，成功創造粉絲經濟。<br />
![](https://bnextmedia.s3.hicloud.net.tw/image/album/2021-03/img-1614856341-69773@600.jpg)

* **[馬術治療](https://www.inside.com.tw/article/21711-aigo-interview-aifly)**<br />
![](https://inside-assets1.inside.com.tw/2020/11/oz0fu9mal72kdptfhnq8v79sf67c57.png?w=730&fit=max&q=80)

* **[Pose-controlled Lights](https://github.com/burningion/dab-and-tpose-controlled-lights)**<br />
![](https://github.com/burningion/dab-and-tpose-controlled-lights/raw/master/images/dab-tpose.gif?raw=True)
* **[跌倒偵測](https://www.chinatimes.com/realtimenews/20201203005307-260418?chdtv)**
<table>
  <tr>
  <td><img src="https://images.chinatimes.com/newsphoto/2020-12-03/1024/20201203005495.jpg"></td>  
  <td><img src="https://matching.org.tw/website/uploads_product/website_1/P0000100000044_4_123.jpg"></td>
  </tr>
</table>

* **[產線SOP](https://www.inside.com.tw/article/21716-aigo-interview-beseye-alpha)**<br />
以雅文塑膠來說，產線作業員的動作僅集中於上半身，以頭部、頸部、肩膀、手臂、手掌的動作為主。Beseye_alpha 針對需求，複製日本大型製造工廠 AI 模型開發的成功案例、及與客戶多次討論需求、實地作業工作站規劃、實際場域測試資料訓練，開發出一個「肢體律動分析」模型，有效達到降低運算量的目標。

* **[其他應用](https://www.eastwestidea.net/index.php/%E6%9D%90%E6%96%99/item/376)**<br />

---
## Body Pose
**Blog:** [A 2019 Guide to Huamn Pose Estimatioin](https://heartbeat.comet.ml/a-2019-guide-to-human-pose-estimation-c10b79b64b73#7c7f)<br />

### BodyPix - Person Segmentation in the Browser
**Code:** [tfjs-models/body-pix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)<br />
`pip install tf_bodypix`
[Live Demo](https://storage.googleapis.com/tfjs-models/demos/body-pix/index.html)

![](https://github.com/tensorflow/tfjs-models/raw/master/body-pix/images/body-pix-2.0.gif)<br />
---
### OpenPose
> **Paper:** [arxiv.org/abs/1812.08008](https://arxiv.org/abs/1812.08008)<br />
> **Code:** [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)<br />
> **Blog:** [A Guide to OpenPose in 2021](https://viso.ai/deep-learning/openpose/)<br />

![](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/.github/media/pose_face_hands.gif?raw=true)
![](https://viso.ai/wp-content/uploads/2021/01/Keypoints-Detected-by-OpenPose-on-the-COCO-Dataset.jpg)
![](https://media.arxiv-vanity.com/render-output/5509832/x2.png)

---
### PoseNet
<font size="3">
PoseNet is built to run on lightweight devices such as the browser or mobile device where as<br />
OpenPose is much more accurate and meant to be ran on GPU powered systems. You can see the performance benchmarks below.<br />
</font>
>**Paper:**  [arxiv.org/abs/1505.07427](https://arxiv.org/abs/1505.07427)<br />
>**Code:** [rwightman/posenet-pytorch](https://github.com/rwightman/posenet-pytorch)

![](https://debuggercafe.com/wp-content/uploads/2020/10/keypoint_exmp.jpg)
![](https://www.researchgate.net/profile/Soroush-Seifi/publication/335989945/figure/fig2/AS:806499555233793@1569295886946/The-Posenet-architecture-Yellow-modules-are-shared-with-GoogleNet-while-green-modules.ppm)
![](https://i1.wp.com/parleylabs.com/wp-content/uploads/2020/01/image-1.png?resize=1024%2C420&ssl=1)

---
### Pose Recognition 
*using Pose keypoints as dataset to train a DNN*
> **Code:** [burningion/dab-and-tpose-controlled-lights](https://github.com/burningion/dab-and-tpose-controlled-lights)<br />
> **IPYNB:** [pose-control-lights](https://github.com/burningion/dab-and-tpose-controlled-lights/blob/master/Data%20Play.ipynb)<br />

![](https://github.com/burningion/dab-and-tpose-controlled-lights/raw/master/images/dab-tpose.gif)
![](https://raw.githubusercontent.com/burningion/dab-and-tpose-controlled-lights/master/images/neural1.png)

---
### MMPose
**Code:** [open-mmlab](https://github.com/open-mmlab/mmpose)<br />
**[Model Zoo](https://github.com/open-mmlab/mmpose#model-zoo)** <br />
<iframe width="920" height="520" src="https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4" title="MMPos Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### DensePose RCNN
> **Paper:** [arxiv.org/abs/1802.00434](https://arxiv.org/abs/1802.00434)<br />
> **Code:** [facebookresearch/DensePose](https://github.com/facebookresearch/Densepose)<br />

<p align="center"><img src="https://camo.githubusercontent.com/be4698fcd4b3b976b2ec5ebd489e91c1b206c8ef350e46031c05071311d1fe6c/68747470733a2f2f646c2e666261697075626c696366696c65732e636f6d2f64656e7365706f73652f7765622f64656e7365706f73655f7465617365725f636f6d707265737365645f32352e676966"></p>

**Region-based DensePose architecture**<br />
<p align="center"><img src="https://miro.medium.com/max/700/1*i4GLy3FNl7SSl2j3uI8mVg.png"></p>

**Multi-task cascaded architectures**<br />
<p align="center"><img src="https://miro.medium.com/max/700/1*FKJRPgm7RUZdnvdqTTMe8g.png"></p>

---
### Multi-Person Part Segmentation
**Paper:** [arxiv.org/abs/1907.05193](https://arxiv.org/abs/1907.05193)<br />
**Code:** [kevinlin311tw/CDCL-human-part-segmentation](https://github.com/kevinlin311tw/CDCL-human-part-segmentation)

![](https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/cdcl_teaser.jpg?raw=true)

---
## Head Pose
### Head Pose Estimation
**Code:**[yinguobing/head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)<br />
<table>
  <tr>
    <td><img src="https://github.com/rkuo2000/head-pose-estimation/blob/master/doc/demo.gif?raw=true"></td>
    <td><img src="https://github.com/rkuo2000/head-pose-estimation/blob/master/doc/demo1.gif?raw=true"></td>
  </tr>
</table>

## VTuber
[Vtuber總數突破16000人，增速不緩一年增加3000人](https://www.4gamers.com.tw/news/detail/50500/virtual-youtuber-surpasses-16000)
依據日本數據調查分析公司 User Local 的報告，在該社最新的 [User Local VTuber](https://virtual-youtuber.userlocal.jp/document/ranking) 排行榜上，有紀錄的 Vtuber 正式突破了 16,000 人。

1位 [Gawr Gura(がうるぐら サメちゃん)](https://virtual-youtuber.userlocal.jp/user/0DCB37A5BB880687_c8ea33) Gawr Gura Ch. hololive-EN
<iframe width="730" height="411" src="https://www.youtube.com/embed/Hq9BBxGqyCY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

2位 [キズナアイ](https://virtual-youtuber.userlocal.jp/user/D780B63C2DEBA9A2_fa95ae) A.I.Channel
<iframe width="730" height="411" src="https://www.youtube.com/embed/ZedmVzmAf3M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

3位 [Mori Calliope(森カリオペ)](https://virtual-youtuber.userlocal.jp/user/CE32A8A748265090_585651) Mori Calliope Ch.
<iframe width="730" height="411" src="https://www.youtube.com/embed/j-QtsaCscyI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### VTuber-Unity = Head-Pose-Estimation + Face-Alignment + GazeTracking**<br />

### [VRoid Studio](https://vroid.com/studio)
### [VTuber_Unity](https://github.com/kwea123/VTuber_Unity)

<p align="center"><img src="https://github.com/kwea123/VTuber_Unity/blob/master/images/debug_gpu.gif?raw=true"></p>

### [OpenVtuber](https://github.com/1996scarlet/OpenVtuber)

<p align="center"><img src="https://camo.githubusercontent.com/83ad3e28fa8a9b51d5e30cdf745324b09ac97650aea38742c8e4806f9526bc91/68747470733a2f2f73332e617831782e636f6d2f323032302f31322f31322f72564f33464f2e676966"></p>

---
## Hand Pose
[Hand Pose Estimation papers](https://codechina.csdn.net/mirrors/xinghaochen/awesome-hand-pose-estimation)

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

![](https://github.com/3d-hand-shape/hand-graph-cnn/blob/master/teaser.png?raw=true)

---
### FaceBook InterHand2.6M
**Paper:** [InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image
](https://arxiv.org/abs/2008.09309)<br />
**Code:** [facebookresearch/InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)<br />

![](https://github.com/facebookresearch/InterHand2.6M/blob/main/assets/teaser.gif?raw=true)

---
### FrankMocap: Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration
**Paper:** [arxiv.org/abs/2008.08324](https://arxiv.org/abs/2008.08324)<br />
**Code:** [facebookresearch/frankmocap](https://github.com/facebookresearch/frankmocap)<br />

![](https://miro.medium.com/max/1366/1*SISQhpf8pphnob3TjT-hWQ.png)
![](https://github.com/jhugestar/jhugestar.github.io/blob/master/img/frankmocap_wholebody.gif?raw=true)

---
### A Skeleton-Driven Neural Occupancy Representation for Articulated Hands
**Paper:** [arxiv.org/abs/2109.11399](https://arxiv.org/abs/2109.11399)<br />

![](https://d3i71xaburhd42.cloudfront.net/f6bbbb9a507b00bbb003e55d888603ccdf47a762/7-Figure5-1.png)

### Towards unconstrained joint hand-object reconstruction from RGB videos
**Paper:** [arxiv.org/abs/2108.07044](https://arxiv.org/abs/2108.07044)<br />
**Code:** [hassony2/homan](https://github.com/hassony2/homan)<br />
<table>
  <tr>
  <td><img src="https://github.com/hassony2/homan/raw/master/5900_in.gif?raw=True"></td>
  <td><img src="https://github.com/hassony2/homan/raw/master/5900_cam.gif?raw=True"></td>
  <td><img src="https://github.com/hassony2/homan/raw/master/0011.gif?raw=True"></td>
  </tr>
</table>

### Fast Monocular Hand Pose Estimation on Embedded Systems
**Paper:** [arxiv.org/abs/2102.07067](https://arxiv.org/abs/2102.07067)<br />

### Recent Advances in 3D Object and Hand Pose Estimation
**Paper:** [arxiv.org/abs/2006.05927](https://arxiv.org/abs/2006.05927)<br />

---
## Object Pose

### [Benchmark for 6D Object Pose ](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/)
**Core datasets:**<br/>
<table>
  <tr>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/lmo_thumb_gt_tZ8rp3d.jpg"></td>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/tless_lm_thumb_gt_cyvAOhR.jpg"></td>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/tudl_thumb_gt_tjEyxW8.jpg"></td>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/icbin_thumb_gt.jpg"></td>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/itodd_thumb_gt3.jpg"></td>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/hb_thumb_gt.jpg"></td>
  <td><img width="160" height="120" src="https://bop.felk.cvut.cz/media/ycbv_thumb_gt2.jpg"></td>
  </tr>
  <tr>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#LM-O">LM-O</a></td>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#T-LESS">T-LESS</a></td>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#TUD-L">TUD-L</a></td>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#IC-BIN">IC-BIN</a></td>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#ITODD">ITODD</a></td>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#HB">HB</a></td>
  <td align="center"><a href="https://bop.felk.cvut.cz/datasets/#YCB-V">YCB-V</a></td>
  </tr>
</table>
**Other datasets:** 
<a href="https://bop.felk.cvut.cz/datasets/#LM">LM</a>, 
<a href="https://bop.felk.cvut.cz/datasets/#RU-APC">RU-APC</a>,
<a href="https://bop.felk.cvut.cz/datasets/#IC-MI">IC-MI</a>,
<a href="https://bop.felk.cvut.cz/datasets/#TYO-L">TYO-L</a>.

---
### Real-Time Seamless Single Shot 6D Object Pose Prediction (YOLO-6D)
**Paper:** [arxiv.org/abs/1711.08848](https://arxiv.org/abs/1711.08848)<br />
**Code:** [microsoft/singleshotpose](https://github.com/microsoft/singleshotpose)<br />

![](https://camo.githubusercontent.com/803dd24670ed987bc9477d7bf7b63dd54509da2fe945de35e123a82c90006d6a/68747470733a2f2f6274656b696e2e6769746875622e696f2f73696e676c655f73686f745f706f73652e706e67)

---
### PoseCNN
**Paper:** [arxiv.org/abs/1711.00199](https://arxiv.org/abs/1711.00199)<br />
**Code:** [yuxng/PoseCNN](https://github.com/yuxng/PoseCNN)<br />

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

---
### DeepIM
**Paper:** [arxiv.org/abs/1804.00175](https://arxiv.org/abs/1804.00175)<br />
**Code:** [liyi14/mx-DeepIM](https://github.com/liyi14/mx-DeepIM)<br />

![](https://github.com/liyi14/mx-DeepIM/blob/master/assets/net_structure.png?raw=tru)

---

### Segmentation-driven Pose
**Paper:** [arxiv.org/abs/1812.02541](https://arxiv.org/abs/1812.02541)<br />
**Code:** [cvlab-epfl/segmentation-driven-pose](https://github.com/cvlab-epfl/segmentation-driven-pose)<br />

![](https://github.com/cvlab-epfl/segmentation-driven-pose/blob/master/images/fig1.jpg?raw=true)

---
### DPOD
**Paper:** [arxiv.org/abs/1902.11020](https://arxiv.org/abs/1902.11020)<br />
**Code:** [yashs97/DPOD](https://github.com/yashs97/DPOD)<br />
<table>
  <tr>
  <td><img src="https://github.com/yashs97/DPOD/blob/master/demo_results/demo1.png?raw=true"></td>
  <td><img src="https://github.com/yashs97/DPOD/blob/master/demo_results/demo2.png?raw=true"></td>
  </tr>
</table>
![](https://d3i71xaburhd42.cloudfront.net/efae64551ff0b38fb6ac938727a001a9892be67f/4-Figure2-1.png)

---
### HO-3D_v3 Dataset
**Paper:** [arxiv.org/abs/2107.00887](https://arxiv.org/abs/2107.00887)<br />
**Github:** [shreyashampali/ho3d](https://github.com/shreyashampali/ho3d)<br />
HO-3D is a dataset with 3D pose annotations for hand and object under severe occlusions from each other.

![](https://github.com/shreyashampali/ho3d/blob/master/teaser.png?raw=true)


<br />
<br />

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

