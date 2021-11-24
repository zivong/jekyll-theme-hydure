---
layout: post
title: Pose Estimation Exercises
author: [Richard Kuo]
category: [Example]
tags: [jekyll, ai]
---

Exercises includes BodyPix, Pose Estimate, Head Pose Estimation & VTuber, 3D Hand Pose Estimation, 6D Object Pose Estimation.

---
## Body Pose Estimation
### BodyPix
**Kaggle:** [rkuo2000/BodyPix](https://kaggle.com/rkuo2000/BodyPix)<br />
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/BodyPix_results.png?raw=true)

### PoseNet
**Kaggle:** [rkuo2000/posenet-pytorch](https://www.kaggle.com/rkuo2000/posenet-pytorch)<br />
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PoseNet_keypoints.png?raw=true"></p>

### MMPose
**Kaggle:**[rkuo2000/MMPose](https://www.kaggle.com/rkuo2000/mmpose) <br />
#### 2D Human Pose
![](https://github.com/open-mmlab/mmpose/blob/master/demo/resources/demo_coco.gif?raw=true)
#### 2D Human Whole-Body
![](https://user-images.githubusercontent.com/9464825/95552839-00a61080-0a40-11eb-818c-b8dad7307217.gif)
#### 2D Hand Pose
![](https://user-images.githubusercontent.com/11788150/109098558-8c54db00-775c-11eb-8966-85df96b23dc5.gif)
#### 2D Face Keypoints
![](https://user-images.githubusercontent.com/11788150/109144943-ccd44900-779c-11eb-9e9d-8682e7629654.gif)
#### 3D Human Pose
![](https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif)
#### 2D Pose Tracking
![](https://user-images.githubusercontent.com/11788150/109099201-a93dde00-775d-11eb-9624-f9676fc0e478.gif)
#### 2D Animal Pose
![](https://user-images.githubusercontent.com/11788150/114201893-4446ec00-9989-11eb-808b-5718c47c7b23.gif)
#### 3D Hand Pose
![](https://user-images.githubusercontent.com/28900607/121288285-b8fcbf00-c915-11eb-98e4-ba846de12987.gif)
#### WebCam Effect
![](https://user-images.githubusercontent.com/15977946/124059525-ce20c580-da5d-11eb-8e4a-2d96cd31fe9f.gif)

---
### Pose Recognition

---
## Head Pose
### Head Pose Estimation
**Kaggle:** [rkuo2000/head-pose-estimation](https://kaggle.com/rkuo2000/head-pose-estimation)<br />
<iframe width="652" height="489" src="https://www.youtube.com/embed/BHwHmCUHRyQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## VTuber
### VTuber-Unity = Head-Pose-Estimation + Face-Alignment + GazeTracking**<br />

<u>Build-up Steps</u>:
1. Create a character: **[VRoid Studio](https://vroid.com/studio)**
2. Synchronize the face: **[VTuber_Unity](https://github.com/kwea123/VTuber_Unity)**
3. Take video: **[OBS Studio](https://obsproject.com/download)**
4. Post-processing:
 - Auto-subtitle: **[Autosub](https://github.com/kwea123/autosub)**
 - Auto-subtitle in live stream: **[Unity_live_caption](https://github.com/kwea123/Unity_live_caption)**
 - Encode the subtitle into video: **[小丸工具箱](https://maruko.appinn.me/)**
5. Upload: YouTube
6. [Optional] Install CUDA & CuDNN to enable GPU acceleration
7. To Run <br />
`$git clone https://github.com/kwea123/VTuber_Unity`
`$python demo.py --debug --cpu`
<p align="center"><img src="https://github.com/kwea123/VTuber_Unity/blob/master/images/debug_gpu.gif?raw=true"></p>

---
### OpenVtuber
<u>Build-up Steps</u>:
* Repro [Github](https://github.com/1996scarlet/OpenVtuber)<br />
`$git clone https://github.com/1996scarlet/OpenVtuber`<br />
`$cd OpenVtuber`<br />
`$pip3 install –r requirements.txt`<br />
* Install node.js for Windows <br />
* run Socket-IO Server <br />
`$cd NodeServer` <br />
`$npm install express socket.io` <br />
`$node. index.js` <br />
* Open a browser at  http://127.0.0.1:6789/kizuna <br />
* PythonClient with Webcam <br />
`$cd ../PythonClient` <br />
`$python3 vtuber_link_start.py` <br />

<p align="center"><img src="https://camo.githubusercontent.com/83ad3e28fa8a9b51d5e30cdf745324b09ac97650aea38742c8e4806f9526bc91/68747470733a2f2f73332e617831782e636f6d2f323032302f31322f31322f72564f33464f2e676966"></p>

---
## Hand Pose
### [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)
![](https://github.com/facebookresearch/InterHand2.6M/blob/main/assets/teaser.gif?raw=true)

1. Download pre-trained InterNet from [here](https://drive.google.com/drive/folders/1BET1f5p2-1OBOz6aNLuPBAVs_9NLz5Jo?usp=sharing)
2. Put the model at `demo` folder
3. Go to `demo` folder and edit `bbox` in [here](https://github.com/facebookresearch/InterHand2.6M/blob/5de679e614151ccfd140f0f20cc08a5f94d4b147/demo/demo.py#L74)
4. run `python demo.py --gpu 0 --test_epoch 20`
5. You can see `result_2D.jpg` and 3D viewer.

**Camera positios visualization demo**
1. `cd tool/camera_visualize`
2. Run `python camera_visualize.py`

---
## 6D Object Pose Estimation
N/A

---


*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

