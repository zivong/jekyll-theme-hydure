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
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/bodypix_result.png?raw=true)

### Pose Estimation
**Kaggle:**[rkuo2000/pose-estimate](https://www.kaggle.com/rkuo2000/pose-estimate) 

---
## Head Pose
### Head Pose Estimation
**Kaggle:** [rkuo2000/head-pose-estimation](https://kaggle.com/rkuo2000/head-pose-estimation)<br />
Download Leonardo_out.mp4 to watch the output video!
<table>
  <tr>
    <td><img src="https://github.com/rkuo2000/head-pose-estimation/blob/master/doc/demo.gif?raw=true"></td>
    <td><img src="https://github.com/rkuo2000/head-pose-estimation/blob/master/doc/demo1.gif?raw=true"></td>
  </tr>
</table>

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
### 3DHand
N/A

---
## 6D Object Pose Estimation
N/A

---


*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

