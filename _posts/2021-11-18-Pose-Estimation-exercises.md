---
layout: post
title: Pose Estimation Exercises
author: [Richard Kuo]
category: [Example]
tags: [jekyll, ai]
---

*Exercises includes BodyPix, Pose Estimate, Head Pose Estimation & VTuber, 3D Hand Pose Estimation, 6D Object Pose Estimation.*

---
## Body Pose Estimation
### BodyPix
**Kaggle:** [rkuo2000/BodyPix](https://kaggle.com/rkuo2000/BodyPix)

![](https://www.kaggleusercontent.com/kf/80090018/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..wiq28Ozrir-2AJ-ZtEn5iw.cSVqA8fZWrY56p72rcdToeQ6nHPmFGjaif7KJHfO5JhXm0YRfQmnON1e05BS5H39vy4S3ojtLmzWLbh4VUoonbDxP7F5H1REZU-ktGZF43Hdvhr6cz8quxq7VfHSlRviDM7U1DEkTzzVQ1wCJrILmdaNGnkFSXC89CURxueXsaugslat6CAlFBzTAuekPFP2TNSn3KUnKFNIoMQlSUC2aZJRL0IFlXHmJllxE5IvLzGKMkWftQfKiuUOINXDNLxexeWDOlBfPv8jYjq_uJhcXzlzqtinX36lYJEj-A8wF39_qlCqjfT6AdiSDABj78DgktHBV6owq8T4ANjqnKzlq2lwgrmYntoNYUjlMGigA5kewr9uEQmP4fohcOV4IsLtEu31nAkz2-22bcTxJHHKr0CJOs7FxM7_OsLffwEJnOMUZiGCI9EPi3ii0aTuV0ALleinIcwUMN8Bodt9Emc6xfXGgfX1WZPsILP2rfXQSh2GmR7mYFMc0c4cyYze8UaxmGVRJa9RNrrUW3SFW4wjXkFge9fItDkSCr3Lma0Ara88Cy4vO-McrdrxAVC9nmKsIwPB_XVgTxaQvBSXSWvl_watbZ0JYS-4oKifNcHwBuTsM5aOCbZ85iU5wnNmnqvQhpU8e_E2tdIBuqM3eIO3YA.O4WqqgTn6X-RD6GusHkayg/__results___files/__results___33_0.png)


### Pose Estimate
**Kaggle:**[rkuo2000/pose-estimate](https://www.kaggle.com/rkuo2000/pose-estimate) 

![](https://www.kaggleusercontent.com/kf/43818384/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..eSJ3kyzfX4P4BksVr8pB-Q.7odouGOvmpfl9QzJYN0y3LKS2OaSZD8CsYh9xIZRQIRUxMn4-eIkiStD65V4cnY3xp82lnn96ZD2A-kyW4Vs-s9Q38zgS0Pzq1DOtqiHaPVKTT7WRXZEFPf-noSXBFkTSt6zknezJm4rhdLtFkrQh_9Eml3RkADFCtZdwYt3QfW4yWFgD-2x0bcMjlOf8A5EkQfnnwTxFmf0YAZkaUBbyt6SX7p8dqegk-61ICDT1IJ9Hj51WE6MZu9MKq5S9Hxo5xtLdbDSABAvcV0zr3UwFUJ5F6wF8SsxFHGHbAPkiga7KjI6gKN-uoixUMx2Yc63YmwOzOnIjKfh2FtoIqsqqzkfHBO25FZD-enKuebG1Lm6BUnK2eUvHARe9l1f1BGY2e-mjri0YYaWQq_jm6xEbtRq9sg-AL-UCZuJzV0K1AXmjycegpN1G8Zg4gIlHIx0HGahmg2fj1wVI0D2DtjBKGUpgleinZwMT2ovnMQf9TSn5KHlZ6FjrM6MmK2wEGk7MYZH_4Ba3gfAbE5NQH9w6Bs0B_UN7xBRfrTNAG0wqiOeWsjUWlXJNE4gt7iZGOnRf6w6RICGPX4p2dzOI7BSnCs2okCSywl-2Yrk-s-XwsqYhcv_4d7XmrrIjaAy-5dMv08Iy88U0PMUMu3K-rQ0KA.XlaB6O5MSaMP6ib8Cp8ZCg/__results___files/__results___10_1.png)

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

