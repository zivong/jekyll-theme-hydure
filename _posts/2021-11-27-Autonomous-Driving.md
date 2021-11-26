---
layout: post
title: Autonomous Driving
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Autonomous Driving includes Survey, LiDAR-based, Camera-based Object Detection Methods, Monocular Camera-baseed Depth Estimation Methods, 3D Object Detection Methods with LiDAR and Camera, Pedestrain Behavior Prediction Methods, End-to-End Learning, and AirSim Car.

--- 
## Datasets

### [KITTI](http://www.cvlibs.net/datasets/kitti/)
![](http://www.cvlibs.net/datasets/kitti/images/passat_sensors.jpg)
<iframe width="920" height="520" src="https://www.youtube.com/embed/KXpZ6B1YB_k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## A Survey of State-of-Art Technologies
**Paper:** [Autonomous Driving with Deep Learning: A Survey of State-of-Art Technologies](https://arxiv.org/abs/2006.06091)
**Sysyem Diagram:** HW and SW of the autonomous driving platform
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Autonomous_Driving_platform.png?raw=true"></p>

---
## LiDAR-based 3D Object Detection Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/LiDAR_3D_Object_Detection_methods.png?raw=true"></p>

### YOLO3D
**Paper:** [YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud](https://arxiv.org/abs/1808.02350)<br />
**Code:** [maudzung/YOLO3D-YOLOv4-PyTorch](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch)<br />

![](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch/blob/master/docs/demo.gif?raw=true)
* **Inputs:** Bird-eye-view (BEV) maps that are encoded by **height, intensity and density** of 3D LiDAR point clouds.
* **The input size:** 608 x 608 x 3
* **Outputs:** 7 degrees of freedom (**7-DOF**) of objects: (`cx, cy, cz, l, w, h, θ`)

**YOLOv4 architecture**<br />
<img  width="70%" height="70%" src="https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch/blob/master/docs/yolov4_architecture.png?raw=true">

---
### YOLO4D
**Paper:** [YOLO4D: A Spatio-temporal Approach for Real-time Multi-object Detection and Classification from LiDAR Point Clouds](https://openreview.net/pdf?id=B1xWZic29m)

![](https://d3i71xaburhd42.cloudfront.net/ad16b7bf2ee7c84823f69a3bee6a30aa07311073/4-Figure1-1.png)

---
## Camera-based 3D Object Detection Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Camera_based_3D_Object_Detection_methods.png?raw=true"></p>

### Pseudo-LiDAR
**Paper:** [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1812.07179)<br />

![](https://mileyan.github.io/pseudo_lidar/cvpr2018-pipeline.png)

<iframe width="555" height="312" src="https://www.youtube.com/embed/mNtXTTo6wzI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---
### Stereo R-CNN based 3D Object Detection
**Paper:** [Stereo R-CNN based 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1902.09738)<br />
**Code:** [HKUST-Aerial-Robotics/Stereo-RCNN](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)<br />

![](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/blob/master/doc/system.png?raw=true)

---
## Monocular Camera-based Depth Estimation Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Monocular_Camera_based_Depth_Estimation_methods.png?raw=true"></p>

### DF-Net
**Paper:** [DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency](https://arxiv.org/abs/1809.01649)<br />
**Code:** [vt-vl-lab/DF-Net](https://github.com/vt-vl-lab/DF-Net)<br />
* Model in the paper uses 2-frame as input, while this code uses 5-frame as input (you might use any odd numbers of frames as input, though you would need to tune the hyper-parameters)
* FlowNet in the paper is pre-trained on SYNTHIA, while this one is pre-trained on Cityscapes

![](https://d3i71xaburhd42.cloudfront.net/c01876292b5d1ce6e746fd2e2053453847905bb2/5-Figure3-1.png)

<iframe width="860" height="360" src="https://filebox.ece.vt.edu/~ylzou/eccv2018dfnet/short_teaser.mp4"></iframe>

---
## Depth Fusion Methods with LiDAR and Camera
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Depth_Fusion_methods_with_LiDAR_and_Camera.png?raw=true"></p>

### DFineNet
**Paper:** [DFineNet: Ego-Motion Estimation and Depth Refinement from Sparse, Noisy Depth Input with RGB Guidance](https://arxiv.org/abs/1903.06397)<br />
**Code:** [vt-vl-lab/DF-Net](https://github.com/vt-vl-lab/DF-Net)<br />

<img width="70%" height="70%" src="https://d3i71xaburhd42.cloudfront.net/8e1c2ef0816598866e110362583c1c4f570401d3/3-Figure2-1.png">
<img src="https://d3i71xaburhd42.cloudfront.net/8e1c2ef0816598866e110362583c1c4f570401d3/4-Figure3-1.png">
<img src="https://www.researchgate.net/profile/Ty-Nguyen-5/publication/331840392/figure/fig4/AS:737828568834049@1552923447261/Qualitative-results-of-our-method-left-RGB-guide-certainty-34-middle-ranking-1st.ppm">

---
## 3D Object Detection Methods with LiDAR and Camera
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/3D_Object_Detection_methods with LiDAR_and_Camera.png?raw=true"></p>

### PointFusion
**Paper:** [PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation](https://arxiv.org/abs/1711.10871)<br />
**Code:** [mialbro/PointFusion](https://github.com/mialbro/PointFusion)<br />

<img src="https://media.arxiv-vanity.com/render-output/5472384/x1.png">
<img width="70%" height="70%" src="https://media.arxiv-vanity.com/render-output/5472384/sun-qualitative-3x4.png">

---
### MVX-Net
**Paper:** [MVX-Net: Multimodal VoxelNet for 3D Object Detection](https://arxiv.org/abs/1904.01649)<br />
**Code:** [mmdetection3d/mvxnet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/mvxnet)<br />

![](https://www.researchgate.net/profile/Vishwanath-Sindagi/publication/332186784/figure/fig2/AS:743809176567810@1554349335734/Overview-of-the-proposed-MVX-Net-PointFusion-method-The-method-uses-convolutional.ppm)
![](https://www.researchgate.net/profile/Vishwanath-Sindagi/publication/332186784/figure/fig4/AS:743809180778497@1554349336248/Sample-3D-detection-results-from-KITTI-validation-dataset-projected-onto-image-for.jpg)

---
### DeepVO
**Paper:** [DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429)<br />
**Code:** [ChiWeiHsiao/DeepVO-pytorch](https://github.com/ChiWeiHsiao/DeepVO-pytorch)<br />

![](https://www.researchgate.net/profile/Khaled-Alyousefi/publication/341427132/figure/fig4/AS:891775065534466@1589627152759/DeepVO-Deep-Learning-Architecture-8-In-the-proposed-network-RCNN-takes-a-sequence.jpg)
![](https://camo.githubusercontent.com/775dc6f938052fa29586fed4fde4a477280eda67514af884f7687fbea3e163d4/68747470733a2f2f696d6775722e636f6d2f766f307658676b2e706e67)
<iframe width="832" height="468" src="https://www.youtube.com/embed/M4v_-XyYKHY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Pedestrain Behavior Prediction Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Pedestrain_Behavior_Prediction_methods.png?raw=True"></p>

### Predicting Future Person Activities and Locations in Videos
**Paper:** [Peeking into the Future: Predicting Future Person Activities and Locations in Videos](https://arxiv.org/abs/1902.03748)<br />

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Peaking_Into_The_Future_overview.png?raw=true)

<table>
  <tr>
  <td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Peaking_Into_The_Future_PIM.png?raw=true"></td>  
  <td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Peaking_Into_The_Future_PBM.png?raw=true"></td>
  </tr>
</table>

### Spectral Trajectory and Behavior Prediction
**Papers:** [Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clustering in Graph-LSTMs](https://arxiv.org/abs/1912.01118)<br />
**Code:** [Xiejc97/Spectral-Trajectory-and-Behavior-Prediction](https://github.com/Xiejc97/Spectral-Trajectory-and-Behavior-Prediction)<br />
<img width="70%" height="70%" src="https://github.com/Xiejc97/Spectral-Trajectory-and-Behavior-Prediction/raw/master/figures/predict.png">
<table>
  <tr>
  <td><img src="https://github.com/Xiejc97/Spectral-Trajectory-and-Behavior-Prediction/raw/master/figures/results.gif"></td>
  <td><img src="https://github.com/Xiejc97/Spectral-Trajectory-and-Behavior-Prediction/raw/master/figures/behavior.gif"></td>
  </tr>
</table>

## Vehicle Behavior Prediction Methods
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Vehicle_Behvior_Modeling_and_Decision_Making.png?raw=true)

**[Autonomous Vehicle Papers](https://github.com/DeepTecher/AutonomousVehiclePaper)**

---
### End-to-End Deep Learning for Self-Driving Cars
**Blog:** [End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)<br />

![](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/data-collection-system-624x411.png)
![](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/training-624x291.png)
<img width="512" height="512" src="https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png">
<iframe width="740" height="416" src="https://www.youtube.com/embed/NJU9ULQUwng" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### End-to-End Learning of Driving Models
**Paper:** [End-to-End Learning of Driving Models with Surround-View Cameras and Route Planners](https://arxiv.org/abs/1803.10158)<br />

![](https://d3i71xaburhd42.cloudfront.net/951dede2758451825dceed238d356a3a984c3670/2-Figure1-1.png)

---
### End-to-end Learning in Simulated Urban Environments
**Paper:** [Autonomous Vehicle Control: End-to-end Learning in Simulated Urban Environments](https://arxiv.org/abs/1905.06712)<br />

![](https://d3i71xaburhd42.cloudfront.net/c567e98e6d7c9d6f74223315e729708553b38103/6-Figure1-1.png)

---
### NeuroTrajectory
**Paper:** [NeuroTrajectory: A Neuroevolutionary Approach to Local State Trajectory Learning for Autonomous Vehicles](https://arxiv.org/abs/1906.10971)<br />

![](https://www.researchgate.net/profile/Sorin-Grigorescu/publication/334155766/figure/fig2/AS:837409541455872@1576665401121/Local-state-trajectory-estimation-for-autonomous-driving-Given-the-current-position-of.ppm)
![](https://www.researchgate.net/profile/Sorin-Grigorescu/publication/334155766/figure/fig3/AS:837409541459968@1576665401194/Examples-of-synthetic-a-GridSim-and-b-real-world-occupancy-grids-The-top-images-in.ppm)
![](https://www.researchgate.net/profile/Sorin-Grigorescu/publication/334155766/figure/fig4/AS:837409541468160@1576665401291/Deep-neural-network-architecture-for-estimating-local-driving-trajectories-The-training.ppm)

---
### ChauffeurNet [(Waymo)](https://sites.google.com/view/waymo-learn-to-drive/)
**Paper:** [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079)<br />
**Code:** [aidriver/ChauffeurNet](https://github.com/aidriver/ChauffeurNet)<br />

![](https://www.researchgate.net/profile/Mayank-Bansal-4/publication/329525538/figure/fig1/AS:702131904450561@1544412699075/Training-the-driving-model-a-The-core-ChauffeurNet-model-with-a-FeatureNet-and-an.png)

<table>
  <tr>
  <td><p>With Stop Signs Rendered</p><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Waymo_WithStopSigns.gif?raw=true"></td>
  <td><p>No Stop Signs Rendered</p><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Waymo_NoStopSigns.gif?raw=true"></td>
  </tr>
  <tr>
  <td><p>With Perception Boxes Rendered</p><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Waymo_WithPerceptionBoxes.gif?raw=true"></td>
  <td><p>No Perception Boxes Rendered</p><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Waymo_NoPerceptionBoxes.gif?raw=true"></td>  
  </tr>
</table>

**Github:** [Iftimie/ChauffeurNet](https://github.com/Iftimie/ChauffeurNet)<br />

![](https://github.com/Iftimie/ChauffeurNet/blob/master/assets/carla-sim.gif?raw=true)

---
## AirSim
**[Github](https://github.com/microsoft/AirSim)**  
**[Document](https://microsoft.github.io/AirSim/)**

Drones in AirSim
<iframe width="709" height="399" src="https://www.youtube.com/embed/-WfTr1-OBGQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Cars in AirSim
<iframe width="683" height="399" src="https://www.youtube.com/embed/gnz1X3UNM5Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Install AirSim
**Download [Binaries](https://github.com/Microsoft/AirSim/releases) and Run**<br />

For **Windows**, the following environments are available:
1. AbandonedPark
2. Africa (uneven terrain and animated animals)
3. AirSimNH (small urban neighborhood block)
4. Blocks
5. Building_99
6. CityEnviron
7. Coastline
8. LandscapeMountains
9. MSBuild2018 (soccer field)
10. TrapCamera
11. ZhangJiajie

For **Linux**, the following environments are available:
1. AbandonedPark
2. Africa (uneven terrain and animated animals)
3. AirSimNH (small urban neighborhood block)
4. Blocks
5. Building_99
6. LandscapeMountains
7. MSBuild2018 (soccer field)
8. TrapCamera
9. ZhangJiajie

* **Download & Unzip** a environment (AirSimNH.zip)<br />
* Find the exe file, **click to run**<br />
> Windows : *AirSimNH/WindowsNoEditor/AirSimNH/Binaries/Win64/AirSimNH.exe* <br />
> Linux   : *AirSimNH/LinuxNoEditor/AirSimNH/Binaries/Linux/AirSimNH* <br />

<table>
  <tr>
  <td><iframe width="456" height="260" src="https://www.youtube.com/embed/G5ersvlGZMw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
  <td><iframe width="456" height="260" src="https://www.youtube.com/embed/GtrQzdlFMFs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
  </tr>
</table>

### User Interface
<img width="50%" height="50%" src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirSimNH_help.png?raw=true">
* press **F1** for hot-keys help
* press **Arrow-keys** or **A-S-W-D** to drive
* press **Backspace** to rest & restart
* press **Alt-F4** to quit AirSim
* press **0** for sub-windows (shown below)
* Press **REC** button for recording Car/Drone info

For Car, 
```
VehicleName TimeStamp   POS_X   POS_Y   POS_Z   Q_W Q_X Q_Y Q_Z Throttle    Steering    Brake   Gear    Handbrake   RPM Speed   ImageFile
```
For Drone,
```
VehicleName TimeStamp   POS_X   POS_Y   POS_Z   Q_W Q_X Q_Y Q_Z ImageFile
```

---
### Manual Drive 
[remote_control](https://microsoft.github.io/AirSim/remote_control/)
<table>
  <tr>
  <td><img src="https://microsoft.github.io/AirSim/images/AirSimDroneManual.gif"></td>
  <td><img src="https://microsoft.github.io/AirSim/images/AirSimCarManual.gif"></td>  
  </tr>
</table>

### [Reinforcement Learning in AirSim](https://microsoft.github.io/AirSim/reinforcement_learning/)
[RL with Car](https://github.com/Microsoft/AirSim/tree/master/PythonClient/reinforcement_learning)
Note that the simulation needs to be up and running before you execute dqn_car.py

1. Keep AirSim Env running first<br />
`cd AirSimNH/LinuxNoEditor` <br />
`./AirSimNH.sh -ResX=640 -ResY=480 -windowed` (use AirSimNH.bat for Windows) <br />

2. To train DQN model for 500,000 eposides <br />
`pip install gym`<br />
`pip install stable-baselines3`<br />
`git clone https://github.com/microsoft/AirSim` <br />
`cd AirSim/PythonClient/reinforcement_learning` <br />
`python dqn_car.py` <br />

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirSimNH_windowed_DQN_Car.png?raw=true)
<br />

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

