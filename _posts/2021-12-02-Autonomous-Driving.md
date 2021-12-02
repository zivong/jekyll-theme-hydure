---
layout: post
title: Autonomous Driving
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Autonomous Driving includes Datasets, Survey, LiDAR-based, Camera-based Object Detection Methods, Monocular Camera-baseed Depth Estimation Methods, 3D Object Detection Methods with LiDAR and Camera, Pedestrain Behavior Prediction Methods, End-to-End Learning, and AirSim Car.

--- 
## Open-Source Autonomous Driving Datasets
1. [A2D2 Dataset for Autonomous Driving](https://www.a2d2.audi/a2d2/en.html)<br>
Released by Audi, the Audi Autonomous Driving Dataset (A2D2) was released to support startups and academic researchers working on autonomous driving. The dataset includes over 41,000 labeled with 38 features. Around 2.3 TB in total, A2D2 is split by annotation type (i.e. semantic segmentation, 3D bounding box). In addition to labelled data, A2D2 provides unlabelled sensor data (~390,000 frames) for sequences with several loops.

2. [ApolloScape Open Dataset for Autonomous Driving](http://apolloscape.auto/)<br>
Part of the Apollo project for autonomous driving, ApolloScape is an evolving research project that aims to foster innovation across all aspects of autonomous driving, from perception to navigation and control. Via their website, users can explore a variety of simulation tools and over 100K street view frames, 80k lidar point cloud and 1000km trajectories for urban traffic.
![](https://images.squarespace-cdn.com/content/v1/5e662d05298fd51947b065be/1623118246092-5GOSW6OQVM3KO1IDVZR6/apolloscape-lanemark-segmentation.gif?format=500w)

3. [Argoverse Dataset](https://www.argoverse.org/)<br>
Argoverse is made up of two datasets designed to support autonomous vehicle machine learning tasks such as 3D tracking and motion forecasting. Collected by a fleet of autonomous vehicles in Pittsburgh and Miami, the dataset includes 3D tracking annotations for 113 scenes and over 324,000 unique vehicle trajectories for motion forecasting. Unlike most other open source autonomous driving datasets, Argoverse is the only modern AV dataset that provides forward-facing stereo imagery.

4. [Berkeley DeepDrive Dataset](https://www.bdd100k.com/)<br>
Also known as BDD 100K, the DeepDrive dataset gives users access to 100,000 annotated videos and 10 tasks to evaluate image recognition algorithms for autonomous driving. The dataset represents more than 1000 hours of driving experience with more than 100 million frames, as well as information on geographic, environmental, and weather diversity.

5. [CityScapes Dataset](https://www.cityscapes-dataset.com/)<br>
CityScapes is a large-scale dataset focused on the semantic understanding of urban street scenes in 50 German cities. It features semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories. The entire dataset  includes 5,000 annotated images with fine annotations, and an additional 20,000 annotated images with coarse annotations.
![](https://images.squarespace-cdn.com/content/v1/5e662d05298fd51947b065be/1623118354736-TECH4VM0QUF5S1MGYNPC/cityscapes-dataset.png?format=1500w)

6. [Comma2k19 Dataset](https://github.com/commaai/comma2k19)<br>
This dataset includes 33 hours of commute time recorded on highway 280 in California. Each 1-minute scene was captured on a 20km section of highway driving between San Jose and San Francisco. The data was collected using comma EONs, which features a road-facing camera, phone GPS, thermometers and a 9-axis IMU. 

7. [Google-Landmarks Dataset](https://ai.googleblog.com/2018/03/google-landmarks-new-dataset-and.html)<br>
Published by Google in 2018, the Landmarks dataset is divided into two sets of images to evaluate recognition and retrieval of human-made and natural landmarks. The original dataset contains over 2 million images depicting 30 thousand unique landmarks from across the world. In 2019, Google published Landmarks-v2, an even larger dataset with 5 million images and 200k landmarks.

8. [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti-360/)<br>
First released in 2012 by Geiger et al, the KITTI dataset was released with the intent of advancing autonomous driving research with a novel set of real-world computer vision benchmarks. One of the first ever autonomous driving datasets, KITTI boasts over 4000 academic citations and counting.<br>
KITTI provides 2D, 3D, and bird’s eye view object detection datasets, 2D object and multi-object tracking and  segmentation datasets, road/lane evaluation detection datasets, both pixel and instance-level semantic datasets, as well as raw datasets.
![](https://images.squarespace-cdn.com/content/v1/5e662d05298fd51947b065be/1623118515861-LTHVVJHBA6RSJFXGJFMT/kitti-vision-benchmark-examples.png?format=1000w)

9. [LeddarTech PixSet Dataset](https://leddartech.com/solutions/leddar-pixset-dataset/)<br>
Launched in 2021, Leddar PixSet is a new, publicly available dataset for autonomous driving research and development that contains data from a full AV sensor suite (cameras, LiDARs, radar, IMU), and includes full-waveform data from the Leddar Pixell, a 3D solid-state flash LiDAR sensor. The dataset contains 29k frames in 97 sequences, with more than 1.3M 3D boxes annotated

10. [Level 5 Open Data](https://level-5.global/data/)<br>
Published by popular rideshare app Lyft, the Level5 dataset is another great source for autonomous driving data. It includes over 55,000 human-labeled 3D annotated frames, surface map, and an underlying HD spatial semantic map that is captured by 7 cameras and up to 3 LiDAR sensors that can be used to contextualize the data.

11. [nuScenes Dataset](https://www.nuscenes.org/)<br>
Developed by Motional, the nuScenes dataset is one of the largest open-source datasets for autonomous driving. Recorded in Boston and Singapore using a full sensor suite (32-beam LiDAR, 6 360° cameras and radars), the dataset contains over 1.44 million camera images capturing a diverse range of traffic situations, driving maneuvers, and unexpected behaviors.
![](https://images.squarespace-cdn.com/content/v1/5e662d05298fd51947b065be/1623118644069-5KCA7MBNLLO4W0SCHE0M/nuscenes-dataset-examples.jpeg?format=1500w)

12. [Oxford Radar RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/)<br>
The Oxford RobotCar Dataset contains over 100 recordings of a consistent route through Oxford, UK, captured over a period of over a year. The dataset captures many different environmental conditions, including weather, traffic and pedestrians, along with longer term changes such as construction and roadworks.

13. [PandaSet](https://pandaset.org/)<br>
PandaSet was the first open-source AV dataset available for both academic and commercial use. It contains 48,000 camera images, 16,000 LiDAR sweeps, 28 annotation classes, and 37 semantic segmentation labels taken from a full sensor suite.

14. [Udacity Self Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car)<br>
Online education platform Udacity has open sourced access to a variety of projects for autonomous driving, including neural networks trained to predict steering angles of the car, camera mounts, and dozens of hours of real driving data.
![](https://i.imgur.com/A5J3qSt.jpg)

15. [Waymo Open Dataset](https://waymo.com/open/#)<br>
The Waymo Open dataset is an open-source multimodal sensor dataset for autonomous driving. Extracted from Waymo self-driving vehicles, the data covers a wide variety of driving scenarios and environments. It contains 1000 types of different segments where each segment captures 20 seconds of continuous driving, corresponding to 200,000 frames at 10 Hz per sensor.

### KITTI 
<table>
  <tr>
  <td><img src="http://www.cvlibs.net/datasets/kitti/images/passat_sensors.jpg"></td>
  <td><iframe width="460" height="260" src="https://www.youtube.com/embed/KXpZ6B1YB_k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
  </tr>
</table>

## A Survey of State-of-Art Technologies
**Paper:** [Autonomous Driving with Deep Learning: A Survey of State-of-Art Technologies](https://arxiv.org/abs/2006.06091)
**Sysyem Diagram:** HW and SW of the autonomous driving platform
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Autonomous_Driving_platform.png?raw=true"></p>

---
## LiDAR-based 3D Object Detection Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/LiDAR_3D_Object_Detection_methods.png?raw=true"></p>

### YOLO3D
**Paper:** [YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud](https://arxiv.org/abs/1808.02350)<br>
**Code:** [maudzung/YOLO3D-YOLOv4-PyTorch](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch)<br>

![](https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch/blob/master/docs/demo.gif?raw=true)
* **Inputs:** Bird-eye-view (BEV) maps that are encoded by **height, intensity and density** of 3D LiDAR point clouds.
* **The input size:** 608 x 608 x 3
* **Outputs:** 7 degrees of freedom (**7-DOF**) of objects: (`cx, cy, cz, l, w, h, θ`)

**YOLOv4 architecture**<br>
<img  width="70%" height="70%" src="https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch/blob/master/docs/yolov4_architecture.png?raw=true">

---
### YOLO4D
**Paper:** [YOLO4D: A Spatio-temporal Approach for Real-time Multi-object Detection and Classification from LiDAR Point Clouds](https://openreview.net/pdf?id=B1xWZic29m)

![](https://d3i71xaburhd42.cloudfront.net/ad16b7bf2ee7c84823f69a3bee6a30aa07311073/4-Figure1-1.png)

---
## Camera-based 3D Object Detection Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Camera_based_3D_Object_Detection_methods.png?raw=true"></p>

### Pseudo-LiDAR
**Paper:** [Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1812.07179)<br>

![](https://mileyan.github.io/pseudo_lidar/cvpr2018-pipeline.png)

<iframe width="555" height="312" src="https://www.youtube.com/embed/mNtXTTo6wzI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
---
### Stereo R-CNN based 3D Object Detection
**Paper:** [Stereo R-CNN based 3D Object Detection for Autonomous Driving](https://arxiv.org/abs/1902.09738)<br>
**Code:** [HKUST-Aerial-Robotics/Stereo-RCNN](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)<br>

![](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN/blob/master/doc/system.png?raw=true)

---
## Monocular Camera-based Depth Estimation Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Monocular_Camera_based_Depth_Estimation_methods.png?raw=true"></p>

### DF-Net
**Paper:** [DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Task Consistency](https://arxiv.org/abs/1809.01649)<br>
**Code:** [vt-vl-lab/DF-Net](https://github.com/vt-vl-lab/DF-Net)<br>
* Model in the paper uses 2-frame as input, while this code uses 5-frame as input (you might use any odd numbers of frames as input, though you would need to tune the hyper-parameters)
* FlowNet in the paper is pre-trained on SYNTHIA, while this one is pre-trained on Cityscapes

![](https://d3i71xaburhd42.cloudfront.net/c01876292b5d1ce6e746fd2e2053453847905bb2/5-Figure3-1.png)

<iframe width="860" height="360" src="https://filebox.ece.vt.edu/~ylzou/eccv2018dfnet/short_teaser.mp4"></iframe>

---
## Depth Fusion Methods with LiDAR and Camera
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Depth_Fusion_methods_with_LiDAR_and_Camera.png?raw=true"></p>

### DFineNet
**Paper:** [DFineNet: Ego-Motion Estimation and Depth Refinement from Sparse, Noisy Depth Input with RGB Guidance](https://arxiv.org/abs/1903.06397)<br>
**Code:** [vt-vl-lab/DF-Net](https://github.com/vt-vl-lab/DF-Net)<br>

<img width="70%" height="70%" src="https://d3i71xaburhd42.cloudfront.net/8e1c2ef0816598866e110362583c1c4f570401d3/3-Figure2-1.png">
<img src="https://d3i71xaburhd42.cloudfront.net/8e1c2ef0816598866e110362583c1c4f570401d3/4-Figure3-1.png">
<img src="https://www.researchgate.net/profile/Ty-Nguyen-5/publication/331840392/figure/fig4/AS:737828568834049@1552923447261/Qualitative-results-of-our-method-left-RGB-guide-certainty-34-middle-ranking-1st.ppm">

---
## 3D Object Detection Methods with LiDAR and Camera
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/3D_Object_Detection_methods with LiDAR_and_Camera.png?raw=true"></p>

### PointFusion
**Paper:** [PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation](https://arxiv.org/abs/1711.10871)<br>
**Code:** [mialbro/PointFusion](https://github.com/mialbro/PointFusion)<br>

<img src="https://media.arxiv-vanity.com/render-output/5472384/x1.png">
<img width="70%" height="70%" src="https://media.arxiv-vanity.com/render-output/5472384/sun-qualitative-3x4.png">

---
### MVX-Net
**Paper:** [MVX-Net: Multimodal VoxelNet for 3D Object Detection](https://arxiv.org/abs/1904.01649)<br>
**Code:** [mmdetection3d/mvxnet](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/mvxnet)<br>

![](https://www.researchgate.net/profile/Vishwanath-Sindagi/publication/332186784/figure/fig2/AS:743809176567810@1554349335734/Overview-of-the-proposed-MVX-Net-PointFusion-method-The-method-uses-convolutional.ppm)
![](https://www.researchgate.net/profile/Vishwanath-Sindagi/publication/332186784/figure/fig4/AS:743809180778497@1554349336248/Sample-3D-detection-results-from-KITTI-validation-dataset-projected-onto-image-for.jpg)

---
### DeepVO
**Paper:** [DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429)<br>
**Code:** [ChiWeiHsiao/DeepVO-pytorch](https://github.com/ChiWeiHsiao/DeepVO-pytorch)<br>

![](https://www.researchgate.net/profile/Khaled-Alyousefi/publication/341427132/figure/fig4/AS:891775065534466@1589627152759/DeepVO-Deep-Learning-Architecture-8-In-the-proposed-network-RCNN-takes-a-sequence.jpg)
![](https://camo.githubusercontent.com/775dc6f938052fa29586fed4fde4a477280eda67514af884f7687fbea3e163d4/68747470733a2f2f696d6775722e636f6d2f766f307658676b2e706e67)
<iframe width="832" height="468" src="https://www.youtube.com/embed/M4v_-XyYKHY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Pedestrain Behavior Prediction Methods
<p align="center"><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Pedestrain_Behavior_Prediction_methods.png?raw=True"></p>

### Predicting Future Person Activities and Locations in Videos
**Paper:** [Peeking into the Future: Predicting Future Person Activities and Locations in Videos](https://arxiv.org/abs/1902.03748)<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Peaking_Into_The_Future_overview.png?raw=true)

<table>
  <tr>
  <td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Peaking_Into_The_Future_PIM.png?raw=true"></td> 
  <td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Peaking_Into_The_Future_PBM.png?raw=true"></td>
  </tr>
</table>

### Spectral Trajectory and Behavior Prediction
**Papers:** [Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clustering in Graph-LSTMs](https://arxiv.org/abs/1912.01118)<br>
**Code:** [Xiejc97/Spectral-Trajectory-and-Behavior-Prediction](https://github.com/Xiejc97/Spectral-Trajectory-and-Behavior-Prediction)<br>
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
**Blog:** [End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)<br>

![](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/data-collection-system-624x411.png)
![](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/training-624x291.png)
<img width="512" height="512" src="https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png">
<iframe width="740" height="416" src="https://www.youtube.com/embed/NJU9ULQUwng" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### End-to-End Learning of Driving Models
**Paper:** [End-to-End Learning of Driving Models with Surround-View Cameras and Route Planners](https://arxiv.org/abs/1803.10158)<br>

![](https://d3i71xaburhd42.cloudfront.net/951dede2758451825dceed238d356a3a984c3670/2-Figure1-1.png)

---
### End-to-end Learning in Simulated Urban Environments
**Paper:** [Autonomous Vehicle Control: End-to-end Learning in Simulated Urban Environments](https://arxiv.org/abs/1905.06712)<br>

![](https://d3i71xaburhd42.cloudfront.net/c567e98e6d7c9d6f74223315e729708553b38103/6-Figure1-1.png)

---
### NeuroTrajectory
**Paper:** [NeuroTrajectory: A Neuroevolutionary Approach to Local State Trajectory Learning for Autonomous Vehicles](https://arxiv.org/abs/1906.10971)<br>

![](https://www.researchgate.net/profile/Sorin-Grigorescu/publication/334155766/figure/fig2/AS:837409541455872@1576665401121/Local-state-trajectory-estimation-for-autonomous-driving-Given-the-current-position-of.ppm)
![](https://www.researchgate.net/profile/Sorin-Grigorescu/publication/334155766/figure/fig3/AS:837409541459968@1576665401194/Examples-of-synthetic-a-GridSim-and-b-real-world-occupancy-grids-The-top-images-in.ppm)
![](https://www.researchgate.net/profile/Sorin-Grigorescu/publication/334155766/figure/fig4/AS:837409541468160@1576665401291/Deep-neural-network-architecture-for-estimating-local-driving-trajectories-The-training.ppm)

---
### ChauffeurNet [(Waymo)](https://sites.google.com/view/waymo-learn-to-drive/)
**Paper:** [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079)<br>
**Code:** [aidriver/ChauffeurNet](https://github.com/aidriver/ChauffeurNet)<br>

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

**Github:** [Iftimie/ChauffeurNet](https://github.com/Iftimie/ChauffeurNet)<br>

![](https://github.com/Iftimie/ChauffeurNet/blob/master/assets/carla-sim.gif?raw=true)

---
## AirSim
**[Github](https://github.com/microsoft/AirSim)**<br> 
**[Document](https://microsoft.github.io/AirSim/)**<br>

Drones in AirSim
<iframe width="709" height="399" src="https://www.youtube.com/embed/-WfTr1-OBGQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Cars in AirSim
<iframe width="683" height="399" src="https://www.youtube.com/embed/gnz1X3UNM5Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Install AirSim
**[Releases](https://github.com/Microsoft/AirSim/releases)** (.zip)<br>

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

* **Download & Unzip** a environment (AirSimNH.zip)<br>
* Find the exe file, **click to run**<br>
> Windows : *AirSimNH/WindowsNoEditor/AirSimNH/Binaries/Win64/AirSimNH.exe* <br>
> Linux   : *AirSimNH/LinuxNoEditor/AirSimNH/Binaries/Linux/AirSimNH* <br>

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
&emsp; recorded file path will be shown on screen

For Car, 
```
VehicleName TimeStamp   POS_X   POS_Y   POS_Z   Q_W Q_X Q_Y Q_Z Throttle    Steering    Brake   Gear    Handbrake   RPM Speed   ImageFile
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirSimNH_rec_txt.png?raw=true)

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

1. Keep AirSim Env running first<br>
`cd AirSimNH/LinuxNoEditor` <br>
`./AirSimNH.sh -ResX=640 -ResY=480 -windowed` (use AirSimNH.bat for Windows) <br>

2. To train DQN model for 500,000 eposides <br>
`pip install gym`<br>
`pip install stable-baselines3`<br>
`git clone https://github.com/rkuo2000/AirSim` <br>
`cd AirSim/PythonClient/reinforcement_learning` <br>
`python dqn_car.py` <br>

[dqn_car.py](https://github.com/rkuo2000/AirSim/blob/master/PythonClient/reinforcement_learning/dqn_car.py)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirSimNH_DQN_Car.png?raw=true)

[a2c_car.py](https://github.com/rkuo2000/AirSim/blob/master/PythonClient/reinforcement_learning/a2c_car.py)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirSimNH_A2C_Car.png?raw=true)

---
### [Autonomous Driving Cookbook](https://github.com/Microsoft/AutonomousDrivingCookbook)
<table>
  <tr>
  <td><img src="https://github.com/microsoft/AutonomousDrivingCookbook/raw/master/AirSimE2EDeepLearning/car_driving.gif?raw=true"></td>
  <td><img width="400" height="220" src="https://github.com/microsoft/AutonomousDrivingCookbook/raw/master/DistributedRL/car_driving_2.gif?raw=true"></td>
  </tr>
</table>

Currently, the following tutorials are available:

- [Autonomous Driving using End-to-End Deep Learning: an AirSim tutorial](https://github.com/microsoft/AutonomousDrivingCookbook/tree/master/AirSimE2EDeepLearning)
- [Distributed Deep Reinforcement Learning for Autonomous Driving](https://github.com/microsoft/AutonomousDrivingCookbook/tree/master/DistributedRL)

**[[AirSim settings]](https://microsoft.github.io/AirSim/settings/)**

---
### [Kaggle: AirSim End-to-End Learning](https://kaggle.com/rkuo2000/airsim-end-to-end-learning)
#### Build Model with input image and state (driving angle)
```
image_input_shape = sample_batch_train_data[0].shape[1:]
state_input_shape = sample_batch_train_data[1].shape[1:]

#Create the convolutional stacks
pic_input = layers.Input(shape=image_input_shape)

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pic_input)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)

#Inject the state input
state_input = layers.Input(shape=state_input_shape)
m = layers.concatenate([x, state_input])

# Add a few dense layers to finish the model
m = layers.Dense(64, activation='relu')(m)
m = layers.Dropout(0.2)(m)
m = layers.Dense(10, activation='relu')(m)
m = layers.Dropout(0.2)(m)
m = layers.Dense(1)(m)

model = models.Model(inputs=[pic_input, state_input], outputs=m)

model.summary()
```
#### Test Result
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirSim_End_to_End_Learning.png?raw=true)

<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

