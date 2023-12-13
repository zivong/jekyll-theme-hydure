---
layout: post
title: Reinforcement Learning for Robotics
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Introduction to Reinforcement Learning for Robot / Drone.

---
## Embodied AI
**Blog:** [Overview of Embodied Artificial Intelligence](https://medium.com/machinevision/overview-of-embodied-artificial-intelligence-b7f19d18022)<br>
Embodied AI is the field for solving AI problems for virtual robots that can move, see, speak, and interact in the virtual world and with other virtual robots — these simulated robot solutions are then transferred to real world robots<br>
The simulated environments for Embodied AI training: SUNCG, Matterport3D, iGibson, Replica, Habitat, and DART<br>

---
### Matterport3D
**Paper:** [Matterport3D: Learning from RGB-D Data in Indoor Environments](https://arxiv.org/abs/1709.06158)<br>
**Code:** [Matterport3D](https://github.com/niessner/Matterport)<br>

![](https://github.com/niessner/Matterport/blob/master/img/teaser.jpg?raw=true)

---
### Replica 
**Code:** [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset)<br>
![](https://github.com/facebookresearch/Replica-Dataset/blob/main/assets/ReplicaModalities.png?raw=true)

---
### iGibson
**Paper:** [iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks](https://arxiv.org/abs/2108.03272)<br>
**Code:** [StanfordVL/iGibson](https://github.com/StanfordVL/iGibson)<br>
![](https://github.com/StanfordVL/iGibson/blob/master/docs/images/igibson.gif?raw=true)

---
### Habitat 2.0
**Paper:** [Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405)<br>
**Code:** [facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim)<br>
<video controls>
  <source src="https://user-images.githubusercontent.com/2941091/126080914-36dc8045-01d4-4a68-8c2e-74d0bca1b9b8.mp4" type="video/mp4">
</video>

---
## Indoor Navigation

### Autonomous Indoor Robot Navigation
**Paper:** [Deep Reinforcement learning for real autonomous mobile robot navigation in indoor environments](https://arxiv.org/abs/2005.13857)<br>
**Code:** [](https://github.com/RoblabWh/RobLearn)<br>
<iframe width="742" height="417" src="https://www.youtube.com/embed/KyA2uTIQfxw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://d3i71xaburhd42.cloudfront.net/8a47843c2e664e5e7e218e2d891726d023619403/3-Figure4-1.png)

---
### DDPG 路徑規劃
**Blog:** [智慧送餐服務型機器人導航路徑之設計](https://www.phdbooks.com.tw/cn/magazine/detail/1225)<br>
路徑跟隨器有四個主軸：<br>
* 送餐路徑生成：從文件或上層發佈訊息獲取預先定義的路徑。
* 編輯航線路徑點：清除路徑中不合適的航線路徑點。
* MFAC無模型自適應控制之航段管制：自動調整送餐路徑之導航點之間的航段長度，依序共分成路徑跟隨之依據以及MFAC無模型自適應控制之應用。
* DWA之區域路徑傳遞：依照MFAC調整之結果，產出相關生成路徑，並以DWA進行區域設定。

* **自走車基於DDPG的室內路徑規劃**<br>
<iframe width="506" height="285" src="https://www.youtube.com/embed/TNRjb8q6XxM" title="自走車基於DDPG的室內路徑規劃" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Long-Range Indoor Navigation
**Paper:** [Long-Range Indoor Navigation with PRM-RL](https://arxiv.org/abs/1902.09458)<br>
<iframe width="742" height="417" src="https://www.youtube.com/embed/xN-OWX5gKvQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Long-Range-Indoor-Navigation.png?raw=true)

---
## Gym-Gazebo
**Code:** [erlerobot/gym-gazebo](https://github.com/erlerobot/gym-gazebo)<br>
<table>
<tr>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboCircuit2TurtlebotLidar-v0.png"></td>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/cartpole.jpg"></td>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboModularScara3DOF-v3.png"></td>
</tr>
</table>

---
## DART (Dynamic Animation and Robotics Toolkit)

### [Dartsim/dart](https://github.com/dartsim/dart)
* Python bindings: dartpy, pydart2 (deprecated)
* OpenAI Gym with DART support: gym-dart (dartpy based), DartEnv (pydart2 based, deprecated)

<iframe width="742" height="417" src="https://www.youtube.com/embed/Ve_MRMTvGX8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## [PyBullet-Gym](https://github.com/benelot/pybullet-gym)
[PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)<br>

**code:** [rkuo2000/pybullet-gym](https://github.com/rkuo2000/pybullet-gym)<br>
* installation
```
pip install gym
pip install stable-baselines3
git clone https://github.com/rkuo2000/pybullet-gym
export PYTHONPATH=$PATH:/home/yourname/pybullet-gym
```

**Train**<br>
`python train.py Ant 10000000`<br>

**Enjoy** with trained-model<br>
`python enjoy.py Ant`<br>

**Enjoy** with pretrained weights<br>
`python enjoy_Ant.py`<br>
`python enjoy_HumanoidFlagrunHarder.py` (a copy from pybulletgym/examples/roboschool-weights/enjoy_TF_*.py)<br>

---
### [PyBullet-Robots](https://github.com/erwincoumans/pybullet_robots)
<img width="50%" height="50%" src="https://raw.githubusercontent.com/erwincoumans/pybullet_robots/master/images/collection.png">

**env_name = "AtlasPyBulletEnv-v0"**<br>
[atlas_v4_with_multisense.urdf](https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/assets/robots/atlas/atlas_description/atlas_v4_with_multisense.urdf)<br>
<iframe width="580" height="435" src="https://www.youtube.com/embed/aqAk701ylIk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## RoboCar Gym

### Pybullet-RoboCar
**Blog:** <br>
[Creating OpenAI Gym Environments with PyBullet (Part 1)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24)<br>
[Creating OpenAI Gym Environments with PyBullet (Part 2)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e)<br>
![](https://media0.giphy.com/media/VI3OuvQShK3gzENiVz/giphy.gif?cid=790b761131bda06b74fcd9bb06c6a43939cf446edf403a68&rid=giphy.gif&ct=g)

---
## Quadruped Gym

### [Motion Imitation](https://github.com/google-research/motion_imitation)
**Code:** [TF 1.15](https://github.com/google-research/motion_imitation)<br>
<iframe width="784" height="441" src="https://www.youtube.com/embed/NPvuap-SD78" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
**Code:** [PyTorch](https://github.com/newera-001/motor-system)<br>
For Training:<br>
`python motion_imitation/run_torch.py --mode train --motion_file 'dog_pace.txt|dog_spin.txt' \
--int_save_freq 10000000 --visualize --num_envs 50 --type_name 'dog_pace'`<br>
For Testing:<br>
`python motion_imitation/run_torch.py --mode test --motion_file 'dog_pace.txt' --model_file 'file_path' \ 
--encoder_file 'file_path' --visualize`<br>

---
### Rex: an open-source quadruped robot
**Code:** [nicrusso7/rex-gym](https://github.com/nicrusso7/rex-gym)<br>
![](https://github.com/nicrusso7/rex-gym/blob/master/images/intro.gif?raw=true)

---
## Drones Gym

### [PyBullet-Gym for Drones](https://github.com/utiasDSL/gym-pybullet-drones)
![](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/files/readme_images/helix.gif?raw=true)
![](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/files/readme_images/helix.png?raw=true)

* Installation
```
sudo apt install ffmpeg
pip install numpy pillow matplotlib cycler
pip install gym pybullet stable_baselines3 ray[rllib]
git clone https://github.com/rkuo2000/gym-pybullet-drones.git
cd gym-pybullet-drones
```

* Train & Enjoy<br>
`python train.py` # modify train.py for different env, algorithm and timesteps<br>
`python enjoy.py` # modify enjoy.py for different env<br>

* Fly using [DSLPIDControl.py](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/control/DSLPIDControl.py):（PID飛行）<br>
`python examples/fly.py --num_drones 1`<br>
![](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/files/readme_images/wp.gif?raw=true)

* To learn take-off:（起飛）  <br>
`python examples/learn.py`<br>
![](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/files/readme_images/learn2.gif?raw=true)

* `compare.py` which replays and compare to a trace saved in `files/example_trace.pkl`

**Experiments**<br>
`cd experiments/learning`<br>

env : hover, takeoff, flythrugate, tune（旋停, 起飛, 穿越, 調整）<br>
algo: a2c, ppo, sac, td3, ddpg<br>

* To learn hover:（旋停）<br>
`python singleagent.py --env hover --algo a2c`<br>

To visualize the best trained agent:<br>
`python test_singleagent.py --exp ./results/save-hover-a2c`<br>

For multi-agent RL, using rllib:<br>
`python multiagent.py --num_drones 3 --env hover --algo a2c --num_workers 2`<br>

---
### [Flightmare](https://github.com/uzh-rpg/flightmare)
Flightmare is a flexible modular quadrotor simulator. 
<iframe width="768" height="432" src="https://www.youtube.com/embed/m9Mx1BCNGFU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<img width="50%" height="50%" src="https://github.com/uzh-rpg/flightmare/raw/master/docs/flightmare.png">
* [Introduction](https://github.com/uzh-rpg/flightmare/wiki/Introduction)
* [Prerequisites](https://github.com/uzh-rpg/flightmare/wiki/Prerequisites)
* [Install Python Packages](https://github.com/uzh-rpg/flightmare/wiki/Install-with-pip)
* [Install ROS](https://github.com/uzh-rpg/flightmare/wiki/Install-with-ROS)

* **running ROS**
```
roslaunch flightros rotors_gazebo.launch
```

* **flighRL**<br>
```
cd /path/to/flightmare/flightrl
pip install .
cd examples
python3 run_drone_control.py --train 1
```

---
### [AirSim](https://github.com/microsoft/AirSim)
<iframe width="768" height="448" src="https://www.youtube.com/embed/-WfTr1-OBGQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://github.com/microsoft/AirSim/blob/master/docs/images/AirSimDroneManual.gif?raw=true)

---
## Assistive Gym

**Paper:** [Assistive Gym: A Physics Simulation Framework for Assistive Robotics](https://arxiv.org/abs/1910.04700)<br>
![](https://github.com/Healthcare-Robotics/assistive-gym/blob/main/images/assistive_gym.jpg?raw=true)
<iframe width="705" height="397" src="https://www.youtube.com/embed/EFKqNKO3P60" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

* Four collaborative robots (PR2, Jaco, Baxter, Sawyer)<br>
![](https://github.com/Healthcare-Robotics/assistive-gym/raw/main/images/robot_models.gif)

* Support for the Stretch and PANDA robots<br>
<table>
<tr>
<td><img src="https://github.com/Healthcare-Robotics/assistive-gym/blob/main/images/v1_stretch.jpg?raw=true"></td>
<td><img src="https://github.com/Healthcare-Robotics/assistive-gym/blob/main/images/v1_panda.jpg?raw=true"></td>
</tr>
</table>

**Code:** [Healthcare-Robotics/assistive-gym](https://github.com/Healthcare-Robotics/assistive-gym)<br>

---
### Assistive VR Gym
**Paper:** [Assistive VR Gym: Interactions with Real People to Improve Virtual Assistive Robots](https://arxiv.org/abs/2007.04959)<br>
**Code:** [Healthcare-Robotics/assistive-vr-gym](https://github.com/Healthcare-Robotics/assistive-vr-gym)<br>
![](https://github.com/Healthcare-Robotics/assistive-vr-gym/blob/master/images/avr_gym_2.jpg?raw=true)
<iframe width="705" height="397" src="https://www.youtube.com/embed/tcyPMkAphNs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Learning Dexity
<iframe width="696" height="392" src="https://www.youtube.com/embed/jwSbzNHGflM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [Dexterous Gym](https://github.com/henrycharlesworth/dexterous-gym)
![](https://github.com/henrycharlesworth/dexterous-gym/raw/master/dexterous_gym/examples/penspin.gif)
![](https://github.com/henrycharlesworth/dexterous-gym/raw/master/dexterous_gym/examples/egghandover.gif)

---
### [DexPilot](https://research.nvidia.com/publication/2020-05_dexpilot-vision-based-teleoperation-dexterous-robotic-hand-arm-system)
**Paper:** [DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System](https://arxiv.org/abs/1910.03135)<br>
![](https://research.nvidia.com/sites/default/files/styles/wide/public/publications/dexpilot.jpg?itok=7Re04wXI)
<iframe width="883" height="497" src="https://www.youtube.com/embed/qGE-deYfb8I" title="dexpilot highlights" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [TriFinger](https://sites.google.com/view/trifinger)
<iframe width="464" height="287" src="https://www.youtube.com/embed/RxkS6dzO1dU" title="Writing" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**Paper:** [TriFinger: An Open-Source Robot for Learning Dexterity](https://arxiv.org/abs/
2008.03596)<br>
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/f223b2a438fe20ac55300c278509bf4a13072ca8/3-Figure2-1.png)
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/f223b2a438fe20ac55300c278509bf4a13072ca8/4-Figure3-1.png)
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/f223b2a438fe20ac55300c278509bf4a13072ca8/4-Figure4-1.png)

**Code:** [TriFinger Robot Simulation](https://github.com/open-dynamic-robot-initiative/trifinger_simulation)<br>
<iframe width="736" height="410" src="https://www.youtube.com/embed/V767AGlyDOs" title="CoRL 2020, Spotlight Talk 421: TriFinger: An Open-Source Robot for Learning Dexterity" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

--- 
### [Multi-Task Reset-Free (MTRF) Learning](https://sites.google.com/view/mtrf)
**Paper:** [Reset-Free Reinforcement Learning via Multi-Task Learning: Learning Dexterous Manipulation Behaviors without Human Intervention](https://arxiv.org/abs/2104.11203)<br>
<iframe width="562" height="296" src="https://www.youtube.com/embed/64FLPhvqgrw" title="MTRF Overview: Reset-Free Reinforcement Learning via Multi-Task Learning" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Dexterous Anthropomorphic Robotic Hand
**Blog:** [Robotic hand can crush beer cans and hold eggs without breaking them](https://www.newscientist.com/article/2301641-robotic-hand-can-crush-beer-cans-and-hold-eggs-without-breaking-them/)<br>
![](https://images.newscientist.com/wp-content/uploads/2021/12/14124151/PRI_215070295.jpg?width=778)

**Paper:** [Integrated linkage-driven dexterous anthropomorphic robotic hand](https://www.nature.com/articles/s41467-021-27261-0#Abs1)<br>
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41467-021-27261-0/MediaObjects/41467_2021_27261_Fig2_HTML.png?as=webp)

<iframe width="883" height="497" src="https://www.youtube.com/embed/TJzfgipEACU" title="Watch a highly dexterous robotic hand use scissors and tweezers" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Robotic Telekinesis
[Robotic Telekinesis: Learning a Robotic Hand Imitator by Watching Humans on Youtube](https://arxiv.org/abs/2202.10448)<br>
<iframe width="883" height="497" src="https://www.youtube.com/embed/fVrcBY0lOWw" title="Finally, Robotic Telekinesis is Here! 🤖" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Fixed-Finger Gripper
**Paper:** [F1 Hand: A Versatile Fixed-Finger Gripper for Delicate Teleoperation and Autonomous Grasping](https://arxiv.org/abs/2205.07066)<br>
<iframe width="884" height="497" src="https://www.youtube.com/embed/iWXXIX4Mkl8" title="F1 Hand: A Versatile Fixed-Finger Gripper for Delicate Teleoperation and Autonomous Grasping" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Learning Diverse Dexterous Grasps
**Paper:** [Learning Diverse and Physically Feasible Dexterous Grasps with Generative Model and Bilevel Optimization](https://arxiv.org/abs/2207.00195)<br>
<iframe width="883" height="497" src="https://www.youtube.com/embed/9DTrImbN99I" title="Learning Diverse & Physically Feasible Dexterous Grasps w/ Generative Model and Bilevel Optimization" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### ViLa
**Blog:** [https://bangqu.com/9Fa2ra.html](https://bangqu.com/9Fa2ra.html)<br>
![](https://i3.res.bangqu.com/farm/j/news/2023/12/12/42e2981b43a54c3ac7a8647515a0ecaa.gif)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


