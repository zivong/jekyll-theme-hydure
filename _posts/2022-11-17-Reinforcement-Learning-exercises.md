---
layout: post
title: Reinforcement Learning Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes examples for running OpenAI Gym, RL-gym, Gaming RL, Multi Agents, Robot Gym, Assistive Gym, Drone Gym, FinRL.

---
## RL in C++
### DQN RobotCar
**Code:** [https://github.com/aslamahrahman/Policy-Gradient-Network-Arduino](https://github.com/aslamahrahman/Policy-Gradient-Network-Arduino)<br>
<iframe width="506" height="285" src="https://www.youtube.com/embed/d7NcoepWlyU" title="Real time reinforcement learning demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>




---
## [PyBullet Gymperium](https://github.com/benelot/pybullet-gym)

`pip install pybullet`<br>

**Repro [rkuo2000/pybullet-gym](https://github.com/rkuo2000/pybullet-gym)**<br>
`git clone https://github.com/rkuo2000/pybullet-gym`<br>
`export PYTHONPATH=$PATH:/home/yourname/pybullet-gym`
### gym

`cd ~/pybullet-gym/gym`<br>
`python enjoy_HumanoidFlagrunHarder.py` (copy from pybulletgym/examples/roboschool-weights/enjoy_TF_*.py)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PyBullet_Gym_HumanoidFlagrunHarder.gif?raw=true)

**Env_Name:** *Ant, Atlas, HalfCheetah, Hopper, Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder, InvertedPendulum, InvertedDoublePendulum, InvertedPendulumSwingup, Reacher, Walker2D*<br>
`python train.py Ant 10000000`<br>
`python enjoy.py Ant`<br>

---
### [unitree](https://github.com/unitreerobotics/unitree_pybullet)
`cd ~/pybullet-gym/unitree`<br>
`python a1.py`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PyBullet_Gym_Unitree_A1.gif?raw=true)

---
## Multi Agents
### Unity ML-agents playing Volleyball
**Code:** [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)<br>
**Blog:** [A hands-on introduction to deep reinforcement learning using Unity ML-Agents](https://medium.com/coder-one/a-hands-on-introduction-to-deep-reinforcement-learning-using-unity-ml-agents-e339dcb5b954)<br>

![](https://miro.medium.com/max/540/0*ojKXHwzo_a-rjwpz.gif)

---
## Robot Gym
### [erlerobot/gym-gazebo](https://github.com/erlerobot/gym-gazebo)<br>
<img width="50%" height="50%" src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboCircuit2TurtlebotLidar-v0.png">

---
### [Motion Imitation](https://github.com/google-research/motion_imitation)

**Code:** [PyTorch](https://github.com/newera-001/motor-system)<br>
For Training:<br>
`python motion_imitation/run_torch.py --mode train --motion_file 'dog_pace.txt|dog_spin.txt' \
--int_save_freq 10000000 --visualize --num_envs 50 --type_name 'dog_pace'`<br>
For Testing:<br>
`python motion_imitation/run_torch.py --mode test --motion_file 'dog_pace.txt' --model_file 'file_path' \ 
--encoder_file 'file_path' --visualize`<br>

---
### [Rex-Gym](https://github.com/nicrusso7/rex-gym)

![](https://github.com/nicrusso7/rex-gym/blob/master/images/intro.gif?raw=true)

---
## Car Simulator

### DDQN Donkeycar
[Train Donkey Car in Unity Simulator with Reinforcement Learning](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html)<br>
![](https://flyyufelix.github.io/img/ddqn_demo.gif)
![](https://flyyufelix.github.io/img/donkey_racing.gif)

* [Donkey Simulator User Guide](https://docs.donkeycar.com/guide/simulator/)

* Install

```
cd ~/projects
git clone https://github.com/tawnkramer/gym-donkeycar
cd gym-donkeycar
conda activate donkey
pip install -e .[gym-donkeycar]
```

```
donkey createcar --path ~/mysim
cd ~/mysim
```
Edit your myconfig.py (replace <user-name> with yours<br>
```
DONKEY_GYM = True
DONKEY_SIM_PATH = "/home/<user-name>/projects/DonkeySimLinux/donkey_sim.x86_64"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"
```
* Drive
`python manage.py drive`<br>
`python manage.py drive --js` (Unbuntu plug-in joystick)<br>
* Train
`donkey train --tub ./data --model models/mypilot.h5`<br>
* Test
`python manage.py drive --model models/mypilot.h5`<br>

---
## MuZero
### [MuZero General](https://github.com/werner-duvaud/muzero-general)
`git clone https://github.com/werner-duvaud/muzero-general`<br>
`cd muzero-general`<br>
`pip install -r requirements.txt`<br>

`tensorboard --logdir ./results`<br>

`python muzero.py`<br>
```
Welcome to MuZero! Here's a list of games:
0. atari
1. breakout
2. cartpole
3. connect4
4. gomoku
5. gridworld
6. lunarlander
7. simple_grid
8. spiel
9. tictactoe
10. twentyone
Enter a number to choose the game: 6
```
2022-01-21 14:43:12,207	INFO services.py:1338 -- View the Ray dashboard at http://127.0.0.1:8265

```
0. Train
1. Load pretrained model
2. Diagnose model
3. Render some self play games
4. Play against MuZero
5. Test the game manually
6. Hyperparameter search
7. Exit
Enter a number to choose an action: 1
```

```
0. Specify paths manually
Enter a number to choose a model to load: 0
Enter a path to the model.checkpoint, or ENTER if none: results/lunarlander/model.checkpoint
```

Using checkpoint from ./results/lunarlander/model.checkpoint

Done

```
0. Train
1. Load pretrained model
2. Diagnose model
3. Render some self play games
4. Play against MuZero
5. Test the game manually
6. Hyperparameter search
7. Exit
Enter a number to choose an action: 3
```
Testing 1/1<br>
(SelfPlay pid=5884) Press enter to take a step <br>

(SelfPlay pid=5884) Tree depth: 10<br>
(SelfPlay pid=5884) Root value for player 0: 17.66<br>
(SelfPlay pid=5884) Played action: 3. Fire right orientation engine<br>
(SelfPlay pid=5884) Press enter to take a step <br>
...<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


