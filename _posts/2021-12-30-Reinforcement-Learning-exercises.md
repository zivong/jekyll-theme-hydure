---
layout: post
title: Reinforcement Learning Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes examples for running OpenAI Gym, RL-gym, Gaming RL, Multi Agents, Robot Gym, Assistive Gym, Drone Gym, FinRL.

---
## Gym
### Installation Gym & SB3
`pip install stable-baselines3`<br>
`pip install pyglet`<br>

For Ubuntu: `pip install gym[atari]`<br>
For Win10 : `pip install --no-index -f ttps://github.com/Kojoley/atari-py/releases atari-py`<br>

Code example:<br>
```
import gym 
env = gym.make('CartPole-v1')

# env is created, now we can use it: 
for episode in range(10): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample() # random action
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
env.close()
```

---
## RL-gym
`cd ~`<br>
`git clone https://github.com/rkuo2000/RL-gym`<br>
`cd RL-gym`<br>

### cartpole
`cd ~/RL-gym/cartpole`<br>

**Ex0.** Env state.shape<br>
`python env_stateshape.py CartPole-v1`<br>
(4,)  CartPole環境輸出的state包括位置、加速度、杆子垂直夾角和角加速度。<br>
`python env_stateshape.py Pong-v4`<br>
(210,160,3)<br>
`python env_stateshape.py Breakout-v4`<br>
(210,160,3)<br>

**Ex1.** Cartpole with random action<br>
`python intro1.py`<br>

**Ex2.** Cartpole, random action, if done: env.reset<br>
`python intro2.py`<br>

**Ex3.** Cartpole Q learning<br>
`python q_learning.py`<br>

**Ex4.** Cartpole DQN<br> 
`python dqn.py`<br>

**Ex5.** Cartpole DDQN (Double DQN)<br>
`python ddqn.py`<br>

**Ex6.** Cartpole A2C<br>
`python a2c.py`<br>

---
### Multi-Env
Code example:<br>
```
class MultiEnv:
    def __init__(self, env_id, num_env):
        self.envs = []
        for _ in range(num_env):
            self.envs.append(gym.make(env_id))

    def reset(self):
        for env in self.envs:
	        env.reset()

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            if done:
                env.reset()
	
        return obs, rewards, dones, infos
        
multi_env = MultiEnv('Breakout-v0', 10)
```

---
### sb3
**[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)**<br>
`cd ~/RL-gym/sb3`<br>

**CartPole, Pendulum, LunarLander:**<br>
* **Train**<br>
`python train.py CartPole 160000`<br>
`python train.py Pendulum 640000`<br>
`python train.py LunarLander 480000`<br>

* **Enjoy**<br>
`python enjoy.py CartPole`<br>
`python enjoy.py Pendulum`<br>
`python enjoy.py LunarLander`<br>
 
* **Enjoy + Gif**<br>
`python enjoy_gif.py LunarLander`<br>

![](https://github.com/rkuo2000/RL-gym/blob/main/assets/CartPole.gif?raw=true)

`python enjoy_gif.py LunarLander`<br>

![](https://github.com/rkuo2000/RL-gym/blob/main/assets/LunarLander.gif?raw=true)

**Atari**<br>
Env Name listed in Env_Name.txt<br>

`python train_atari.py Pong 10000000`<br>
`python enjoy_atari.py Pong`<br>

---
### [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
`git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`<br>
`cd rl-baselines3-zoo`<br>

**Run pretrained agents**
`python enjoy.py --algo a2c --env SpaceInvadersNoFrameskip-v4 --folder rl-trained-agents/ -n 5000`<br>

**Pong**<br>
`python train.py --algo a2c --env PongNoFrameskip-v4 -i rl-trained-agents/a2c/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip -n 5000`<br>
`python enjoy.py --algo a2c --env PongNoFrameskip-v4 --folder logs/ -n 5000`<br>

**Breakout**<br>
`python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i rl-trained-agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000`<br>
`python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder logs/ -n 5000`<br>

**PyBulletEnv**<br>
`python enjoy.py --algo a2c --env AntBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RL_SB3_Zoo_AntBulletEnv.gif?raw=true)
`python enjoy.py --algo a2c --env HalfCheetahBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RL_SB3_Zoo_HalfCheetahBulletEnv.gif?raw=true)
`python enjoy.py --algo a2c --env HopperBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RL_SB3_Zoo_HopperBulletEnv.gif?raw=true)
`python enjoy.py --algo a2c --env Walker2DBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RL_SB3_Zoo_Walker2DBulletEnv.gif?raw=true)

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
### Gym-Gazebo
**Code:** [erlerobot/gym-gazebo](https://github.com/erlerobot/gym-gazebo)<br>
<table>
<tr>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboCircuit2TurtlebotLidar-v0.png"></td>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboModularScara3DOF-v3.png"></td>
</tr>
</table>

---
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
### Rex-Gym
**Code:** [Rex: an open-source quadruped robot](https://github.com/nicrusso7/rex-gym)<br>

![](https://github.com/nicrusso7/rex-gym/blob/master/images/intro.gif?raw=true)

---
## Assistive Gym
### Assistive Gym
**Paper:** [Assistive Gym: A Physics Simulation Framework for Assistive Robotics](https://arxiv.org/abs/1910.04700)<br>
**Code:** [Healthcare-Robotics/assistive-gym](https://github.com/Healthcare-Robotics/assistive-gym)<br>
![](https://github.com/Healthcare-Robotics/assistive-gym/blob/main/images/assistive_gym.jpg?raw=true)
<iframe width="705" height="397" src="https://www.youtube.com/embed/EFKqNKO3P60" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Assistive VR Gym
**Paper:** [Assistive VR Gym: Interactions with Real People to Improve Virtual Assistive Robots](https://arxiv.org/abs/2007.04959)<br>
**Code:** [Healthcare-Robotics/assistive-vr-gym](https://github.com/Healthcare-Robotics/assistive-vr-gym)<br>
![](https://github.com/Healthcare-Robotics/assistive-vr-gym/blob/master/images/avr_gym_2.jpg?raw=true)
<iframe width="705" height="397" src="https://www.youtube.com/embed/tcyPMkAphNs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Drone Gym
### [PyBullet-Gym for Drones](https://github.com/utiasDSL/gym-pybullet-drones)
![](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/files/readme_images/helix.gif?raw=true)
![](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/files/readme_images/helix.png?raw=true)

---
### [Flightmare](https://github.com/uzh-rpg/flightmare)
Flightmare is a flexible modular quadrotor simulator. 
<iframe width="768" height="432" src="https://www.youtube.com/embed/m9Mx1BCNGFU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [AirSim](https://github.com/microsoft/AirSim)
<iframe width="768" height="448" src="https://www.youtube.com/embed/-WfTr1-OBGQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://github.com/microsoft/AirSim/blob/master/docs/images/AirSimDroneManual.gif?raw=true)

---
## Stock RL
### Stock Price
**Kaggle:** [rkuo2000/stock-lstm](https://www.kaggle.com/rkuo2000/stock-lstm)<br>

**LSTM model**<br>
```
model = Sequential()
model.add(Input(shape=(history_points, 5)))
model.add(LSTM(history_points))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
```
**Training parameters:**<br>
```
history_points = 50
batch_size = 32
num_epochs = 100
```

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_LSTM_GOOGL.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_LSTM_MSFT.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_LSTM_AAPL.png?raw=true)

---
### Stock Trading
**Blog:** [Predicting Stock Prices using Reinforcement Learning (with Python Code!)](https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/)<br>
![](https://editor.analyticsvidhya.com/uploads/770801_26xDRHI-alvDAfcPPJJGjQ.png)

**Kaggle:** [Stock-DQN](https://kaggle.com/rkuo2000/stock-dqn)<br>
`cd ~/RL-gym/stock`<br>
`python train_dqn.py`<br>
`python enjoy_dqn.py`<br>

---
### FinRL
**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


