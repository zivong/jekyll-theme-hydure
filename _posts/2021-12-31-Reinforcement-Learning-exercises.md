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
### [SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)<br>
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

<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_CrazyClimber_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_DoubleDunk_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_IceHockey_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_boxing_ppo.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_qbert_ppo.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_tennis_a2c.gif?raw=true"></td>
</tr>
</table>

---
### [ElegentRL examples](https://github.com/AI4Finance-Foundation/ElegantRL)
![](https://miro.medium.com/max/2000/1*YE_Tzguk0LuGSuvGvzl7kQ.png)

* DDPG, TD3, SAC, PPO, PPO (GAE),REDQ for continuous actions
* DQN, DoubleDQN, D3QN, SAC for discrete actions
* QMIX, VDN; MADDPG, MAPPO, MATD3 for multi-agent environment

`cd ~/RL-gym`<br>
`pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git`<br>
`cd elegantrl`<br>
`python train_pendulum.py`<br>
`python train_bipedalwalker.py`<br>

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
## Assistive Gym
### [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym)
Using Pybullet & Tensorflow 1.14.0<br>
`conda create -n assist`<br>
`conda activate assist`<br>

`git clone https://github.com/Healthcare-Robotics/assistive-gym.git`<br>
`cd assistive-gym`<br>
`pip3 install -e .`<br> 

`python3 -m assistive_gym --env "FeedingJaco-v1"`<br>

---
### [Assistive VR Gym](https://github.com/Healthcare-Robotics/assistive-vr-gym)
`conda create -n assist-vr`<br>
`conda activate assist-vr`<br>

`git clone -b vr https://github.com/Zackory/bullet3.git`<br>
`cd bullet3`<br>
`pip install .`<br>

`git clone https://github.com/Healthcare-Robotics/assistive-vr-gym.git`<br>
`cd assistive-vr-gym`<br>
`pip install .`<br>

---
## Drone Gym
### [PyBullet-Gym for Drones](https://github.com/utiasDSL/gym-pybullet-drones)
`sudo apt install ffmpeg`<br>
`pip install numpy pillow matplotlib cycler`<br>
`pip install gym pybullet stable_baselines3 'ray[rllib]'`<br>
`git clone https://github.com/utiasDSL/gym-pybullet-drones.git`<br>
`cd gym-pybullet-drones`<br>
`pip install -e .`<br>

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
* [Introduction](https://github.com/uzh-rpg/flightmare/wiki/Introduction)
* [Prerequisites](https://github.com/uzh-rpg/flightmare/wiki/Prerequisites)
* [Install Python Packages](https://github.com/uzh-rpg/flightmare/wiki/Install-with-pip)
* [Install ROS](https://github.com/uzh-rpg/flightmare/wiki/Install-with-ROS)

**running ROS**<br>
`roslaunch flightros rotors_gazebo.launch`<br>

**flighRL**<br>
`cd /path/to/flightmare/flightrl`<br>
`pip install .`<br>
`cd examples`<br>
`python3 run_drone_control.py --train 1`<br>

---
## Stock RL

### Stock Trading
**Kaggle:** [Stock-DQN](https://kaggle.com/rkuo2000/stock-dqn)<br>
`cd ~/RL-gym/stock`<br>
`python train_dqn.py`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/stock_dqn.png?raw=true)

`python enjoy_dqn.py`<br>


---
### FinRL
**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


