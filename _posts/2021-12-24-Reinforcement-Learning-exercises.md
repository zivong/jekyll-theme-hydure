---
layout: post
title: Reinforcement Learning Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes examples for running OpenAI Gym, Basic RL, Gaming RL, Robot Gym, Assistive Gym, Drone Gym.

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

### Download sample codes
`cd ~`<br>
`git clone https://github.com/rkuo2000/RL-Gym`<br>
`cd RL-Gym`<br>

---
## Basic RL
### RL-Gym
`cd ~/RL-Gym/Cartpole`<br>

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

SB3 code:<br>
```
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('Pong-v4', n_envs=16)
```

---
### Stable Baselines3
**[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)**<br>
```
import gym
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
```

---
### Cartpole/Pendulum SB3
**SB3 Gym sample codes:** [RL-Gym/SB3](https://github.com/rkuo2000/RL-Gym/tree/main/SB3)<br>
**cartpole**<br>
* cartpole_a2c.py
* cartpole_dqn.py
* cartpole_ppo.py

&emsp;`python cartple_a2c.py`<br>
&emsp;`python play_cartpole.py`<br>

**pendulum**<br>
* pendulum_ddpg.py
* pendulum_sac.py
* pendulum_td3.py

&emsp;`python pendulum_ddpg.py`<br>
&emsp;`python play_pendulum.py`<br>

---
### Atari SB3
**SB3 Atari sample codes:** [RL-Gym/SB3](https://github.com/rkuo2000/RL-Gym/tree/main/SB3)
* pong_a2c.py
* breakout_ppo.py
* lunarlander_a2c.py 
* spaceinvaders_a2c.py
* qbert_ppo.py
* pacman_a2c.py

`cd ~/RL-Gym/SB3`<br>

`python pong_a2c.py`<br>
`python play_pong.py`<br>

`python breakout_ppo.py`<br>
`python play_breakout.py`<br>

To report execution time:<br>
```
import time

start_t = time.time()

...

end_t = time.time()
print("Execution Time = ', end_t - start_t)
```

timesteps=int(2e5)<br>
`python pong_dqn.py`<br>

You can train the code on Kaggle.<br>
**Kaggle:** [pong-a2c](https://github.com/rkuo2000/pong-a2c)<br>

---
### LunarLander
`apt install wig`<br>
`apt install box2d box2d_keng`<br>
`pip install Box2D box2d-py`<br>
`pip install pyglet`<br>

`python lunarlander_a2c.py` (n_envs=**64**, timesteps=int(**1e6**))<br> 
`python play_lunarlander.py`<br>

.gif writer<br>
`python gif_lunarlander.py`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/lunarlander_a2c.gif?raw=true)

---
### SuperMario SB3
`pip install gym-super-mario-bros`<br>
`cd ~/RL-Gym/mario`<br>

Training time is around 80 hours on CPU and 20 hours on GPU.<br>
To train  : (epochs=40000)<br>
`python main.py`<br>

To replay : (modify `checkpoint = Path('trained_mario.chkpt')`)<br>
`python replay.py`<br>

![](https://pytorch.org/tutorials/_images/mario.gif)

---
### [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
A Training Framework for Stable Baselines3 Reinforcement Learning Agents<br>

![](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/images/panda_pick.gif?raw=true)

**Train an agent**<br>
The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.<br>

Train with tensorboard support:<br>
`python train.py --algo ppo --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/`<br>

Save a checkpoint of the agent every 100000 steps:<br>
`python train.py --algo td3 --env HalfCheetahBulletEnv-v0 --save-freq 100000`<br>

Continue training (here, load pretrained agent for Breakout and continue training for 5000 steps):<br>
`python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i rl-trained-agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000`

---
### [PyBullet-Gym](https://github.com/benelot/pybullet-gym)
`pip install pybullet`<br>
`pip install pybullet-gym`<br>

`git clone https://github.com/rkuo2000/pybullet-gym`<br>
`cd pybullet-gym`<br>

**Humanoid with pretrained weights:** 
`ls pybulletgym/examples/roboschool-weights` <br>
`cp pybulletgym/examples/roboschool-weights/enjoy_TF_HumanoidPyBulletEnv_v0_2017may.py .`<br>
`python enjoy_TF_HumanoidPyBulletEnv_v0_2017may.py`<br>

**Humanoid sample codes:**<br>
* humanoid_a2c.py
* humanoid_ddpg.py
* humanoid_ppo.py
* humanoidrun_a2c.py

**Inverted Pendulum sample codes:**<br>
* invertedpendulum_a2c.py
* invertedpendulum_ddpg.py
* invertedpendulum_sac.py
* invertedPendulum_td3.py

---
## Gaming RL
### Othello Zero
`git clone https://github.com/suragnair/alpha-zero-general`<br>
`cd alpha-zero-general`<br>
`pip install coloredlogs`<br>

Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn) took about 3 days on an NVIDIA Tesla K80. <br>
To start training othello:<br>
`python main.py`<br>

To play othello (using `pretrained_models/othello/pytorch/.`)<br>
`python pit.py`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AlphaZero_othello.png?raw=true)

---
### Chess Zero
**Code:** [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)<br> 

<img width="50%" height="50%" src="https://user-images.githubusercontent.com/4205182/34057277-e9c99118-e19b-11e7-91ee-dd717f7efe9d.PNG">

---
### Texas Hold'em Poker
**Code:** [Pluribus Poker AI](https://github.com/kanagle2312/pluribus-poker-AI)
```
from pluribus import utils
from pluribus.ai.dummy import RandomPlayer
from pluribus.poker.table import PokerTable
from pluribus.poker.engine import PokerEngine
from pluribus.poker.pot import Pot

# Seed so things are deterministic.
utils.random.seed(42)

# Some settings for the amount of chips.
initial_chips_amount = 10000
small_blind_amount = 50
big_blind_amount = 100

# Create the pot.
pot = Pot()
# Instanciate six players that will make random moves, make sure 
# they can reference the pot so they can add chips to it.
players = [
    RandomPlayer(
        name=f'player {player_i}',
        initial_chips=initial_chips_amount,
        pot=pot)
    for player_i in range(6)
]
# Create the table with the players on it.
table = PokerTable(players=players, pot=pot)
# Create the engine that will manage the poker game lifecycle.
engine = PokerEngine(
    table=table,
    small_blind=small_blind_amount,
    big_blind=big_blind_amount)
# Play a round of Texas Hold'em Poker!
engine.play_one_round()
```
![](https://github.com/fedden/poker_ai/blob/develop/assets/visualisation.png?raw=truehttps://github.com/fedden/poker_ai/blob/develop/assets/visualisation.png?raw=true)

---
### [Unity ML-agents Toolkit](https://unity.com/products/machine-learning-agents)
**Code:** [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)<br>

![](https://unity.com/sites/default/files/styles/16_9_s_scale_width/public/2020-05/Complex-AI-environments_0.jpg)

---
## Robot Navigation

### RoboCar DQN in Arduino
**Blog:** [Deep Reinforcement Learning on ESP32](https://www.hackster.io/aslamahrahiman/deep-reinforcement-learning-on-esp32-843928)<br>
**Code:** [Policy-Gradient-Network-Arduino](https://github.com/aslamahrahman/Policy-Gradient-Network-Arduino)<br>
<iframe width="482" height="271" src="https://www.youtube.com/embed/d7NcoepWlyU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Autonomous Indoor Robot Navigation
**Code:** [](https://github.com/RoblabWh/RobLearn)
<iframe width="742" height="417" src="https://www.youtube.com/embed/KyA2uTIQfxw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://d3i71xaburhd42.cloudfront.net/8a47843c2e664e5e7e218e2d891726d023619403/3-Figure4-1.png)

---
### FAMLE (Fast Adaption by Meta-Learning Embeddings)
**Paper:** [Fast Online Adaptation in Robotics through Meta-Learning Embeddings of Simulated Priors](https://arxiv.org/abs/2003.04663)<br>
<iframe width="687" height="386" src="https://www.youtube.com/embed/QIY1Sm7wHhE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Habitat 2.0
**Code:** [facebookresearch/habitat-sim](https://github.com/facebookresearch/habitat-sim)<br>
<video controls>
  <source src="https://user-images.githubusercontent.com/2941091/126080914-36dc8045-01d4-4a68-8c2e-74d0bca1b9b8.mp4" type="video/mp4">
</video>

---
## Robot Gym
### Gym-Gazebo
**Code:** [erlerobot/gym-gazebo](https://github.com/erlerobot/gym-gazebo)<br>
<table>
<tr>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboCircuit2TurtlebotLidar-v0.png"></td>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/cartpole.jpg"></td>
<td><img src="https://github.com/erlerobot/gym-gazebo/raw/master/imgs/GazeboModularScara3DOF-v3.png"></td>
</tr>
</table>

---
### PyBullet-Gym 
[Creating OpenAI Gym Environments with PyBullet (Part 1)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24)<br>
[Creating OpenAI Gym Environments with PyBullet (Part 2)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e)<br>
![](https://media0.giphy.com/media/VI3OuvQShK3gzENiVz/giphy.gif?cid=790b761131bda06b74fcd9bb06c6a43939cf446edf403a68&rid=giphy.gif&ct=g)

---
### DART
**Code:** [dartsim/dart](https://github.com/dartsim/dart)<br>
OpenAI Gym with DART support: gym-dart (dartpy based), DartEnv (pydart2 based, deprecated)<br>
<iframe width="580" height="435" src="https://www.youtube.com/embed/aqAk701ylIk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


