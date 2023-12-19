---
layout: post
title: Reinforcement Learning
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Policy Gradient, Taxonomy of RL Algorithms, OpenAI Gym, PyBullet,
AI in Games, Multi-Agent RL, Imitation Learning , Meta Learning, RL-Stock, Social Tranmission.

---
## Introduction of Reinforcement Learning
![](https://i.stack.imgur.com/eoeSq.png)
<p><img width="50%" height="50%" src="https://www.tensorflow.org/agents/tutorials/images/rl_overview.png"></p>

---
### What is Reinforcement Learning ?
[概述增強式學習 (Reinforcement Learning, RL) (一) ](https://www.youtube.com/watch?v=XWukX-ayIrs)<br>
<table>
<tr>
<td><img src="https://miro.medium.com/max/2000/1*JuRvWsTyaRWZaYVirSsWiA.png"></td>
<td><img src="https://miro.medium.com/max/2000/1*GMGAfQeLvxJnTRQOEuTMDw.png"></td>
</tr>
</table>

---
### Policy Gradient
**Blog:** [DRL Lecture 1: Policy Gradient (Review)](https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FHkH2hSKuS)<br>
<iframe width="560" height="315" src="https://www.youtube.com/embed/US8DFaAZcp4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Actor-Critic

<iframe width="560" height="315" src="https://www.youtube.com/embed/kk6DqWreLeU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Reward Shaping

<iframe width="560" height="315" src="https://www.youtube.com/embed/73YyF1gmIus" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Algorithms

### Taxonomy of RL Algorithms
**Blog:** [Kinds of RL Alogrithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)<br>

* **Value-based methods** : Deep Q Learning
  - Where we learn a value function that will map each state action pair to a value.
* **Policy-based methods** : Reinforce with Policy Gradients
  - where we directly optimize the policy without using a value function
  - This is useful when the action space is continuous (連續) or stochastic (隨機)
  - use total rewards of the episode 
* **Hybrid methods** : Actor-Critic
  - a Critic that measures how good the action taken is (value-based)
  - an Actor that controls how our agent behaves (policy-based)
* **Model-based methods** : Partially-Observable Markov Decision Process (POMDP)
  - State-transition models
  - Observation-transition models

---
### List of RL Algorithms
1. **Q-Learning** 
  - [An Analysis of Temporal-Difference Learning with Function Approximation](http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)
  - [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
  - [A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation](https://arxiv.org/abs/1806.02450)
2. **A2C** (Actor-Critic Algorithms): [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
3. **DQN** (Deep Q-Networks): [1312.5602](https://arxiv.org/abs/1312.5602)
4. **TRPO** (Trust Region Policy Optimizaton): [1502.05477](https://arxiv.org/abs/1502.05477)
5. **DDPG** (Deep Deterministic Policy Gradient): [1509.02971](https://arxiv.org/abs/1509.02971)
6. **DDQN** (Deep Reinforcement Learning with Double Q-learning): [1509.06461](https://arxiv.org/abs/1509.06461)
7. **DD-Qnet** (Double Dueling Q Net): [1511.06581](https://arxiv.org/abs/1511.06581)
8. **A3C** (Asynchronous Advantage Actor-Critic): [1602.01783](https://arxiv.org/abs/1602.01783)
9. **ICM** (Intrinsic Curiosity Module): [1705.05363](https://arxiv.org/abs/1705.05363)
10. **I2A** (Imagination-Augmented Agents): [1707.06203](https://arxiv.org/abs/1707.06203)
11. **PPO** (Proximal Policy Optimization): [1707.06347](https://arxiv.org/abs/1707.06347)
12. **C51** (Categorical 51-Atom DQN): [1707.06887](https://arxiv.org/abs/1707.06887)
13. **HER** (Hindsight Experience Replay): [1707.01495](https://arxiv.org/abs/1707.01495)
14. **MBMF** (Model-Based RL with Model-Free Fine-Tuning): [1708.02596](https://arxiv.org/abs/1708.02596)
15. **Rainbow** (Combining Improvements in Deep Reinforcement Learning): [1710.02298](https://arxiv.org/abs/1710.02298)
16. **QR-DQN** (Quantile Regression DQN): [1710.10044](https://arxiv.org/abs/1710.10044)
17. **AlphaZero** : [1712.01815](https://arxiv.org/abs/1712.01815)
18. **SAC** (Soft Actor-Critic): [1801.01290](https://arxiv.org/abs/1801.01290)
19. **TD3** (Twin Delayed DDPG): [1802.09477](https://arxiv.org/abs/1802.09477)
20. **MBVE** (Model-Based Value Expansion): [1803.00101](https://arxiv.org/abs/1803.00101)
21. **World Models**: [1803.10122](https://arxiv.org/abs/1803.10122)
22. **IQN** (Implicit Quantile Networks for Distributional Reinforcement Learning): [1806.06923](https://arxiv.org/abs/1806.06923)
23. **SHER** (Soft Hindsight Experience Replay): [2002.02089](https://arxiv.org/abs/2002.02089)
24. **LAC** (Actor-Critic with Stability Guarantee): [2004.14288](https://arxiv.org/abs/2004.14288)
25. **AGAC** (Adversarially Guided Actor-Critic): [2102.04376](https://arxiv.org/abs/2102.04376)
26. **TATD3** (Twin actor twin delayed deep deterministic policy gradient learning for batch process control): [2102.13012](https://arxiv.org/abs/2102.13012)
27. **SACHER** (Soft Actor-Critic with Hindsight Experience Replay Approach): [2106.01016](https://arxiv.org/abs/2106.01016)
28. **MHER** (Model-based Hindsight Experience Replay): [2107.00306](https://arxiv.org/abs/2107.00306)

---
## Open Environments

### [Best Benchmarks for Reinforcement Learning: The Ultimate List](https://neptune.ai/blog/best-benchmarks-for-reinforcement-learning)
* **AI Habitat** – Virtual embodiment; Photorealistic & efficient 3D simulator;
* **Behaviour Suite** – Test core RL capabilities; Fundamental research; Evaluate generalization;
* **DeepMind Control Suite** – Continuous control; Physics-based simulation; Creating environments;
* **DeepMind Lab** – 3D navigation; Puzzle-solving;
* **DeepMind Memory Task Suite** – Require memory; Evaluate generalization;
* **DeepMind Psychlab** – Require memory; Evaluate generalization;
* **Google Research Football** – Multi-task; Single-/Multi-agent; Creating environments;
* **Meta-World** – Meta-RL; Multi-task;
* **MineRL** – Imitation learning; Offline RL; 3D navigation; Puzzle-solving;
* **Multiagent emergence environments** – Multi-agent; Creating environments; Emergence behavior;
* **OpenAI Gym** – Continuous control; Physics-based simulation; Classic video games; RAM state as observations;
* **OpenAI Gym Retro** – Classic video games; RAM state as observations;
* **OpenSpiel** – Classic board games; Search and planning; Single-/Multi-agent;
* **Procgen Benchmark** – Evaluate generalization; Procedurally-generated;
* **PyBullet Gymperium** – Continuous control; Physics-based simulation; MuJoCo unpaid alternative;
* **Real-World Reinforcement Learning** – Continuous control; Physics-based simulation; Adversarial examples;
* **RLCard** – Classic card games; Search and planning; Single-/Multi-agent;
* **RL Unplugged** – Offline RL; Imitation learning; Datasets for the common benchmarks;
* **Screeps** – Compete with others; Sandbox; MMO for programmers;
* **Serpent.AI – Game Agent Framework** – Turn ANY video game into the RL env;
* **StarCraft II Learning Environment** – Rich action and observation spaces; Multi-agent; Multi-task;
* **The Unity Machine Learning Agents Toolkit (ML-Agents)** – Create environments; Curriculum learning; Single-/Multi-agent; Imitation learning;
* **WordCraft** -Test core capabilities; Commonsense knowledge;

---
### [OpenAI Gym](https://github.com/openai/gym)
[Reinforcement Learning 健身房](https://rkuo2000.github.io/AI-course/lecture/2023/12/14/Reinforcement-Learning.html)<br>

---
### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
RL Algorithms in PyTorch : **A2C, DDPG, DQN, HER, PPO, SAC, TD3**.<br>
**QR-DQN, TQC, Maskable PPO** are in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)<br>
**[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)**<br>
`pip install stable-baselines3`<br>
For Ubuntu: `pip install gym[atari]`<br>
For Win10 : `pip install --no-index -f ttps://github.com/Kojoley/atari-py/releases atari-py`<br>
Downloading and installing visual studio 2015-2019 x86 and x64 from [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)<br>

---
### Q Learning
**Blog:** [A Hands-On Introduction to Deep Q-Learning using OpenAI Gym in Python](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)<br>
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM-670x440.png)

---
**Blog:** [An introduction to Deep Q-Learning: let’s play Doom](https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/)<br>
<img width="50%" height="50%" src="https://cdn-media-1.freecodecamp.org/images/1*Q4XjhLC0IAOznnk5613PsQ.gif">
![](https://cdn-media-1.freecodecamp.org/images/1*js8r4Aq2ZZoiLK0mMp_ocg.png)
![](https://cdn-media-1.freecodecamp.org/images/1*LglEewHrVsuEGpBun8_KTg.png)

---
### DQN
**Paper:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)<br>
![](https://www.researchgate.net/publication/338248378/figure/fig3/AS:842005408141312@1577761141285/This-is-DQN-framework-for-DRL-DNN-outputs-the-Q-values-corresponding-to-all-actions.jpg)

**[PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)**<br>
**Gym Cartpole**: [dqn.py](https://github.com/rkuo2000/RL-Gym/blob/main/cartpole/dqn.py)<br>
![](https://pytorch.org/tutorials/_images/cartpole.gif)

---
### DQN RoboCar
**Blog:** [Deep Reinforcement Learning on ESP32](https://www.hackster.io/aslamahrahiman/deep-reinforcement-learning-on-esp32-843928)<br>
**Code:** [Policy-Gradient-Network-Arduino](https://github.com/aslamahrahman/Policy-Gradient-Network-Arduino)<br>
<iframe width="482" height="271" src="https://www.youtube.com/embed/d7NcoepWlyU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### DQN for MPPT control
**Paper:** [A Deep Reinforcement Learning-Based MPPT Control for PV Systems under Partial Shading Condition](https://www.researchgate.net/publication/341720872_A_Deep_Reinforcement_Learning-Based_MPPT_Control_for_PV_Systems_under_Partial_Shading_Condition)<br>

![](https://www.researchgate.net/publication/341720872/figure/fig1/AS:896345892196354@1590716922926/A-diagram-of-the-deep-Q-network-DQN-algorithm.ppm)

---
### DDQN 
**Paper:** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)<br>
**Tutorial:** [Train a Mario-Playing RL Agent](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)<br>
**Code:** [MadMario](https://github.com/YuansongFeng/MadMario)<br>
![](https://pytorch.org/tutorials/_images/mario.gif)

---
### Duel DQN
**Paper:** [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)<br>
![](https://theaisummer.com/static/b0f4c8c3f3a5158b5899aa52575eaea0/95a07/DDQN.jpg)

### Double Duel Q Net
**Code:** [mattbui/dd_qnet](https://github.com/mattbui/dd_qnet)<br>

![](https://github.com/mattbui/dd_qnet/blob/master/screenshots/running.gif?raw=true)

---
### A2C
**Paper:** [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)<br>
![](https://miro.medium.com/max/1400/0*g0jtX8lIdplzJ8oo.png)
  - The **“Critic”** estimates the **value** function. This could be the action-value (the Q value) or state-value (the V value).
  - The **“Actor”** updates the **policy** distribution in the direction suggested by the Critic (such as with policy gradients).
  - A2C: Instead of having the critic to learn the Q values, we make him learn the Advantage values.
  
---
### A3C
**Paper:** [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)<br>
**Blog:** [The idea behind Actor-Critics and how A2C and A3C improve them](https://towardsdatascience.com/the-idea-behind-actor-critics-and-how-a2c-and-a3c-improve-them-6dd7dfd0acb8)<br>
**Blog:** [李宏毅_ATDL_Lecture_23](https://hackmd.io/@shaoeChen/SkRbRFBvH#)<br>

![](https://miro.medium.com/max/770/0*OWRT4bcbfcansOwA.jpg)

---
### DDPG
**Paper:** [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)<br>
**Blog:** [Deep Deterministic Policy Gradients Explained](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b)<br>
**Blog:** [人工智慧-Deep Deterministic Policy Gradient (DDPG)](https://www.wpgdadatong.com/tw/blog/detail?BID=B2541)<br>
DDPG是在A2C中加入**經驗回放記憶體**，在訓練的過程中會持續的收集經驗，並且會設定一個buffer size，這個值代表要收集多少筆經驗，每當經驗庫滿了之後，每多一個經驗則最先收集到的經驗就會被丟棄，因此可以讓經驗庫一值保持滿的狀態並且避免無限制的收集資料造成電腦記憶體塞滿。<br>
學習的時候則是從這個經驗庫中隨機抽取成群(batch)經驗來訓練DDPG網路，周而復始的不斷進行學習最終網路就能達到收斂狀態，請參考下圖DDPG演算架構圖。<br>
![](https://edit.wpgdadawant.com/uploads/news_file/blog/2020/2976/tinymce/2020-12-27_18h15_54.png)
**Code:** [End to end motion planner using Deep Deterministic Policy Gradient (DDPG) in gazebo](https://github.com/m5823779/motion-planner-reinforcement-learning)<br>
<p><img src="https://github.com/m5823779/MotionPlannerUsingDDPG/raw/master/demo/demo.gif" width="50%" height="50%"></p>

---
### [Intrinsic Curiosity Module (ICM)](https://pathak22.github.io/noreward-rl/)
**Paper:** [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)<br>
**Code:** [pathak22/noreward-rl](https://github.com/pathak22/noreward-rl)<br>
<iframe width="800" height="450" src="https://www.youtube.com/embed/J3FHOyhUn3A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### PPO
**Paper:** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)<br>
**On-policy vs Off-policy**<br>
On-Policy 方式是指用於學習的agent與觀察環境的agent是同一個，所以引數θ始終保持一致。**(邊做邊學)**<br>
Off-Policy方式是指用於學習的agent與用於觀察環境的agent不是同一個，他們的引數θ可能不一樣。**(在旁邊透過看別人做來學習)**<br>
比如下圍棋，On-Policy方式是agent親歷親為，而Off-Policy是一個agent看其他的agent下棋，然後去學習人家的東西。<br>

---
### TRPO
**Paper:** [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)<br>
**Blog:** [Trust Region Policy Optimization講解](https://www.twblogs.net/a/5d5ead97bd9eee541c32568c)<br>
TRPO 算法 (Trust Region Policy Optimization)和PPO 算法 (Proximal Policy Optimization)都屬於MM(Minorize-Maximizatio)算法。<br>

---
### HER 
**Paper:** [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)<br>
**Code:** [OpenAI HER](https://github.com/openai/baselines/tree/master/baselines/her)<br>
<iframe width="640" height="360" src="https://www.youtube.com/embed/Dz_HuzgMxzo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### MBMF
**Paper:** [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596)<br>
<iframe width="800" height="480" src="https://www.youtube.com/embed/G7lXiuEC8x0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### SAC
**Paper:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)<br>
![](https://miro.medium.com/max/974/0*NgZ_bq_nUOq73jK_.png)

---
### TD3
**Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)<br>
**Code:** [sfujim/TD3](https://github.com/sfujim/TD3)<br>
TD3 with RAMDP<br>
![](https://www.researchgate.net/publication/338605159/figure/fig2/AS:847596578947080@1579094180953/Structure-of-TD3-Twin-Delayed-Deep-Deterministic-Policy-Gradient-with-RAMDP.jpg)

---
### POMDP (Partially-Observable Markov Decision Process)
**Paper:** [Planning and acting in partially observable stochastic domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)<br>
![](https://ars.els-cdn.com/content/image/1-s2.0-S2352154618301670-gr1_lrg.jpg)

---
### SHER
**Paper:** [Soft Hindsight Experience Replay](https://arxiv.org/abs/2002.02089)
![](https://d3i71xaburhd42.cloudfront.net/6253a0f146a36663e908509e14648f8e2a5ab581/5-Figure3-1.png)

---
### Exercises: [RL-gym](https://github.com/rkuo2000/RL-gym)
Downloading and installing visual studio 2015-2019 x86 and x64 from [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)<br>

```
sudo apt-get install ffmpeg freeglut3-dev xvfb
pip install tensorflow
pip install pyglet==1.5.27
pip install stable_baselines3[extra]
pip install gym[all]
pip install autorom[accept-rom-license]
git clone https://github.com/rkuo2000/RL-gym
cd RL-gym
cd cartpole
```

---
#### ~/RL-gym/cartpole
`python3 random_action.py`<br>
`python3 q_learning.py`<br>
`python3 dqn.py`<br>
![](https://github.com/rkuo2000/RL-gym/blob/main/assets/CartPole.gif?raw=true)

---
#### ~/RL-gym/sb3/
alogrithm = A2C, output = xxx.zip<br>
`python3 train.py LunarLander-v2 640000`<br>
`python3 enjoy.py LunarLander-v2`<br>
`python3 enjoy_gif.py LunarLander-v2`<br>
![](https://github.com/rkuo2000/RL-gym/blob/main/assets/LunarLander.gif?raw=true)

---
### Atari
env_name listed in Env_Name.txt<br>
you can train on [Kaggle](https://www.kaggle.com/code/rkuo2000/rl-sb3-atari), then download .zip to play on PC<br>

`python3 train_atari.py Pong-v0 1000000`<br>
`python3 enjoy_atari.py Pong-v0`<br>
`python3 enjoy_atari_gif.py Pong-v0`<br>

<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/main/images/gym_CrazyClimber_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/main/images/gym_DoubleDunk_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/main/images/gym_IceHockey_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/main/images/gym_boxing_ppo.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/main/images/gym_qbert_ppo.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/main/images/gym_tennis_a2c.gif?raw=true"></td>
</tr>
</table>

---
### [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
**PyBulletEnv**<br>
`python enjoy.py --algo a2c --env AntBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/RL_SB3_Zoo_AntBulletEnv.gif?raw=true)
`python enjoy.py --algo a2c --env HalfCheetahBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/RL_SB3_Zoo_HalfCheetahBulletEnv.gif?raw=true)
`python enjoy.py --algo a2c --env HopperBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/RL_SB3_Zoo_HopperBulletEnv.gif?raw=true)
`python enjoy.py --algo a2c --env Walker2DBulletEnv-v0 --folder rl-trained-agents/ -n 5000`<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/RL_SB3_Zoo_Walker2DBulletEnv.gif?raw=true)

---
### [Pybullet](https://pybullet.org) - Bullet Real-Time Physics Simulation
<iframe width="474" height="267" src="https://www.youtube.com/embed/-MfUXSAehnw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="474" height="267" src="https://www.youtube.com/embed/EFKqNKO3P60" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="474" height="267" src="https://www.youtube.com/embed/lKYh6uuCwRY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="474" height="267" src="https://www.youtube.com/embed/_QPMCDdFC3E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [PyBullet-Gym](https://github.com/benelot/pybullet-gym)
**Code:** [rkuo2000/pybullet-gym](https://github.com/rkuo2000/pybullet-gym)<br>
* installation
```
pip install gym
pip install pybullet
pip install stable-baselines3
git clone https://github.com/rkuo2000/pybullet-gym
export PYTHONPATH=$PATH:/home/yourname/pybullet-gym
```

#### gym
**Env names:** *Ant, Atlas, HalfCheetah, Hopper, Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder, InvertedPendulum, InvertedDoublePendulum, InvertedPendulumSwingup, Reacher, Walker2D*<br>

**Blog:** <br>
[Creating OpenAI Gym Environments with PyBullet (Part 1)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24)<br>
[Creating OpenAI Gym Environments with PyBullet (Part 2)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e)<br>
![](https://media0.giphy.com/media/VI3OuvQShK3gzENiVz/giphy.gif?cid=790b761131bda06b74fcd9bb06c6a43939cf446edf403a68&rid=giphy.gif&ct=g)

---
### [OpenAI Gym Environments for Donkey Car](https://github.com/tawnkramer/gym-donkeycar)
* [Documentation](https://gym-donkeycar.readthedocs.io/en/latest/)
* Download [simulator binaries](https://github.com/tawnkramer/gym-donkeycar/releases)
* [Donkey Simulator User Guide](https://docs.donkeycar.com/guide/simulator/)
![](https://docs.donkeycar.com/assets/sim_screen_shot.png)

---
### [Google Dopamine](https://github.com/google/dopamine)
Dopamine is a research framework for fast prototyping of reinforcement learning algorithms.<br>
*Dopamine supports the following agents, implemented with [jax](https://github.com/google/jax): DQN, C51, Rainbow, IQN, SAC.* <br>

---
### [ViZDoom](https://github.com/mwydmuch/ViZDoom)
![](https://camo.githubusercontent.com/a7d9d95fc80903bcb476c2bbdeac3fa7623953c05401db79101c2468b0d90ad9/687474703a2f2f7777772e63732e7075742e706f7a6e616e2e706c2f6d6b656d706b612f6d6973632f76697a646f6f6d5f676966732f76697a646f6f6d5f636f727269646f725f7365676d656e746174696f6e2e676966)
`sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-ev libopenal-dev zlib1g-dev timidity tar nasm`<br>
`pip install vizdoom`<br>

---
## AI in Games
**Paper:** [AI in Games: Techniques, Challenges and Opportunities](https://arxiv.org/abs/2111.07631)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/AI_in_Games_survey.png?raw=true)

---
### AlphaGo
2016 年 3 月，AlphaGo 這一台 AI 思維的機器挑戰世界圍棋冠軍李世石（Lee Sedol）。比賽結果以 4 比 1 的分數，AlphaGo 壓倒性的擊倒人類世界最會下圍棋的男人。<br>
**Paper:** [Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)<br>
**Paper:** [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)<br>
**Blog:** [Day 27 / DL x RL / 令世界驚艷的 AlphaGo](https://ithelp.ithome.com.tw/articles/10252358)<br>

AlphaGo model 主要包含三個元件：<br>
* **Policy network**：根據盤面預測下一個落點的機率。
* **Value network**：根據盤面預測最終獲勝的機率，類似預測盤面對兩方的優劣。
* **Monte Carlo tree search (MCTS)**：類似在腦中計算後面幾步棋，根據幾步之後的結果估計現在各個落點的優劣。

![](https://i.imgur.com/xdc52cv.png)

* **Policy Networks**: 給定 input state，會 output 每個 action 的機率。<br>
AlphaGo 中包含三種 policy network：<br>
* [Supervised learning (SL) policy network](https://chart.googleapis.com/chart?cht=tx&chl=p_%7B%5Csigma%7D)
* [Reinforcement learning (RL) policy network](https://chart.googleapis.com/chart?cht=tx&chl=p_%7B%5Crho%7D)
* [Rollout policy network](https://chart.googleapis.com/chart?cht=tx&chl=p_%7B%5Cpi%7D)

* **Value Network**: 預測勝率，Input 是 state，output 是勝率值。<br>
這個 network 也可以用 supervised learning 訓練，data 是歷史對局中的 state-outcome pair，loss 是 mean squared error (MSE)。

* **Monte Carlo Tree Search (MCTS)**: 結合這些 network 做 planning，決定遊戲進行時的下一步。<br>
![](https://i.imgur.com/aXdpcz6.png)
1. Selection：從 root 開始，藉由 policy network 預測下一步落點的機率，來選擇要繼續往下面哪一步計算。選擇中還要考量每個 state-action pair 出現過的次數，盡量避免重複走同一條路，以平衡 exploration 和 exploitation。重複這個步驟直到樹的深度達到 max depth L。
2. Expansion：到達 max depth 後的 leaf node sL，我們想要估計這個 node 的勝算。首先從 sL 往下 expand 一層。
3. Evaluation：每個 sL 的 child node 會開始 rollout，也就是跟著 rollout policy network 預測的 action 開始往下走一陣子，取得 outcome z。最後 child node 的勝算會是 value network 對這個 node 預測的勝率和 z 的結合。
4. Backup：sL 會根據每個 child node 的勝率更新自己的勝率，並往回 backup，讓從 root 到 sL 的每個 node 都更新勝率。

---
### AlphaZero
2017 年 10 月，AlphaGo Zero 以 100 比 0 打敗 AlphaGo。<br>
**Blog:** [AlphaGo beat the world’s best Go player. He helped engineer the program that whipped AlphaGo.](https://www.technologyreview.com/innovator/julian-schrittwieser/)<br>
**Paper:** [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)<br>
![](https://s.newtalk.tw/album/news/180/5c0f5a4489883.png)
AlphaGo 用兩個類神經網路，分別估計策略函數和價值函數。AlphaZero 用一個多輸出的類神經網路<br>
AlphaZero 的策略函數訓練方式是直接減少類神經網路與MCTS搜尋出來的πₜ之間的差距，這就是在做regression，而 AlpahGo 原本用的方式是RL演算法做 Policy gradient。(πₜ：當時MCTS後的動作機率值)<br>
**Blog:** [優拓 Paper Note ep.13: AlphaGo Zero](https://blog.yoctol.com/%E5%84%AA%E6%8B%93-paper-note-ep-13-alphago-zero-efa8d4dc538c)<br>
**Blog:** [Monte Carlo Tree Search (MCTS) in AlphaGo Zero](https://jonathan-hui.medium.com/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a)<br>
**Blog:** [The 3 Tricks That Made AlphaGo Zero Work](https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef)<br>
1. MTCS with intelligent lookahead search
2. Two-headed Neural Network Architecture
3. Using residual neural network architecture 
<table>
<tr>
<td><img src="https://hackernoon.com/hn-images/1*hBzorPuADtitET2SZaLN2A.png"></td>
<td><img src="https://hackernoon.com/hn-images/1*96DnPFNDD8YyN-GK737bBQ.png"></td>
<td><img src="https://hackernoon.com/hn-images/1*aJCekYFA3jG0NDBmBEYYPA.png"></td>
</tr>
</table>

![](https://github.com/rkuo2000/AI-course/blob/main/images/AlphaGo_version_comparison.png?raw=true)

---
### AlphaZero with a Learned Model
**Paper:** [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)<br>
RL can be divided into Model-Based RL (MBRL) and Model-Free RL (MFRL). Model-based RL uses an environment model for planning, whereas model-free RL learns the optimal policy directly from interactions. Model-based RL has achieved superhuman level of performance in Chess, Go, and Shogi, where the model is given and the game requires sophisticated lookahead. However, model-free RL performs better in environments with high-dimensional observations where the model must be learned.
![](https://www.endtoend.ai/assets/blog/rl-weekly/36/muzero.png)

---
### Minigo
**Code:** [tensorflow minigo](https://github.com/tensorflow/minigo)<br>

---
### ELF OpenGo
**Code:** [https://github.com/pytorch/ELF](https://github.com/pytorch/ELF)<br>
**Blog:** [A new ELF OpenGo bot and analysis of historical Go games](https://ai.facebook.com/blog/open-sourcing-new-elf-opengo-bot-and-go-research/)<br>

---
### Chess Zero
**Code:** [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)<br> 
<img width="50%" height="50%" src="https://user-images.githubusercontent.com/4205182/34057277-e9c99118-e19b-11e7-91ee-dd717f7efe9d.PNG">

---
### AlphaStar
**Blog:** [AlphaStar: Mastering the real-time strategy game StarCraft II](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)<br>
**Blog:** [AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning)<br>
**Code:** [PySC2 - StarCraft II Learning Environment](https://github.com/deepmind/pysc2)<br>
![](https://lh3.googleusercontent.com/ckm-3GlBQJ4zbNzfiW97yPqj5PVC0qIbRg42FL35EbDkhWoCNxyNZMMJN-f6VZmLMRbyBk2PArLQ-jDxlHbsE3_YaDUmcxUvMf8M=w1440-rw-v1)

---
### [OpenAI Five](https://openai.com/blog/openai-five/) at Dota2
<iframe width="964" height="542" src="https://www.youtube.com/embed/UZHTNBMAfAA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [DeepMind FTW](https://deepmind.com/blog/article/capture-the-flag-science)
![](https://lh3.googleusercontent.com/CFlAYmP49qitU-SOP_PaKtV1kOrlpNnvo4oEDFhyxelrVwKyAbkdXwUuDFRTmiRSQle4955mmOAB4jrrIrWzIXDt8hOajZGtJzNaDRw=w1440-rw-v1)
![](https://lh3.googleusercontent.com/RterJzRGidwT9R_Dqeu5LY5MZPjjYRc-MQdQyca7gACnA7w0bjCu_hIcoXLC4xV5zebvdZnN7ocZkemGnF4K7_p5SMZCLRWbNq1IDQ=w1440-rw-v1)

---
### Texas Hold'em Poker
**Code:** [fedden/poker_ai](https://github.com/fedden/poker_ai)<br>
**Code:** [Pluribus Poker AI](https://github.com/kanagle2312/pluribus-poker-AI) + [poker table](https://codepen.io/Rovak/pen/ExYeQar)<br>
**Blog:** [Artificial Intelligence Masters The Game of Poker – What Does That Mean For Humans?](https://www.forbes.com/sites/bernardmarr/2019/09/13/artificial-intelligence-masters-the-game-of-poker--what-does-that-mean-for-humans/?sh=dcaa18f5f9ea)<br>
<iframe width="832" height="468" src="https://www.youtube.com/embed/wInvN096he8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Suphx
**Paper:** [2003.13590](https://arxiv.org/abs/2003.13590)<br>
**Blog:** [微软超级麻将AI Suphx论文发布，研发团队深度揭秘技术细节](https://www.msra.cn/zh-cn/news/features/mahjong-ai-suphx-paper)<br>
![](https://d3i71xaburhd42.cloudfront.net/b30c663690c3a096c7d92f307ba7d17bdfd48553/6-Figure2-1.png)

---
### DouZero
**Paper:** [2106.06135](https://arxiv.org/abs/2106.06135)<br>
**Code:** [kwai/DouZero](https://github.com/kwai/DouZero)<br>
**Demo:** [douzero.org/](https://douzero.org/)<br>
![](https://camo.githubusercontent.com/45f00ff00a26f0df47ebbab3a993ccbf83e4715d7a0f1132665c8c045ebd52c2/68747470733a2f2f646f757a65726f2e6f72672f7075626c69632f64656d6f2e676966)

---
### JueWu
**Paper:** [Supervised Learning Achieves Human-Level Performance in MOBA Games: A Case Study of Honor of Kings](https://arxiv.org/abs/2011.12582)<br>
**Blog:** [Tencent AI ‘Juewu’ Beats Top MOBA Gamers](https://medium.com/syncedreview/tencent-ai-juewu-beats-top-moba-gamers-acdb44133d24)<br>
![](https://miro.medium.com/max/2000/1*bDp5a8gKiHynxiK-TQPJdQ.png)
![](https://github.com/rkuo2000/AI-course/blob/main/images/JueWu_structure.png?raw=true)

---
### StarCraft Commander 
**[启元世界](http://www.inspirai.com/research/scc?language=en)**<br>
**Paper:** [SCC: an efficient deep reinforcement learning agent mastering the game of StarCraft II](https://arxiv.org/abs/2012.13169)<br>

---
### Hanabi ToM
**Paper:** [Theory of Mind for Deep Reinforcement Learning in Hanabi](https://arxiv.org/abs/2101.09328)<br>
**Code:** [mwalton/ToM-hanabi-neurips19](https://github.com/mwalton/ToM-hanabi-neurips19)<br>
Hanabi (from Japanese 花火, fireworks) is a cooperative card game created by French game designer Antoine Bauza and published in 2010.
<iframe width="640" height="360" src="https://www.youtube.com/embed/LQ8iwNjBW_s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## MARL (Multi-Agent Reinforcement Learning)

### Neural MMO
**Paper:** [The Neural MMO Platform for Massively Multiagent Research](https://arxiv.org/abs/2110.07594)<br>
**Blog:** [User Guide](https://neuralmmo.github.io/build/html/rst/userguide.html)<br>
![](https://neuralmmo.github.io/build/html/_images/splash.png)
<iframe width="742" height="417" src="https://www.youtube.com/embed/hYYA8_wFF7Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Multi-Agent Locomotion
**Paper:** [Emergent Coordination Through Competition](https://arxiv.org/abs/1902.07151)<br>
**Code:** [Locomotion task library](https://github.com/deepmind/dm_control/tree/master/dm_control/locomotion)<br>
**Code:** [DeepMind MuJoCo Multi-Agent Soccer Environment](https://github.com/deepmind/dm_control/tree/master/dm_control/locomotion/soccer)<br>
![](https://github.com/deepmind/dm_control/blob/master/dm_control/locomotion/soccer/soccer.png?raw=true)

---
### [Unity ML-agents Toolkit](https://unity.com/products/machine-learning-agents)
**Code:** [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)<br>
![](https://unity.com/sites/default/files/styles/16_9_s_scale_width/public/2020-05/Complex-AI-environments_0.jpg)

**Blog:** [A hands-on introduction to deep reinforcement learning using Unity ML-Agents](https://medium.com/coder-one/a-hands-on-introduction-to-deep-reinforcement-learning-using-unity-ml-agents-e339dcb5b954)<br>
![](https://miro.medium.com/max/540/0*ojKXHwzo_a-rjwpz.gif)

---
### DDPG Actor-Critic Reinforcement Learning Reacher Environment
**Code:** [https://github.com/Remtasya/DDPG-Actor-Critic-Reinforcement-Learning-Reacher-Environment](https://github.com/Remtasya/DDPG-Actor-Critic-Reinforcement-Learning-Reacher-Environment)<br>
![](https://github.com/Remtasya/DDPG-Actor-Critic-Reinforcement-Learning-Reacher-Environment/raw/master/project_images/reacher%20environment.gif)

---
### Multi-Agent Mobile Manipulation
**Paper:** [Spatial Intention Maps for Multi-Agent Mobile Manipulation](https://arxiv.org/abs/2103.12710)<br>
**Code:** [jimmyyhwu/spatial-intention-maps](https://github.com/jimmyyhwu/spatial-intention-maps)<br>
![](https://user-images.githubusercontent.com/6546428/111895195-42af8700-89ce-11eb-876c-5f98f6b31c96.gif)

---
### DeepMind Cultural Transmission
**Paper** [Learning few-shot imitation as cultural transmission](https://www.nature.com/articles/s41467-023-42875-2)<br>
**Blog:** [DeepMind智慧體訓練引入GoalCycle3D](https://cdn.technews.tw/2023/12/14/learning-few-shot-imitation-as-cultural-transmission)<br>
以模仿開始，然後深度強化學習繼續最佳化甚至找到超越前者的實驗，顯示AI智慧體能觀察別的智慧體學習並模仿。<br>
這從零樣本開始，即時取得利用資訊的能力，非常接近人類積累和提煉知識的方式。<br>
![](https://img.technews.tw/wp-content/uploads/2023/12/13113144/41467_2023_42875_Fig1_HTML.jpg)

---
## Imitation Learning
**Blog:** [A brief overview of Imitation Learning](https://smartlabai.medium.com/a-brief-overview-of-imitation-learning-8a8a75c44a9c)<br>
<iframe width="742" height="417" src="https://www.youtube.com/embed/WjFdD7PDGw0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Self-Imitation Learning
directly use past good experiences to train current policy.<br>
**Paper:** [Self-Imitation Learming](https://arxiv.org/abs/1806.05635)<br>
**Code:** [junhyukoh/self-imitation-learning](https://github.com/junhyukoh/self-imitation-learning)<br>
**Blog:** [[Paper Notes 2] Self-Imitation Learning](https://medium.com/intelligentunit/paper-notes-2-self-imitation-learning-b3a0fbdee351)<br>
![](https://miro.medium.com/max/2000/1*tvoSPpq7zSNscaVIxQX5hg@2x.png)

---
### Self-Imitation Learning by Planning
**Paper:** [Self-Imitation Learning by Planning](https://arxiv.org/abs/2103.13834)<br>
<iframe width="742" height="417" src="https://www.youtube.com/embed/2rx-roLYJ5k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Surgical Robotics
**Paper:** [Open-Sourced Reinforcement Learning Environments for Surgical Robotics](https://arxiv.org/abs/1903.02090)<br>
**Code:** [RL Environments for the da Vinci Surgical System](https://github.com/ucsdarclab/dVRL)<br>

---
## Meta Learning (Learning to Learn)
**Blog:** [Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)<br>

### Meta-Learning Survey
**Paper:** [Meta-Learning in Neural Networks: A Survey](https://arxiv.org/abs/2004.05439)<br>
![](https://d3i71xaburhd42.cloudfront.net/020bb2ba5f3923858cd6882ba5c5a44ea8041ab6/6-Figure1-1.png)

---
### MAML (Model-Agnostic Meta-Learning)
**Paper:** [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)<br>
**Code:** [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl)<br>

---
### Reptile
**Paper:** [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)<br>
**Code:** [openai/supervised-reptile](https://github.com/openai/supervised-reptile)<br>

---
### MAML++
**Paper:** [How to train your MAML](https://arxiv.org/abs/1810.09502)<br>
**Code:** [AntreasAntoniou/HowToTrainYourMAMLPytorch](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch)<br>
**Blog:** [元學習——從MAML到MAML++](https://www.twblogs.net/a/60e689df1cf175147a0e2084)<br>

---
**Paper:** [First-order Meta-Learned Initialization for Faster Adaptation in Deep Reinforcement Learning](https://www.andrew.cmu.edu/user/abhijatb/assets/Deep_RL_project.pdf)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/Meta_Learning_algorithms.png?raw=true)

---
### FAMLE (Fast Adaption by Meta-Learning Embeddings)
**Paper:** [Fast Online Adaptation in Robotics through Meta-Learning Embeddings of Simulated Priors](https://arxiv.org/abs/2003.04663)<br>
<iframe width="687" height="386" src="https://www.youtube.com/embed/QIY1Sm7wHhE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://media.arxiv-vanity.com/render-output/5158097/x1.png)
![](https://media.arxiv-vanity.com/render-output/5158097/x3.png)

---
### Bootstrapped Meta-Learning
**Paper:** [Bootstrapped Meta-Learning](https://arxiv.org/abs/2109.04504)<br>
**Blog:** [DeepMind’s Bootstrapped Meta-Learning Enables Meta Learners to Teach Themselves](https://syncedreview.com/2021/09/20/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-107/)<br>

![](https://i0.wp.com/syncedreview.com/wp-content/uploads/2021/09/image-77.png?w=549&ssl=1)

---
## Unsupervised Learning

### Understanding the World Through Action
**Blog:** [Understanding the World Through Action: RL as a Foundation for Scalable Self-Supervised Learning](https://medium.com/@sergey.levine/understanding-the-world-through-action-rl-as-a-foundation-for-scalable-self-supervised-learning-636e4e243001)<br>
**Paper:** [Understanding the World Through Action](https://arxiv.org/abs/2110.12543)<br>
![](https://miro.medium.com/max/1400/1*79ztJveD6kanHz9H8VY2Lg.gif)
**Actionable Models**<br>
a self-supervised real-world robotic manipulation system trained with offline RL, performing various goal-reaching tasks. Actionable Models can also serve as general pretraining that accelerates acquisition of downstream tasks specified via conventional rewards.
![](https://miro.medium.com/max/1280/1*R7-IP07Inc7K6v4i_dQ-RQ.gif)

---
### RL-Stock 
**Kaggle:** [https://www.kaggle.com/rkuo2000/stock-lstm](https://www.kaggle.com/rkuo2000/stock-lstm)<br>
**Kaggle:** [https://kaggle.com/rkuo2000/stock-dqn](https://kaggle.com/rkuo2000/stock-dqn)<br>

---
### Stock Trading
**Blog:** [Predicting Stock Prices using Reinforcement Learning (with Python Code!)](https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/)<br>
![](https://editor.analyticsvidhya.com/uploads/770801_26xDRHI-alvDAfcPPJJGjQ.png)

**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>
**Blog:** [Automated stock trading using Deep Reinforcement Learning with Fundamental Indicators](https://medium.com/@mariko.sawada1/automated-stock-trading-with-deep-reinforcement-learning-and-financial-data-a63286ccbe2b)<br>

---
### FinRL
**Papers:** <br>
[2010.14194](https://arxiv.org/abs/2010.14194): Learning Financial Asset-Specific Trading Rules via Deep Reinforcement Learning<br>
[2011.09607](https://arxiv.org/abs/2011.09607): FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance<br>
[2101.03867](https://arxiv.org/abs/2101.03867): A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules<br>
[2106.00123](https://arxiv.org/abs/2106.00123): Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review<br>
[2111.05188](https://arxiv.org/abs/2111.05188): FinRL-Podracer: High Performance and Scalable Deep Reinforcement Learning for Quantitative Finance<br>
[2112.06753](https://arxiv.org/abs/2112.06753): FinRL-Meta: A Universe of Near-Real Market Environments for Data-Driven Deep Reinforcement Learning in Quantitative Finance<br>

**Blog:** [FinRL­-Meta: A Universe of Near Real-Market En­vironments for Data­-Driven Financial Reinforcement Learning](https://medium.datadriveninvestor.com/finrl-meta-a-universe-of-near-real-market-en-vironments-for-data-driven-financial-reinforcement-e1894e1ebfbd)<br>
![](https://miro.medium.com/max/2000/1*rOW0RH56A-chy3HKaxcjNw.png)
**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


