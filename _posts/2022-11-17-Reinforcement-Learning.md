---
layout: post
title: Reinforcement Learning
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Policy Gradient, Taxonomy of RL Algorithms, Open Environments (OpenAI Gym, DeepMind OpenSpeil, PyBullet),
AI in Games, Multi-Agent RL, Imitation Learning , Meta Learning.

---
## Introduction of Reinforcement Learning
**Blog:** [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)<br>
**Blog:** [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)<br>

<img src="https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png">
<img width="50%" height="50%" src="https://www.tensorflow.org/agents/tutorials/images/rl_overview.png">

---
### What is Reinforcement Learning ?
**Blog:** [李宏毅老師 Deep Reinforcement Learning (2017 Spring)【筆記】](https://medium.com/change-the-world-with-technology/%E6%9D%8E%E5%AE%8F%E6%AF%85%E8%80%81%E5%B8%AB-deep-reinforcement-learning-2017-spring-%E7%AD%86%E8%A8%98-3784ddb23e0)<br>

<iframe width="560" height="315" src="https://www.youtube.com/embed/XWukX-ayIrs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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

![](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

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
**[Best Benchmarks for Reinforcement Learning: The Ultimate List](https://neptune.ai/blog/best-benchmarks-for-reinforcement-learning)**<br>
* [AI Habitat](https://aihabitat.org/) – Virtual embodiment; Photorealistic & efficient 3D simulator;
* [Behaviour Suite](https://github.com/deepmind/bsuite) – Test core RL capabilities; Fundamental research; Evaluate generalization;
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) – Continuous control; Physics-based simulation; Creating environments;
* [DeepMind Lab](https://github.com/deepmind/lab) – 3D navigation; Puzzle-solving;
* [DeepMind Memory Task Suite](https://github.com/deepmind/dm_memorytasks) – Require memory; Evaluate generalization;
* [DeepMind Psychlab](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/psychlab) – Require memory; Evaluate generalization;
* [Google Research Football](https://github.com/google-research/football) – Multi-task; Single-/Multi-agent; Creating environments;
* [Meta-World](https://github.com/rlworkgroup/metaworld) – Meta-RL; Multi-task;
* [MineRL](https://minerl.readthedocs.io/en/latest/) – Imitation learning; Offline RL; 3D navigation; Puzzle-solving;
* [Multiagent emergence environments](https://github.com/openai/multi-agent-emergence-environments) – Multi-agent; Creating environments; Emergence behavior;
* [OpenAI Gym](https://gym.openai.com/) – Continuous control; Physics-based simulation; Classic video games; RAM state as observations;
* [OpenAI Gym Retro](https://github.com/openai/retro) – Classic video games; RAM state as observations;
* [OpenSpiel](https://github.com/deepmind/open_spiel) – Classic board games; Search and planning; Single-/Multi-agent;
* [Procgen Benchmark](https://github.com/openai/procgen) – Evaluate generalization; Procedurally-generated;
* [PyBullet Gymperium](https://github.com/benelot/pybullet-gym) – Continuous control; Physics-based simulation; MuJoCo unpaid alternative;
* [Real-World Reinforcement Learning](https://github.com/google-research/realworldrl_suite) – Continuous control; Physics-based simulation; Adversarial examples;
* [RLCard](https://github.com/datamllab/rlcard) – Classic card games; Search and planning; Single-/Multi-agent;
* [RL Unplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged) – Offline RL; Imitation learning; Datasets for the common benchmarks;
* [Screeps](https://screeps.com/) – Compete with others; Sandbox; MMO for programmers;
* [Serpent.AI](https://github.com/SerpentAI/SerpentAI) – Game Agent Framework – Turn ANY video game into the RL env;
* [StarCraft II Learning Environment](https://github.com/deepmind/pysc2) – Rich action and observation spaces; Multi-agent; Multi-task;
* [The Unity Machine Learning Agents Toolkit (ML-Agents)](https://github.com/Unity-Technologies/ml-agents) – Create environments; Curriculum learning; Single-/Multi-agent; Imitation learning;
* [WordCraft](https://github.com/minqi/wordcraft) -Test core capabilities; Commonsense knowledge;

---
### [OpenAI Gym](https://gym.openai.com/)
**Ref.** [Reinforcement Learning 健身房](https://pyliaorachel.github.io/blog/tech/python/2018/06/01/openai-gym-for-reinforcement-learning.html)
![](https://i.stack.imgur.com/eoeSq.png)
1. **Agent** 藉由 action 跟 environment 互動。
2. **Environment** agent 的行動範圍，根據 agent 的 action 給予不同程度的 reward。
3. **State** 在特定時間點 agent 身處的狀態。
4. **Action** agent 藉由自身 policy 進行的動作。
5. **Reward** environment 給予 agent 所做 action 的獎勵或懲罰。

---
### Q Learning
**Blog:** [A Hands-On Introduction to Deep Q-Learning using OpenAI Gym in Python](https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/)<br>
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM-670x440.png)
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/1_lTVHyzT3d26Bd_znaKaylQ-768x84.png)
*immediate reward r(s,a) plus the highest Q-value possible from the next state s’.* <br>
*Gamma here is the discount factor which controls the contribution of rewards further in the future.*<br>

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-17-at-7.15.35-PM-768x56.png)
*Adjusting the value of gamma will diminish or increase the contribution of future rewards.*<br>

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-03-26-at-7.57.30-PM1-768x64.png)
*where alpha is the learning rate or step size*<br>

The loss function here is mean squared error of the predicted Q-value and the target Q-value – Q*<br>

**Blog:** [An introduction to Deep Q-Learning: let’s play Doom](https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/)<br>

<img width="50%" height="50%" src="https://cdn-media-1.freecodecamp.org/images/1*Q4XjhLC0IAOznnk5613PsQ.gif">
![](https://cdn-media-1.freecodecamp.org/images/1*js8r4Aq2ZZoiLK0mMp_ocg.png)
![](https://cdn-media-1.freecodecamp.org/images/1*LglEewHrVsuEGpBun8_KTg.png)

---
### [Gym](https://github.com/openai/gym)
Gym is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API.<br>
`pip install gym`<br>

```
import gym 
env = gym.make('CartPole-v1')

# env is created, now we can use it: 
for episode in range(10): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
env.close()
```
CartPole環境輸出的state包括位置、加速度、杆子垂直夾角和角加速度。

---
### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
a set of reliable implementations of reinforcement learning algorithms in PyTorch.<br>
Implemented Algorithms : **A2C, DDPG, DQN, HER, PPO, SAC, TD3**.<br>
**QR-DQN, TQC, Maskable PPO** are in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)<br>

`pip install stable-baselines3`<br>
For Ubuntu: `pip install gym[atari]`<br>
For Win10 : `pip install --no-index -f ttps://github.com/Kojoley/atari-py/releases atari-py`<br>

**[SB3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)**<br>

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
```
git clone https://github.com/YuansongFeng/MadMario
cd MadMario

pip install scikit-image
pip install gym-super-mario-bros
```

Training time is around 80 hours on CPU and 20 hours on GPU.<br>
To train  : (epochs=40000)<br>
`python main.py`<br>

To replay : (modify `checkpoint = Path('trained_mario.chkpt')`)<br>
`python replay.py`<br>

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
**Code:** [Keras DDPG Pendulum](https://keras.io/examples/rl/ddpg_pendulum/)<br>
![](https://i.imgur.com/mS6iGyJ.jpg)

---
**Code:** [End to end motion planner using Deep Deterministic Policy Gradient (DDPG) in gazebo](https://github.com/m5823779/motion-planner-reinforcement-learning)<br>
![](https://github.com/m5823779/MotionPlannerUsingDDPG/raw/master/demo/demo.gif)

---
### Efficient Path Planning for Mobile Robot Based on DDPG
**Paper:** [Efficient Path Planning for Mobile Robot Based on Deep Deterministic Policy Gradient](https://www.mdpi.com/1424-8220/22/9/3579)<br>
![](https://www.mdpi.com/sensors/sensors-22-03579/article_deploy/html/images/sensors-22-03579-g003-550.jpg)
![](https://www.researchgate.net/publication/356204068/figure/fig4/AS:1094058088906760@1637855183678/Experimental-simulation-environment.jpg)

---
### ICM
**Paper:** [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)<br>
**Code:** [pathak22/noreward-rl](https://github.com/pathak22/noreward-rl)<br>
**Blog:** [Intrinsic Curiosity Module (ICM)](https://pathak22.github.io/noreward-rl/)<br>
<table>
<tr>
<td><img src="https://github.com/pathak22/noreward-rl/blob/master/images/mario1.gif?raw=true"></td>
<td><img src="https://github.com/pathak22/noreward-rl/blob/master/images/vizdoom.gif?raw=true"></td>
</tr>
</table>
<iframe width="800" height="450" src="https://www.youtube.com/embed/J3FHOyhUn3A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://pathak22.github.io/noreward-rl/resources/method.jpg)

---
### PPO
**Paper:** [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)<br>
**Blog:** [Proximal Policy Optimization (PPO)詳解](https://www.796t.com/article.php?id=120451)<br>
**On-policy vs Off-policy**<br>
On-Policy 方式是指用於學習的agent與觀察環境的agent是同一個，所以引數θ始終保持一致。**(邊做邊學)**<br>
Off-Policy方式是指用於學習的agent與用於觀察環境的agent不是同一個，他們的引數θ可能不一樣。**(在旁邊透過看別人做來學習)**<br>
比如下圍棋，On-Policy方式是agent親歷親為，而Off-Policy是一個agent看其他的agent下棋，然後去學習人家的東西。<br>

**Blog:** [以刺蝟索尼克遊戲為例講解PPO](https://www.gushiciku.cn/pl/2iWn/zh-tw)<br>
![](https://mdimg.wxwenku.com/getimg/356ed03bdc643f9448b3f6485edc229b2d79c2f2b070b36baaee3123e0b4a4a85fda5fe86d46de4209003cce987359ad.jpg)
Policy Gradient演算法存在步長選擇問題（對step size敏感）：步長太小; 訓練過於緩慢步長太大，訓練中誤差波動較大，當面對訓練過程波動較大的問題時，PPO可以輕鬆應對。
PPO近端策略優化的想法是通過限定每步訓練的策略更新的大小，來提高訓練智慧體行為時的穩定性。<br>
PPO引入了一個新的目標函式 Clipped surrogate objective function(裁剪的替代目標函式)，通過裁剪將策略更新約束在小範圍內。<br>
![](https://www.researchgate.net/publication/339651408/figure/fig3/AS:864987794907143@1583240569937/The-actor-critic-proximal-policy-optimization-Actor-Critic-PPO-algorithm-process.ppm)
**Code:** [Keras PPO Cartpole](https://keras.io/examples/rl/ppo_cartpole/)<br>
![](https://i.imgur.com/rd5tda1.png)

---
### TRPO
**Paper:** [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)<br>
**Blog:** [Trust Region Policy Optimization講解](https://www.twblogs.net/a/5d5ead97bd9eee541c32568c)<br>
TRPO 算法 (Trust Region Policy Optimization)和PPO 算法 (Proximal Policy Optimization)都屬於MM(Minorize-Maximizatio)算法。<br>
![](https://pic1.xuehuaimg.com/proxy/csdn/https://imgconvert.csdnimg.cn/aHR0cDovLzViMDk4OGU1OTUyMjUuY2RuLnNvaHVjcy5jb20vaW1hZ2VzLzIwMTkwMjAyLzkxZTgxMzNiZjljZTQ4Y2FiZGQ1MzI5ZjVjMTlkYzVjLmpwZWc)
在信任的區域之中，我們用 δ 變量 限制我們的搜索區域。文章中有用數學證明，這樣的區域可以保證在它達到局部或者全局最優策略之前，它的優化策略將會優於當前策略。

---
### C51 
**Paper:** [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)<br>
**Blog:** [A Distributional Perspective on Reinforcement Learning](https://vitalab.github.io/article/2021/02/12/C51.html)<br>
**Code:** [flyyufelix/C51-DDQN-Keras](https://github.com/flyyufelix/C51-DDQN-Keras)<br>

![](https://github.com/flyyufelix/C51-DDQN-Keras/blob/master/resources/c51.gif?raw=true)
![](https://flyyufelix.github.io/img/z_visual_1.png)

---
### HER 
**Paper:** [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)<br>
**Code:** [OpenAI HER](https://github.com/openai/baselines/tree/master/baselines/her)<br>
<iframe width="640" height="360" src="https://www.youtube.com/embed/Dz_HuzgMxzo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://d3i71xaburhd42.cloudfront.net/e29687df9e9a465c1b74f7487ccab23aec6f45d6/2-Figure1-1.png)

---
### MBMF
**Paper:** [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596)<br>
<iframe width="800" height="480" src="https://www.youtube.com/embed/G7lXiuEC8x0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://d3i71xaburhd42.cloudfront.net/cce22bf6405042a965a86557684c46a441f2a736/4-Figure2-1.png)

---
### SAC
**Paper:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)<br>

![](https://miro.medium.com/max/974/0*NgZ_bq_nUOq73jK_.png)

---
### TD3
**Paper:** [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)<br>
**Code:** [sfujim/TD3](https://github.com/sfujim/TD3)<br>

RAMDP(Robot Arm Markov Decision Process)<br>
![](https://www.researchgate.net/publication/338605159/figure/fig1/AS:847596578930701@1579094180896/Robot-Arm-Markov-Decision-Process-RAMDP.jpg)
TD3 with RAMDP<br>
![](https://www.researchgate.net/publication/338605159/figure/fig2/AS:847596578947080@1579094180953/Structure-of-TD3-Twin-Delayed-Deep-Deterministic-Policy-Gradient-with-RAMDP.jpg
)

---
### POMDP (Partially-Observable Markov Decision Process)
**Paper:** [Planning and acting in partially observable stochastic domains](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)<br>

![](https://ars.els-cdn.com/content/image/1-s2.0-S2352154618301670-gr1_lrg.jpg)

---
### SHER
**Paper:** [Soft Hindsight Experience Replay](https://arxiv.org/abs/2002.02089)

![](https://d3i71xaburhd42.cloudfront.net/6253a0f146a36663e908509e14648f8e2a5ab581/5-Figure3-1.png)

---
## Exercises: [RL-gym](https://github.com/rkuo2000/RL-gym)
```
sudo apt-get install ffmpeg freeglut3-dev xvfb
pip install pyglet==1.5.27
pip install stable_baselines3[extra]
pip install gym[all]
pip install autorom[accept-rom-license]
git clone https://github.com/rkuo2000/RL-gym
cd RL-gym
cd cartpole
```

---
### Cartpole 
~/RL-gym/cartpole/random_action.py, q_learn.py, dqn.py<br>
`python3 random_action.py`<br>
`python3 q_learning.py`<br>
`python3 dqn.py`<br>
![](https://github.com/rkuo2000/RL-gym/blob/main/assets/CartPole.gif?raw=true)

---
### Stable-Baselines3
~/RL-gym/sb3/train.py, enjoy.py<br>

alogrithm = A2C, output = xxx.zip<br>
`python3 train.py CartPole-v0 160000`<br>
`python3 enjoy.py CartPole-v0`<br>

`python3 train.py Pendulum-v1 640000`<br>
`python3 enjoy.py Pendulum-v1`<br>

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
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_CrazyClimber_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_DoubleDunk_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_IceHockey_a2c.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_boxing_ppo.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_qbert_ppo.gif?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/gym_tennis_a2c.gif?raw=true"></td>
</tr>
</table>

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
### RL-SB3 Zoo Exercises
```
git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
cd rl-baselines3-zoo

conda install -c conda-forge huggingface_hub
```

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
### [Pybullet](https://pybullet.org)
Bullet Real-Time Physics Simulation<br>

![](https://pybullet.org/wordpress/wp-content/uploads/2019/03/cropped-pybullet.png)

![](https://pybullet.org/wordpress/wp-content/uploads/2021/04/download.png)

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
pip install stable-baselines3
git clone https://github.com/rkuo2000/pybullet-gym
export PYTHONPATH=$PATH:/home/yourname/pybullet-gym
```

* gym
**Env names:** *Ant, Atlas, HalfCheetah, Hopper, Humanoid, HumanoidFlagrun, HumanoidFlagrunHarder, InvertedPendulum, InvertedDoublePendulum, InvertedPendulumSwingup, Reacher, Walker2D*<br>

**Train**<br>
`python train.py Ant 10000000`<br>

**Enjoy** with trained-model<br>
`python enjoy.py Ant`<br>

**Enjoy** with pretrained weights<br>
`python enjoy_Ant.py`<br>
`python enjoy_HumanoidFlagrunHarder.py` (a copy from pybulletgym/examples/roboschool-weights/enjoy_TF_*.py)<br>


**Blog:** <br>
[Creating OpenAI Gym Environments with PyBullet (Part 1)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24)<br>
[Creating OpenAI Gym Environments with PyBullet (Part 2)](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e)<br>
![](https://media0.giphy.com/media/VI3OuvQShK3gzENiVz/giphy.gif?cid=790b761131bda06b74fcd9bb06c6a43939cf446edf403a68&rid=giphy.gif&ct=g)

---
### [OpenAI procgen](https://github.com/openai/procgen)
![](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)
`pip install procgen`<br>
`python -m procgen.interactive --env-name starpilot`<br>

```
import gym
env = gym.make('procgen:procgen-coinrun-v0')
obs = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        break
```

---
### [OpenAI Gym Environments for Donkey Car](https://github.com/tawnkramer/gym-donkeycar)
* [Donkey Simulator User Guide](https://docs.donkeycar.com/guide/simulator/)
![](https://docs.donkeycar.com/assets/sim_screen_shot.png)
* [Documentation](https://gym-donkeycar.readthedocs.io/en/latest/)
* Download [simulator binaries](https://github.com/tawnkramer/gym-donkeycar/releases)
* Environments:
  - "donkey-warehouse-v0"
  - "donkey-generated-roads-v0"
  - "donkey-avc-sparkfun-v0"
  - "donkey-generated-track-v0"
  - "donkey-roboracingleague-track-v0"
  - "donkey-waveshare-v0"
  - "donkey-minimonaco-track-v0"
  - "donkey-warren-track-v0"
  - "donkey-thunderhill-track-v0"
  - "donkey-circuit-launch-track-v0"
* Example Usage:

```
import os
import gym
import gym_donkeycar
import numpy as np

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
exe_path = f"{PATH_TO_APP}/donkey_sim.exe"
port = 9091

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-track-v0", conf=conf)

# PLAY
obs = env.reset()
for t in range(100):
  action = np.array([0.0, 0.5]) # drive straight with small speed
  # execute the action
  obs, reward, done, info = env.step(action)

# Exit the scene
env.close()
```

---
### [Google Dopamine](https://github.com/google/dopamine)
Dopamine is a research framework for fast prototyping of reinforcement learning algorithms.<br>
Dopamine supports the following agents, implemented with [jax](https://github.com/google/jax): DQN, C51, Rainbow, IQN, SAC. 

---
### JAX
[JAX](https://github.com/google/jax) is Autograd and XLA, brought together for high-performance machine learning research.<br>
[Autograd](https://github.com/hips/autograd) can automatically differentiate native Python and NumPy functions.<br>
[XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.<br>
```
import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)  # inputs to the next layer
    return outputs                  # no activation on last layer

def loss(params, inputs, targets):
    preds = predict(params, inputs)
    return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads
```

---
### [ViZDoom](https://github.com/mwydmuch/ViZDoom)
![](https://camo.githubusercontent.com/a7d9d95fc80903bcb476c2bbdeac3fa7623953c05401db79101c2468b0d90ad9/687474703a2f2f7777772e63732e7075742e706f7a6e616e2e706c2f6d6b656d706b612f6d6973632f76697a646f6f6d5f676966732f76697a646f6f6d5f636f727269646f725f7365676d656e746174696f6e2e676966)
`sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-ev libopenal-dev zlib1g-dev timidity tar nasm`<br>
`pip install vizdoom`<br>

---
### Highway Env
**Code:** [highway-env](https://github.com/eleurent/highway-env)<br>

![](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway-env.gif)
![](https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/parking-env.gif)

---
## AI in Games
**Paper:** [AI in Games: Techniques, Challenges and Opportunities](https://arxiv.org/abs/2111.07631)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AI_in_Games_survey.png?raw=true)

---
### AlphaGo
2016 年 3 月，AlphaGo 這一台 AI 思維的機器挑戰世界圍棋冠軍李世石（Lee Sedol）。比賽結果以 4 比 1 的分數，AlphaGo 壓倒性的擊倒人類世界最會下圍棋的男人。<br>
<iframe width="710" height="399" src="https://www.youtube.com/embed/1bc-8iomgB4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AlphaGo_version_comparison.png?raw=true)

---
### AlphaZero with a Learned Model
**Paper:** [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)<br>
RL can be divided into Model-Based RL (MBRL) and Model-Free RL (MFRL). Model-based RL uses an environment model for planning, whereas model-free RL learns the optimal policy directly from interactions. Model-based RL has achieved superhuman level of performance in Chess, Go, and Shogi, where the model is given and the game requires sophisticated lookahead. However, model-free RL performs better in environments with high-dimensional observations where the model must be learned.
![](https://www.endtoend.ai/assets/blog/rl-weekly/36/muzero.png)

---
### Minigo
**Code:** [tensorflow minigo](https://github.com/tensorflow/minigo)<br>

### ELF OpenGo
**Code:** [https://github.com/pytorch/ELF](https://github.com/pytorch/ELF)<br>
ELF is an Extensive, Lightweight, and Flexible platform for game research. <br>
We have used it to build our Go playing bot, ELF OpenGo, which achieved a 14-0 record versus four global top-30 players in April 2018. The final score is 20-0 (each professional Go player plays 5 games).<br>

**Blog:** [A new ELF OpenGo bot and analysis of historical Go games](https://ai.facebook.com/blog/open-sourcing-new-elf-opengo-bot-and-go-research/)<br>
![](https://scontent.ftpe3-2.fna.fbcdn.net/v/t39.2365-6/52493263_1313343392140217_5544097798909067264_n.gif?_nc_cat=102&ccb=1-5&_nc_sid=ad8a9d&_nc_ohc=jeW2W5LSSzoAX--ger8&_nc_ht=scontent.ftpe3-2.fna&oh=00_AT-2aLF-P6DmbL0eDET98OEtOubyWBSnlRmCm_1wCiXwfw&oe=61C3480E)

---
### Othello Zero
![](http://web.stanford.edu/~surag/posts/images/lategame.svg)
**Code:** [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)<br>
`git clone https://github.com/suragnair/alpha-zero-general`<br>
`cd alpha-zero-general`<br>
`pip install coloredlogs`<br>
To start training a model for Othello:<br>
`python main.py`<br>

**Blog:** [A Simple Alpha(Go) Zero Tutorial](http://web.stanford.edu/~surag/posts/alphazero.html)<br>
a PyTorch model for 6x6 Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn).<br>
This took about 3 days on an NVIDIA Tesla K80. <br>
To play othello (using `pretrained_models/othello/pytorch/.`)<br>
`python pit.py`<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AlphaZero_othello.png?raw=true)

---
### Chess Zero
**Code:** [Zeta36/chess-alpha-zero](https://github.com/Zeta36/chess-alpha-zero)<br> 

<img width="50%" height="50%" src="https://user-images.githubusercontent.com/4205182/34057277-e9c99118-e19b-11e7-91ee-dd717f7efe9d.PNG">

---
### AlphaStar
**Blog:** [AlphaStar: Mastering the real-time strategy game StarCraft II](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)<br>
![](https://lh3.googleusercontent.com/ckm-3GlBQJ4zbNzfiW97yPqj5PVC0qIbRg42FL35EbDkhWoCNxyNZMMJN-f6VZmLMRbyBk2PArLQ-jDxlHbsE3_YaDUmcxUvMf8M=w1440-rw-v1)
**Blog:** [AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning)<br>
**Code:** [PySC2 - StarCraft II Learning Environment](https://github.com/deepmind/pysc2)<br>
<iframe width="964" height="542" src="https://www.youtube.com/embed/KPLYhRBCcvk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [OpenAI Five](https://openai.com/blog/openai-five/) at Dota2
<iframe width="964" height="542" src="https://www.youtube.com/embed/UZHTNBMAfAA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [DeepMind FTW](https://deepmind.com/blog/article/capture-the-flag-science)
![](https://lh3.googleusercontent.com/CFlAYmP49qitU-SOP_PaKtV1kOrlpNnvo4oEDFhyxelrVwKyAbkdXwUuDFRTmiRSQle4955mmOAB4jrrIrWzIXDt8hOajZGtJzNaDRw=w1440-rw-v1)
![](https://lh3.googleusercontent.com/RterJzRGidwT9R_Dqeu5LY5MZPjjYRc-MQdQyca7gACnA7w0bjCu_hIcoXLC4xV5zebvdZnN7ocZkemGnF4K7_p5SMZCLRWbNq1IDQ=w1440-rw-v1)
<img width="50%" height="50%" src="https://lh3.googleusercontent.com/lxvZW4tdNAmYZ8ku873U4XYa-SljFgh8EPHG9-HbkqdjTBjEyktRSBTQGML6oBcjkYIJqnoXAnp50Z8t876WaucgotFlcTgfN8urJg=w1440-rw-v1">

---
### Texas Hold'em Poker
**Code:** [fedden/poker_ai](https://github.com/fedden/poker_ai)<br>
**Code:** [Pluribus Poker AI](https://github.com/kanagle2312/pluribus-poker-AI) + [poker table](https://codepen.io/Rovak/pen/ExYeQar)<br>
<iframe width="832" height="468" src="https://www.youtube.com/embed/wInvN096he8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**Blog:** [Artificial Intelligence Masters The Game of Poker – What Does That Mean For Humans?](https://www.forbes.com/sites/bernardmarr/2019/09/13/artificial-intelligence-masters-the-game-of-poker--what-does-that-mean-for-humans/?sh=dcaa18f5f9ea)<br>

**DeepStack:** Scalable Approach to Win at Poker<br>
The DeepStack team, from the University of Alberta in Edmonton, Canada, combined deep machine learning and algorithms to create AI capable of winning at two-player, “no-limit” Texas Hold ’em.<br>

**Libratus:** Masters Two-Player Texas Hold ’Em<br>
Libratus is an AI, built by Noam Brown and Tuomas Sandholm of Carnegie Mellon University in 2017, that was ultimately unbeatable at two-person poker. This system required 100 central processing units (CPUs) to run.<br>

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
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/JueWu_structure.png?raw=true)

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

`git lone https://github.com/rkuo2000/spatial-intention-maps`<br>
`pip install -r requirements.txt`<br>
`cd shortest_paths`<br>
`python setup.py build_ext --inplace`<br>
**Pretrained:**<br>
`cd ..`<br>
`./download-pretrained.sh`<br>

**Playing multiple agents:**
* 4 lifting robots<br>
`python enjoy.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml`<br>
`python enjoy.py --config-path logs/20201214T092812731965-lifting_4-large_empty-ours/config.yml`<br>
![](https://user-images.githubusercontent.com/6546428/111895630-3842bc80-89d1-11eb-9150-1364f80e3a26.gif)

* 4 pushing robots<br>
`python enjoy.py --config-path logs/20201214T092814688334-pushing_4-small_divider-ours/config.yml`<br>
`python enjoy.py --config-path logs/20201217T171253620771-pushing_4-large_empty-ours/config.yml`<br>
* 2 lifting + 2 pushing<br>
`python enjoy.py --config-path logs/20201214T092812868257-lifting_2_pushing_2-large_empty-ours/config.yml`
![](https://user-images.githubusercontent.com/6546428/111895627-35e06280-89d1-11eb-9cf7-0de0595ae68f.gif)
* 2 lifting + 2 throwing<br>
`python enjoy.py --config-path logs/20201217T171253796927-lifting_2_throwing_2-large_empty-ours/config.yml`
* 4 rescue robots<br>
`python enjoy.py --config-path logs/20210120T031916058932-rescue_4-small_empty-ours/config.yml`
![](https://user-images.githubusercontent.com/6546428/111895633-38db5300-89d1-11eb-9993-d508e6c32e7c.gif)


**Playing single agent:**
* 1 lifting robot<br>
`python enjoy.py --config-path logs/20201217T171254022070-lifting_1-small_empty-base/config.yml`
![](https://user-images.githubusercontent.com/6546428/111895625-34169f00-89d1-11eb-8687-689122e6b3f2.gif)
* 1 pushing robot<br>
`python enjoy.py --config-path logs/20201214T092813073846-pushing_1-small_empty-base/config.yml`
![](https://user-images.githubusercontent.com/6546428/111895631-38db5300-89d1-11eb-9ad4-81be3908f383.gif)
* 1 rescue robot<br>
`python enjoy.py --config-path logs/20210119T200131797089-rescue_1-small_empty-base/config.yml`
![](https://user-images.githubusercontent.com/6546428/111895632-38db5300-89d1-11eb-800e-8652d163ff1b.gif)

---
### MARL PPO
**Paper:** [Emergent Autonomous Racing Via Multi-Agent Proximal Policy Optimization](https://www.researchgate.net/profile/Ryan-Sander-2/publication/341769263_Emergent_Autonomous_Racing_Via_Multi-Agent_Proximal_Policy_Optimization/links/5fba8974458515b7976263fd/Emergent-Autonomous-Racing-Via-Multi-Agent-Proximal-Policy-Optimization.pdf)<br>
**Blog:** [Deep Multi-Agent Reinforcement Learning with TensorFlow-Agents](https://medium.com/analytics-vidhya/deep-multi-agent-reinforcement-learning-with-tensorflow-agents-1d4a91734d1f)<br>
**Code:** [rmsander/marl_ppo](https://github.com/rmsander/marl_ppo)<br>
[Multi-Car Racing Gym Environment](https://github.com/igilitschenski/multi_car_racing)<br>
![](https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif)
`
---
## Imitation Learning
**Blog:** [A brief overview of Imitation Learning](https://smartlabai.medium.com/a-brief-overview-of-imitation-learning-8a8a75c44a9c)<br>
<iframe width="742" height="417" src="https://www.youtube.com/embed/WjFdD7PDGw0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

1. **Behavioural Cloning**<br>
**Paper:** [Behavioral Cloning from Observation](https://arxiv.org/abs/1805.01954)<br>
![](https://miro.medium.com/max/620/0*A-G4NfY9Zi5O8e-v.png)
![](https://d3i71xaburhd42.cloudfront.net/cc2fb12eaa4dae74c5de0799b29624b5c585c43b/1-Figure1-1.png)

2. **Direct Policy Learning**
![](https://miro.medium.com/max/464/0*yfAgeqJtgI8eGE-O.png)

3. **Inverse Reinforcement Learning**<br>
**Paper:** [A Survey of Inverse Reinforcement Learning](https://arxiv.org/abs/1806.06877)<br>
![](https://ars.els-cdn.com/content/image/1-s2.0-S0004370221000515-gr003.jpg)

<table>
<tr>
<td><img src="https://www.researchgate.net/profile/Utku-Koese/publication/316786383/figure/fig2/AS:492006683222017@1494314942380/Reinforcement-learning-Cornell-University-2011.png"></td>
<td><img src="https://www.researchgate.net/profile/Utku-Koese/publication/316786383/figure/fig3/AS:492006683222018@1494314942397/Inverse-Reinforcement-learning-Cornell-University-2011.png"></td>
</tr>
</table>

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
![](https://d3i71xaburhd42.cloudfront.net/2d48c903c37cd0776e8b1defb8ec108ce3276aac/4-Figure2-1.png)
<iframe width="742" height="417" src="https://www.youtube.com/embed/2rx-roLYJ5k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Surgical Robotics
**Paper:** [Open-Sourced Reinforcement Learning Environments for Surgical Robotics](https://arxiv.org/abs/1903.02090)<br>
**Code:** [RL Environments for the da Vinci Surgical System](https://github.com/ucsdarclab/dVRL)<br>
**YouTube:** [Open-Sourced Reinforcement Learning Environments for Surgical Robotics](https://youtu.be/xu4sqrO_2AY)<br>

---
## Meta Learning (Learning to Learn)
**Blog:** [Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)<br>
An example of 4-shot 2-class image classification. <br>
![](https://lilianweng.github.io/lil-log/assets/images/few-shot-classification.png)

<iframe width="625" height="469" src="https://www.youtube.com/embed/xoastiYx9JU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="625" height="469" src="https://www.youtube.com/embed/Q68Eh-wm1Ts" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Meta_Learning_algorithms.png?raw=true)

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

---
### Stock Trading
**Blog:** [Predicting Stock Prices using Reinforcement Learning (with Python Code!)](https://www.analyticsvidhya.com/blog/2020/10/reinforcement-learning-stock-price-prediction/)<br>
![](https://editor.analyticsvidhya.com/uploads/770801_26xDRHI-alvDAfcPPJJGjQ.png)

**Code:** [DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading)<br>
**Code:** [FinRL](https://github.com/AI4Finance-Foundation/FinRL)<br>
**Blog:** [Automated stock trading using Deep Reinforcement Learning with Fundamental Indicators](https://medium.com/@mariko.sawada1/automated-stock-trading-with-deep-reinforcement-learning-and-financial-data-a63286ccbe2b)<br>

**Papers:** <br>
[2010.14194](https://arxiv.org/abs/2010.14194): Learning Financial Asset-Specific Trading Rules via Deep Reinforcement Learning<br>
[2011.09607](https://arxiv.org/abs/2011.09607): FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance<br>
[2101.03867](https://arxiv.org/abs/2101.03867): A Reinforcement Learning Based Encoder-Decoder Framework for Learning Stock Trading Rules<br>
[2106.00123](https://arxiv.org/abs/2106.00123): Deep Reinforcement Learning in Quantitative Algorithmic Trading: A Review<br>
[2111.05188](https://arxiv.org/abs/2111.05188): FinRL-Podracer: High Performance and Scalable Deep Reinforcement Learning for Quantitative Finance<br>
[2112.06753](https://arxiv.org/abs/2112.06753): FinRL-Meta: A Universe of Near-Real Market Environments for Data-Driven Deep Reinforcement Learning in Quantitative Finance<br>
**Blog:** [FinRL­-Meta: A Universe of Near Real-Market En­vironments for Data­-Driven Financial Reinforcement Learning](https://medium.datadriveninvestor.com/finrl-meta-a-universe-of-near-real-market-en-vironments-for-data-driven-financial-reinforcement-e1894e1ebfbd)<br>
![](https://miro.medium.com/max/2000/1*rOW0RH56A-chy3HKaxcjNw.png)

---
## Exercises:
### Stock DQN
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


