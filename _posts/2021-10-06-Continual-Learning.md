---
layout: post
title: Continual Learning
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Introduction to Continual Learning / LifeLong Learning / Incremental Learning in image classification, object detection.

---
## Datasets

### CIFAR-10/CIFAR-100
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CIFAR-10.png?raw=true)

**CIFAR-10**<br>
* 50000 training images
* 10000 test images
* 10 classes of 32x32 color images

**CIFAR-100**<br>
* 500 training images per class
* 100 test images per class
* 100 classes of 32x32 color images

---
### [CORe50](https://vlomonaco.github.io/core50/)
![](https://vlomonaco.github.io/core50/imgs/classes.gif)
[CORe50](https://arxiv.org/abs/1705.03550), specifically designed for Continual Object Recognition, is a collection of 50 domestic objects belonging to 10 categories: plug adapters, mobile phones, scissors, light bulbs, cans, glasses, balls, markers, cups and remote controls. 
Classification can be performed at object level (50 classes) or at category level (10 classes).

---
### [SAILenv](https://sailab.diism.unisi.it/sailenv/)
**Paper:** [Evaluating Continual Learning Algorithms by Generating 3D Virtual Environments](https://arxiv.org/abs/2109.07855)<br>

* Pixel-wise Annotations<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/SAILab_Pixel-wise_Annotations.png?raw=true)
* Object Library<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/SAILab_Object_Library.png?raw=true)
* Ready-To-Go Scenes<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/SAILab_Ready-To-Go_Scenes.png?raw=true)

* Server Executables (Sample Scenes): [Version Directory](http://eliza.diism.unisi.it/sailenv/bin)
* Source Unity Project (Customizable): [Source Code](http://eliza.diism.unisi.it/sailenv/source)
* Client Python API: [Source Code (GitHub)](https://github.com/sailab-code/SAILenv), [Pip Package](https://pypi.org/project/sailenv/)
* 3D Models .OBJ for Adversarial Attacks: [ZIP archive](https://sailab.diism.unisi.it/wp-content/uploads/2021/09/meshes.zip)

```
from sailenv.agent import Agent

agent = Agent(width=256, height=192, host="192.168.1.3", port=8085)
agent.register()
agent.change_scene(agent.scenes[2])

while True:
    frame_views = agent.get_frame()
	...

agent.delete()
```

---
## [Continual Learning](https://medium.com/continual-ai/continual-learning-da7995c24bca)
**Blog:** [李宏毅 lifelong learning](https://www.twblogs.net/a/5ef61d425ddd268f20a86ec8)<br>
**Blog:** [Catastrophic Forgetting in Neural Networks Explained](https://mrifkikurniawan.github.io/blog-posts/Catastrophic_Forgetting/)<br>
![](https://mrifkikurniawan.github.io/images/catastrophic_forgetting/forgetting_cl_task.jpg)
![](https://mrifkikurniawan.github.io/images/catastrophic_forgetting/forgetting_forgetting.svg)

**Colab:** [An example of catastrophic forgetting in PyTorch](https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb)<br>
The effect of AR1 (CwR+Syn) is displayed in the figure below, based on the CORe50 dataset.<br>
![](https://miro.medium.com/max/1288/1*TNhs2-QHivjNYxYFdDF9gw.png)

---
### LWF
**Paper:** [Learning without Forgetting (LWF)](https://arxiv.org/abs/1606.09282)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/LWF_architecture.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/LWF_algorithm.png?raw=true)

---
### EWC (Elastic Weights Consolidation)
**Paper:** [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)<br>
**Paper:** [Elastic Weight Consolidation (EWC): Nuts and Bolts](https://arxiv.org/abs/2105.04093)<br>
**Code:** [https://github.com/ariseff/overcoming-catastrophic](https://github.com/ariseff/overcoming-catastrophic)<br>
![](https://slidetodoc.com/presentation_image/3df85a200151a347c4666719e75730f5/image-12.jpg)
![](https://slidetodoc.com/presentation_image/3df85a200151a347c4666719e75730f5/image-14.jpg)

---
### CWR, CWR+, AR1
**Paper:** [Continuous Learning in Single-Incremental-Task Scenarios](https://arxiv.org/abs/1806.08568)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CWR_algorithm.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CWR+_algorithm.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AR1_algorithm.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Continual_Learning_alorithms_analysis.png?raw=true)

---
### GDM (Growing Dual Memory)
**Paper:** [Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization](https://arxiv.org/abs/1805.10966)<br>
![](https://www.frontiersin.org/files/Articles/401624/fnbot-12-00078-HTML/image_m/fnbot-12-00078-g001.jpg)
**Code:** [https://github.com/giparisi/GDM](https://github.com/giparisi/GDM)<br>
**Toolbox:** [GWR Toolbox](https://github.com/giparisi/gwr-tb)<br>

---
### AR1 
**Paper:** [Latent Replay for Real-Time Continual Learning](https://arxiv.org/abs/1912.01100)
![](https://repository-images.githubusercontent.com/239764197/2c621f00-8f13-11ea-8250-162421cbd36b)
**Code:** [AR1* with Latent Replay](https://github.com/vlomonaco/ar1-pytorch)<br>

---
### CAT
**Paper:** [Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](https://arxiv.org/abs/2112.10017)<br>
**Code:** [https://github.com/ZixuanKe/CAT](https://github.com/ZixuanKe/CAT)<br>
![](https://github.com/ZixuanKe/CAT/raw/main/CAT.png)

---
### GDM for Lifelong 3D Object Recognition
**Paper:** [Lifelong 3D Object Recognition and Grasp Synthesis Using Dual Memory Recurrent Self-Organization Networks](https://arxiv.org/abs/2109.11544)<br>
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/e684b962d27817b898839ff62630d8622dd47dac/2-Figure1-1.png)

---
### Class-Incremtnatl Learning with Generative Classifier
**Paper:** [Class-Incremental Learning with Generative Classifiers](https://arxiv.org/abs/2104.10093)<br>
**Code:** [https://github.com/GMvandeVen/class-incremental-learning](https://github.com/GMvandeVen/class-incremental-learning)<br>
**Kaggle:** [https://kaggle.com/rkuo2000/class-incremental-learning](https://kaggle.com/rkuo2000/class-incremental-learning)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Class-Incremental_Learning_with_Generative_Classifiers.png?raw=true)

---
### RMN (Revelance Mapping Network)
**paper:** [Understanding Catastrophic Forgetting and Remembering in Continual Learning with Optimal Relevance Mapping](https://arxiv.org/abs/2102.11343)<br>
**Code:** [https://gitlab.com/prakhark2/relevance-mapping-networks](https://gitlab.com/prakhark2/relevance-mapping-networks)<br>

---
### [PyContinual](https://github.com/ZixuanKe/PyContinual)
PyContinual (An Easy and Extendible Framework for Continual Learning)<br>
![](https://github.com/ZixuanKe/PyContinual/raw/main/docs/benchmarks.png)

**Paper:**<br>
* [Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning](https://arxiv.org/abs/2112.02706)
* [CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks](https://arxiv.org/abs/2112.02714)
* [Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks](https://arxiv.org/abs/2112.03271)
* [Continual Learning with Knowledge Transfer for Sentiment Classification](https://arxiv.org/abs/2112.10021)
* [Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](https://arxiv.org/abs/2112.10017)

**Features:**
* Datasets: It currently supports Language Datasets (Document/Sentence/Aspect Sentiment Classification, Natural Language Inference, Topic Classification) and Image Datasets (CelebA, CIFAR10, CIFAR100, FashionMNIST, F-EMNIST, MNIST, VLCS)
* Scenarios: It currently supports Task Incremental Learning and Domain Incremental Learning
* Training Modes: It currently supports single-GPU. You can also change it to multi-node distributed training and the mixed precision training.

---
### LwF-ECG
**Paper:** [LwF-ECG: Learning-without-forgetting approach for electrocardiogram heartbeat classification based on memory with task selector](https://www.sciencedirect.com/science/article/pii/S0010482521006016)<b>
![](https://ars.els-cdn.com/content/image/1-s2.0-S0010482521006016-ga1_lrg.jpg)

---
### MAML (Model-Agnostic Meta-Learning)
**Blog:** [MAML模型介绍及算法详解](https://zhuanlan.zhihu.com/p/57864886)<br>
**Paper:** [Model-Agnostic Meta-Learning for Fast Adaption of Deep Networks](https://arxiv.org/abs/1703.03400)<br>
**Code:** [https://github.com/cbfinn/maml](https://github.com/cbfinn/maml)<br>

**Blog:** [MAML复现全部细节和经验教训（Pytorch）](https://blog.csdn.net/wangkaidehao/article/details/105507809)<br>
**Code:** [https://github.com/miguealanmath/MAML-Pytorch](https://github.com/miguealanmath/MAML-Pytorch)<br>

---
### Lifelong Object Detection
**Paper:** [Lifelong Object Detection](https://arxiv.org/abs/2009.01129)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Lifelong_Object_Detection.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Lifelong_Object_Detection_architecture.png?raw=true)

---
### RECALL
**Paper:** [RECALL: Replay-based Continual Learning in Semantic Segmentation](https://arxiv.org/abs/2108.03673)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RECALL_architecture.png?raw=true)

---
### Contrast R-CNN
**Paper:** [Contrast R-CNN for Continual Learning in Object Detection](https://arxiv.org/abs/2108.04224)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Contrast_R-CNN_architecture.png?raw=true)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

