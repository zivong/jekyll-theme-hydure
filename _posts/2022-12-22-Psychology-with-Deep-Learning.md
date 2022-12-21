---
layout: post
title: Psychology with Deep Learning
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Sentiment Analysis, Emoton Detection, Speech Emotion Recognition, MBTI, Behavior Prediction, Social Simulation, Theory of Mind, Brain Model, Episodic Memory, Semantic Memory, The Emotion Machine.

---
### Python libraries for Psychology researchers
**[PsychoPy](https://www.psychopy.org/)**<br>
PsychoPy is also a Python application for creating Psychology experiments. 

**Blog:** [Best Python libraries for Psychology researchers](https://www.marsja.se/best-python-libraries-psychology/)<br>
* **PsyUtils** “The psyutils package is a collection of utility functions useful for generating visual stimuli and analysing the results of psychophysical experiments.
* **Psisignifit** is a toolbox that allows you to fit psychometric functions. Further, hypotheses about psychometric data can be tested. 
* **Pygaze** is a Python library for eye-tracking data & experiments. 
* **MNE** is a library designed for processing electroencephalography (EEG) and magnetoencephalography (MEG) data.
* **Kabuki** is a Python library for the effortless creation of hierarchical Bayesian models.
* **Scikit-learn** is an excellent Python package if you want to learn how to do machine learning

---
### [Machine Learning in Psychometrics and Psychological Research](https://www.frontiersin.org/articles/10.3389/fpsyg.2019.02970/full)
**Paper:** [Review of Machine Learning Algorithms for Diagnosing Mental Illness](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6504772/pdf/pi-2018-12-21-2.pdf)<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/machine%20learning%20algorithms%20used%20in%20mental%20health.png?raw=true)

---
## Bio-inspired Computer Vision
### Towards a synergistic approach of artificial and biological vision
**Book:** [Bio-inspired computer vision: Towards a synergistic approach of artificial and biological vision](https://hal.inria.fr/hal-01131645v3/document)<br>
![](https://ars.els-cdn.com/content/image/1-s2.0-S1077314216300339-gr1_lrg.jpg)

---
### Computational graph of a foveated spatial transformer network
**Code:** [int-lab-book](https://github.com/dabane-ghassan/int-lab-book)
![](https://github.com/dabane-ghassan/int-lab-book/raw/main/foveated_st.png)

---
## Sentiment 
### Sentiment Analysis of Tweets
**Blog:** [SENTIMENT ANALYSIS OF TWEETS WITH PYTHON, NLTK, WORD2VEC & SCIKIT-LEARN](https://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost/index.html)<br>
**Dataset:** [First GOP Debate Twitter Sentiment](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)<br>
**Code:** [Sentiment-Analysis-NLTK-ML and LSTM](https://github.com/nagypeterjob/Sentiment-Analysis-NLTK-ML-LSTM)<br>
**Kaggle:** [Sentiment NLTK](https://www.kaggle.com/rkuo2000/sentiment-nltk)<br>
**Kaggle:** [Sentiment LSTM](https://www.kaggle.com/rkuo2000/sentiment-lstm)<br>
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-pos.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-neg.png?raw=true"></td>
</tr>
</table>

---
### Depression Detection
**Paper:** [Deep Learning for Depression Detection of Twitter Users](https://aclanthology.org/W18-0609.pdf)<br>
**Paper:** [Machine Learning-based Approach for Depression Detection in Twitter Using Content and Activity Features](https://arxiv.org/abs/2003.04763)<br>
**Paper:** [A comprehensive empirical analysis on cross-domain semantic enrichment for detection of depressive language](https://arxiv.org/abs/2106.12797)<br>
**Paper:** [DepressionNet: A Novel Summarization Boosted Deep Framework for Depression Detection on Social Media](https://arxiv.org/abs/2105.10878)<br>
**Code:** [Detect Depression In Twitter Posts](https://github.com/peijoy/DetectDepressionInTwitterPosts)<br>

---
## Emotion Detection
### Facial Expression Recognition
**Dataset:** [FER-2013](https://www.kaggle.com/msambare/fer2013)<br>
* 7 facial expression, 28709 training images, 7178 test images
* labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

**Kaggle:** [fer2013-cnn](https://www.kaggle.com/rkuo2000/fer2013-cnn)<br>

**Code:** [EmoPy](https://github.com/thoughtworksarts/EmoPy)<br>
![](https://github.com/thoughtworksarts/EmoPy/blob/master/readme_docs/labeled_images_7.png?raw=true)

---
### First Impression
**Paper:** [Predicting First Impressions with Deep Learning](https://arxiv.org/abs/1610.08119)<br>
**Code:** [mel-2445/Predicting-First-Impressions](https://github.com/mel-2445/Predicting-First-Impressions)<br>
Annotations : Age, Dominance, IQ, Trustworthiness<br>
![](https://d3i71xaburhd42.cloudfront.net/d2abea314816d5479212baead76b5fc18f485781/3-Figure2-1.png)

---
### Large-Scale Facial Expression Recognition
**Paper:** [Suppressing Uncertainties for Large-Scale Facial Expression Recognition](https://arxiv.org/abs/2002.10392)<br>
**Dataset:** [Real-world Affective Faces Database](http://www.whdeng.cn/raf/model1.html) (RAF-DB)<br>
* 29672 number of real-world images,
* a 7-dimensional expression distribution vector for each image,
* two different subsets: single-label subset, including 7 classes of basic emotions; two-tab subset, including 12 classes of compound emotions,
* 5 accurate landmark locations, 37 automatic landmark locations, bounding box, race, age range and gender attributes annotations per image,
* baseline classifier outputs for basic emotions and compound emotions.

**Code:** [Self-Cure-Network](https://github.com/kaiwang960112/Self-Cure-Network)<br>
![](https://github.com/kaiwang960112/Self-Cure-Network/blob/master/imgs/visularization2.png?raw=true)
![](https://github.com/kaiwang960112/Self-Cure-Network/blob/master/imgs/SCNpipeline.png?raw=true)

---
### Emotion Recognition from Body Gestures
**Paper:** [A Generalized Zero-Shot Framework for Emotion Recognition from Body Gestures](https://arxiv.org/abs/2010.06362)<br>
![](https://d3i71xaburhd42.cloudfront.net/04a41b4f1c7b4c346165b6ba567c51ca94ef64ce/4-Figure1-1.png)

---
## Speech Emotion Recognition (SER)
### Datasets
* [Toronto emotional speech set (TESS)](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
* [CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)](https://github.com/CheyneyComputerScience/CREMA-D)
* [Surrey Audio-Visual Expressed Emotion (SAVEE)](https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee)
* [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.YerjzIRBxH5)
* [RAVDESS Emotional speech audio](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

---
### Audio Emotion Recognition
**Kaggle:** [Audio Emotion Recognition](https://www.kaggle.com/dasanurag38/audio-emotion-recognition)<br>
**Kaggle:** [Audio Emotion](https://www.kaggle.com/ejlok1/audio-emotion-part-6-2d-cnn-66-accuracy)<br>

---
### BERT-like self supervised models for SER
**Paper:** [Jointly Fine-Tuning "BERT-like" Self Supervised Models to Improve Multimodal Speech Emotion Recognition](https://arxiv.org/abs/2008.06682)<br>
**Code:** [shamanez/BERT-like-is-All-You-Need](https://github.com/shamanez/BERT-like-is-All-You-Need)<br>
<img width="70%" height="70%" src="https://github.com/shamanez/BERT-like-is-All-You-Need/blob/master/pipeline.jpg?raw=true">

---
### Fine-Grained Cross Modality Excitement for SER
**Paper:** [Learning Fine-Grained Cross Modality Excitement for Speech Emotion Recognition](https://arxiv.org/abs/2010.12733)<br>
**Code:** [tal-ai/FG_CME](https://github.com/tal-ai/FG_CME)<br>
**Dataset:** [IEMOCAP](https://sail.usc.edu/iemocap/index.html), [RAVDESS](https://smartlaboratory.org/ravdess/)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/FG_CME.png?raw=true)

---
### SER Using Wav2vec 2.0 Embeddings
**Paper:** [Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings](https://arxiv.org/abs/2104.03502)<br>
**Code:** [habla-liaa/ser-with-w2v2](https://github.com/habla-liaa/ser-with-w2v2)<br>
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/52a69f1e4bcf5043b51b79fddb6ae0b285e5d7c1/2-Figure1-1.png)

---
### Few-shot Learning in Emotion Recognition of Spontaneous Speech
**Paper:** [Few-shot Learning in Emotion Recognition of Spontaneous Speech Using a Siamese Neural Network with Adaptive Sample Pair Formation](https://arxiv.org/abs/2109.02915)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/FewShots_SER_models.png?raw=true)

---
### Fixed-MAML for Few-Shot Multilingual SER
**Paper:** [Fixed-MAML for Few Shot Classification in Multilingual Speech Emotion Recognition](https://arxiv.org/abs/2101.01356)<br>
**Code:** [Fixed-MAML](https://github.com/AnugunjNaman/Fixed-MAML)<br>
**Dataset:** [EmoFilm - A multilingual emotional speech corpus](https://zenodo.org/record/1326428#.YerzWIRBxH5)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/F-MAML_SER_spectrogram.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/F-MAML_SER_dataset_detail.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/F-MAML_SER_accuracy_in_few_shots.png?raw=true)

---
## Multimodal Sentiment Analysis
### MISA
**Paper:** [MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis](https://arxiv.org/abs/2005.03545)<br>

---
### Transformer-based joint-encoding
**Paper:** [A Transformer-based joint-encoding for Emotion Recognition and Sentiment Analysis](https://arxiv.org/abs/2006.15955)<br>
**Code:** [bdel/MOSEI_UMONS](https://github.com/jbdel/MOSEI_UMONS)<br>

Multimodal Transformer Encoder for two modalities with joint-encoding
![](https://d3i71xaburhd42.cloudfront.net/54a2becf5824274588ab3a9c30654604b77b1f27/3-Figure2-1.png)

---
### Multimodal Fusion
**Paper:** [Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis
](https://arxiv.org/abs/2109.00412)<br>
**Code:** [Multimodal Deep Learning](https://github.com/declare-lab/multimodal-deep-learning)<br>
![](https://github.com/declare-lab/Multimodal-Infomax/raw/main/img/ModelFigSingle.png?raw=true)

### Bi-Bimodal Modality Fusion
**Paper:** [Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis](https://arxiv.org/abs/2107.13669)<br>
**Code:** [declare-lab/BBFN](https://github.com/declare-lab/BBFN)<br>
![](https://github.com/declare-lab/BBFN/raw/main/img/model2.png?raw=true)

---
### K-EmoCon
**Paper:** [K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations](https://arxiv.org/abs/2005.04120)<br>
![](https://www.researchgate.net/profile/Cheul-Young-Park/publication/341284068/figure/fig2/AS:889852312707075@1589168732684/Frontal-view-of-a-participant-equipped-with-wearable-sensors.ppm)

---
### CycleEmotionGAN++ 
**Paper:** [Emotional Semantics-Preserved and Feature-Aligned CycleGAN for Visual Emotion Adaptation](https://arxiv.org/abs/2011.12470)<br>
![](https://d3i71xaburhd42.cloudfront.net/4dfd5a28fb750976979fb8168298e766967af88a/4-Figure3-1.png)

---
## Myers–Briggs Type Indicator (MBTI)
[Myers–Briggs Type Indicator (MBTI) Assignment](https://ilearn.laccd.edu/courses/137336/assignments/3011240)<br>
![](https://ilearn.laccd.edu/courses/137336/files/19180085/download?verifier=V2KCTBhXILbuaYVQTg3wypsIlYfoyl9dZgTJLYhl)

**[An Overview of the Myers-Briggs Type Indicator](https://www.verywellmind.com/the-myers-briggs-type-indicator-2795583)**<br>

MBTI scales:<br>
* Extraversion (E) – Introversion (I) 外向型 vs 內向型
* Sensing (S) – Intuition (N) 實感型 vs 直覺型
* Thinking (T) – Feeling (F) 思考型 vs 情感型
* Judging (J) – Perceiving (P) 判斷型 vs 感覺型

The MBTI Types:<br>
* ISTJ - The Inspector  
* ISTP - The Crafter 
* ISFJ - The Protector
* ISFP - The Artist 
* INFJ - The Advocate
* INFP - The Mediator
* INTJ - The Architect
* INTP - The Thinker
* ESTP - The Persuader
* ESTJ - The Director
* ESFP - The Performer
* ESFJ - The Caregiver
* ENFP - The Champion
* ENFJ - The Giver
* ENTP - The Debater
* ENTJ - The Commander

---
### Predicting MBTI with RNN
**Dataset:** [(MBTI) Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasnaek/mbti-type)<br>
**Kaggle:** [mbti-lstm](https://www.kaggle.com/rkuo2000/mbti-lstm)<br>

---
### Personality Analyzer
**Code:** [personality prediction from text](https://github.com/jcl132/personality-prediction-from-text)<br>
Using FB webscraper, based on RandomForestClassifier & TfidfVectorizer<br>
![](https://github.com/jcl132/personality-prediction-from-text/raw/master/static/My_Network.gif)
* Install MongoDB
* Scrape friends info from your FB account (fb_webscraper.py)
* Train model (model.py)
* Make Prediction (predict.py)

---
## Behavior Prediction

### Understanding Consumer Behavior with RNN
**Paper:** [Understanding Consumer Behavior with Recurrent Neural Networks](https://doogkong.github.io/2017/papers/paper2.pdf)<br>
![](https://d3i71xaburhd42.cloudfront.net/36f83b69154ccf37261d86959b80ecb404e08b3f/1-Figure1-1.png)

---
### CPC-18
**Paper:** [Predicting human decisions with behavioral theories and machine learning](https://arxiv.org/abs/1904.06866)<br>
**Raw Data:** [All raw data for CPC18](https://zenodo.org/record/2571510#.YesJ6oRBxH4)<br>
**Code:** [CPC-18 baseline models and source code](https://cpc-18.com/baseline-models-and-source-code/)<br>

---
### MLP + Cognitive Prior
**Paper:** [Cognitive Model Priors for Predicting Human Decisions](https://arxiv.org/abs/1905.09397)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CPC18_benchmarks.png?raw=true)

---
### Predicting human decision making in psychological tasks
**Paper:** [Predicting human decision making in psychological tasks with recurrent neural networks](https://arxiv.org/abs/2010.11413)<br>
**Code:** [HumanLSTM](https://github.com/doerlbh/HumanLSTM)<br>

---
## Social Simulation
* **System-level simulations (SLS)** - Explore a given situation as a whole and how individuals and groups respond to the presence of certain variables.
* **System-level modeling (SLM)** - Creates a more complex and sophisticated environment and aims to be able to make predictions about the behavior of any individual entity or thing within the simulation.
* **Agent-based social simulation(ABSS)** - Models societies on intelligent agents and studies their behavior within the simulated environment for the application to real-world situations.
* **Agent-based modeling (ABM)** - Involves independent agents with individual-specific behaviors interacting in networks.

---
### Neural MMO
**Blog:** [OpenAI 打造 Neural MMO 遊戲，觀察 AI 在複雜開放世界中表現](https://technews.tw/2019/03/07/openai-neural-mmo/)<br>
**Paper:** [The Neural MMO Platform for Massively Multiagent Research](https://arxiv.org/abs/2110.07594)<br>
**[UserGuide](https://neuralmmo.github.io/build/html/rst/userguide.html)**<br>
<iframe width="640" height="360" src="https://www.youtube.com/embed/hYYA8_wFF7Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Social Influence
**Paper:** [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676)<br>
**Paper:** [Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/1810.08647)<br>
**Paper:** [TarMAC: Targeted Multi-Agent Communication](https://arxiv.org/abs/1810.11187)<br>
**Blog:** [About communication in Multi-Agent Reinforcement Learning](https://gema-parreno-piqueras.medium.com/marl-icml-2019-a3cda00d8fff)<br>
<iframe width="665" height="382" src="https://www.youtube.com/embed/iH_V5WKQxmo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://miro.medium.com/max/557/1*3yGfiWLv8JTnCkcgljwbxw.png)
**Code:** [Sequential Social Dilemma Games](https://github.com/eugenevinitsky/sequential_social_dilemma_games)<br>

---
## Theory of Mind
*Theory of Mind: refers to the mental capacity to understand other people and their behavior by ascribing mental states to them*
**Paper:** [Mindreaders: the cognitive basis of theory of mind](https://www.researchgate.net/publication/263724918_Ian_Apperly_Mindreaders_the_cognitive_basis_of_theory_of_mind)<br>
**Blog:** [How the Theory of Mind Helps Us Understand Others](https://www.verywellmind.com/theory-of-mind-4176826)<br>
One study found that children typically progress through five different theory of mind abilities in sequential, standard order.[6](https://srcd.onlinelibrary.wiley.com/doi/10.1111/j.1467-8624.2011.01583.x)<br>
Tasks Listed From Easiest to Most Difficult:<br>
* The understanding that the reasons why people might want something (i.e. desires) may differ from one person to the next
* The understanding that people can have different beliefs about the same thing or situation
* The understanding that people may not comprehend or have the knowledge that something is true
* The understanding that people can hold false beliefs about the world
* The understanding that people can have hidden emotions, or that they may act one way while feeling another way

---
### Machine Theory of Mind
**Paper:** [Machine Theory of Mind](https://arxiv.org/abs/1802.07740)<br>
**Code:** [VArdulov/ToMNet](https://github.com/VArdulov/ToMNet)<br>

---
### APES
**Paper:** [APES: a Python toolbox for simulating reinforcement learning environments](https://arxiv.org/abs/1808.10692)<br>
**Github:** [Artificial Primate Environment Simulator (APES)](https://github.com/aqeel13932/APES)<br>
**Code:** [APES-simple](https://www.kaggle.com/rkuo2000/apes-simple)<br>
<iframe width="720" height="240" src="https://www.youtube.com/embed/RmvT_JBuuIQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Perspective Taking
**Paper:** [Perspective Taking in Deep Reinforcement Learning Agents](https://arxiv.org/abs/1907.01851)<br>
*Perspective taking is the ability to look at things from a perspective that differs from our own [15].*<br> 
*It could be defined as "the cognitive capacity to consider the world from another individual’s viewpoint"*<br>
![](https://d3i71xaburhd42.cloudfront.net/e1dc8a1a368527b213a5d4947fb25bb5f230ebdb/2-Figure1-1.png)
(a) Overview of the simulation environment and visual encodings. The artificial monkey with green circle
is the subordinate agent and the one with red circle is the dominant agent.<br>
(b) Two examples of a subordinate agent goal-oriented behavior as driven by our neural network controller.
 In the top panel the agent should avoid the food as it is observed by the dominant. In the bottom panel the agent should acquire the food as it is not observed by the dominant. 
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PerspectiveTaking_examples.png?raw=true)
**Code:** [APES_PT](https://github.com/aqeel13932/APES_PT)<br>
**Kaggle:** [rkuo2000/APES-PT](https://www.kaggle.com/rkuo2000/apes-pt)<br>

---
### Visual Perspective Taking
**Paper:** [Visual Perspective Taking for Opponent Behavior Modeling](https://arxiv.org/abs/2105.05145)<br>
*Visual Perspective Taking (VPT) refers to the ability to estimate other agents' viewpoint from its own observation.*
![](https://www.researchgate.net/profile/Yuhang-Hu-6/publication/351510799/figure/fig1/AS:1022491321454594@1620792336771/Visual-Perspective-Taking-VPT-refers-to-the-ability-to-estimate-other-agents-viewpoint.png)

---
### ToMNet+
**Paper:** [Using Machine Theory of Mind to Learn Agent Social Network Structures from Observed Interactive Behaviors with Targets](http://epa.psy.ntu.edu.tw/documents/2021/Chuang_etal_2021.pdf)<br>
**Code:** [ToMnet+ project](https://github.com/yunshiuan/tomnet-project)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ToMNet_grid_world.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ToMNet_networks.png?raw=true)

---
### Visual Behavior Modelling
**Paper:** [Visual behavior modelling for robotic theory of mind](https://www.nature.com/articles/s41598-020-77918-x)<br>
**Code:** [Visual Behavior Modelling](https://github.com/BoyuanChen/visual_behavior_modeling)<br>
<iframe width="681" height="383" src="https://www.youtube.com/embed/f2U7_jZVxcU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-020-77918-x/MediaObjects/41598_2020_77918_Fig3_HTML.png?as=webp)
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-020-77918-x/MediaObjects/41598_2020_77918_Fig4_HTML.png?as=webp)

---
### ToM for Humanoid Robot
**Paper:** [Theory of Mind for a Humanoid Robot](http://groups.csail.mit.edu/lbr/hrg/2000/Humanoids2000-tom.pdf)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ToM_Humanoid_Baron_Cohen_model.png?raw=true)

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ToM_Humanoid_detectors.png?raw=true)
*The phenomenon of habituation has been described generally as decrement in response to repeated stimulation (Harris 1943)*

---
### CX-ToM
**Paper:** [CX-ToM: Counterfactual Explanations with Theory-of-Mind for Enhancing Human Trust in Image Recognition Models](https://arxiv.org/abs/2109.01401)<br>
![](https://www.researchgate.net/publication/354379189/figure/fig2/AS:1064895177031680@1630902203407/Example-of-a-ToM-based-Fault-Line-Selection-Process-The-interaction-is-conducted-through.png)

---
## Brain Model
### Brain-like DNN
**Paper:** [Brain hierarchy score: Which deep neural networks are hierarchically brain-like?](https://www.sciencedirect.com/science/article/pii/S2589004221009810)<br>
![](https://ars.els-cdn.com/content/image/1-s2.0-S2589004221009810-fx1_lrg.jpg)

---
### DNNBrain
**Paper:** [DNNBrain: A Unifying Toolbox for Mapping Deep Neural Networks and Brains](https://www.frontiersin.org/articles/10.3389/fncom.2020.580632/full)<br>
**Code:**[BNUCNL/DNNBrain](https://github.com/BNUCNL/dnnbrain)<br>
DNN Model: AlexNet
![](https://www.frontiersin.org/files/Articles/580632/fncom-14-580632-HTML/image_m/fncom-14-580632-g003.jpg)
![](https://www.frontiersin.org/files/Articles/580632/fncom-14-580632-HTML/image_m/fncom-14-580632-g006.jpg)
![](https://www.frontiersin.org/files/Articles/580632/fncom-14-580632-HTML/image_m/fncom-14-580632-g007.jpg)

---
### GNN
**Paper:** [Graph Neural Networks in Network Neuroscience](https://arxiv.org/abs/2106.03535)<br>
![](https://d3i71xaburhd42.cloudfront.net/4caa75f18d78c2dcc8166f416b56cd4da28cdb46/14-Figure3-1.png)

---
### BrainGNN
**Paper:** [BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis](https://www.biorxiv.org/content/10.1101/2020.05.16.100057v1.full)<br>
![](https://www.biorxiv.org/content/biorxiv/early/2020/05/17/2020.05.16.100057/F1.large.jpg)
![](https://www.biorxiv.org/content/biorxiv/early/2020/05/17/2020.05.16.100057/F9.medium.gif)

---
### DMBN
**Paper:** [Deep Representation Learning For Multimodal Brain Networks](https://arxiv.org/abs/2007.09777)<br>
![](https://d3i71xaburhd42.cloudfront.net/99d305edc3eefabd76c3e6b70adeb1edad88e4c3/5-Figure2-1.png)

---
### BrainNNExplainer
**Paper:** [BrainNNExplainer: An Interpretable Graph Neural Network Framework for Brain Network based Disease Analysis](https://arxiv.org/abs/2107.05097)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/BrainNNExplainer.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/BrainNNExplainer_comparison_of_explanation_graph_connectomes.png?raw=true)

---
### Multi-GCN
**Paper:** [Multiplex Graph Networks for Multimodal Brain Network Analysis](https://arxiv.org/abs/2108.00158)<br>
![](https://images.deepai.org/converted-papers/2108.00158/Figs/Fig_framework_MGNet_NEW5.png)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Multi-GCNs_comparison.png?raw=true)

---
## Learning
### Self-supervised Learning
**Paper:** [Self-supervised learning through the eyes of a child](https://arxiv.org/abs/2007.16189)<br>
**Dataset:** [SAYCam: A large, longitudinal audiovisual dataset recorded from the infant’s perspective](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00039/97495/SAYCam-A-Large-Longitudinal-Audiovisual-Dataset)<br>
**Code:** [eminorhan/baby-vision](https://github.com/eminorhan/baby-vision)<br>
**[databrary](https://nyu.databrary.org/)**<br>
![](https://media.arxiv-vanity.com/render-output/5548788/x1.png)
![](https://media.arxiv-vanity.com/render-output/5548788/x4.png)

---
### Generalized Schema Learning
**Paper:** [Learning to perform role-filler binding with schematic knowledge](https://arxiv.org/abs/1902.09006)<br>
**Code:** [cchen23/generalized_schema_learning](https://github.com/cchen23/generalized_schema_learning)<br>
![](https://dfzljdn9uc3pi.cloudfront.net/2021/11046/1/fig-1-2x.jpg)
**[Coffee shop World](https://github.com/PrincetonCompMemLab/narrative)**<br>
*The "engine" takes a schema and generates a bunch of stories!*<br>
`python run_engine.py poetry fight 2 2`<br>

---
### Neural Model of Schemas and Memory
**Paper:** [A Neural Model of Schemas and Memory Consolidation](https://www.biorxiv.org/content/10.1101/434696v1.full)<br>
![](https://www.biorxiv.org/content/biorxiv/early/2018/10/04/434696/F1.medium.gif)
![](https://www.biorxiv.org/content/biorxiv/early/2018/10/04/434696/F2.medium.gif)

---
## Episodic Memory
### Episodic Memory Reader
**Paper:** [Episodic Memory Reader: Learning What to Remember for Question Answering from Streaming Data](https://arxiv.org/abs/1903.06164)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EMR_overview.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EMR_decoder.png?raw=true)

---
### Episodic Memory in Lifelong Language Learning
**Paper:** [Episodic Memory in Lifelong Language Learning](https://arxiv.org/abs/1906.01076)<br>
**Code:** [episodic-lifelong-learning](https://github.com/h3lio5/episodic-lifelong-learning)<br>
![](https://raw.githubusercontent.com/h3lio5/episodic-lifelong-learning/master/images/train_infer_new.png)

---
### Replay Episodic Memory
**Paper:** [DRILL: Dynamic Representations for Imbalanced Lifelong Learning](https://arxiv.org/abs/2105.08445)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/DRILL_overview.png?raw=true)

---
## Semantic Memory
**Paper:** [Semantic Memory: A Review of Methods, Models, and Current Challenges](https://www.researchgate.net/publication/343982105_Semantic_Memory_A_Review_of_Methods_Models_and_Current_Challenges)<br>
* Semantic Memory Representation
  - Network-based Approaches
  - Feature-based Approaches
  - Distributional Approaches : Distributional Semantic Models (**DSMs**)
* Semantic Memory Learning
  - Error-free Learning-based DSMs
  - Error-driven Learning-based DSMs: **word2vec**
* Contextual and Retrieval-based Semantic Memory
  - Ambiguity Resolution in Error-free Learning-based DSMs
  - Ambiguity Resolution in Predictive DSMs: **ELMo**, **BERT**, **GPT-3**
* Retrieval-based Models of Semantic Memory

### Meta-Learning with Variational Semantic Memory
**Paper:** [Meta-Learning with Variational Semantic Memory for Word Sense Disambiguation](https://arxiv.org/abs/2106.02960)<br>
**Code:** [VSM_WSD](https://github.com/YDU-uva/VSM)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/VSM_computational_graph.png?raw=true)

---
### Growing Dual-Memory (GDM)
**Paper:** [Lifelong Learning of Spatiotemporal Representations with Dual-Memory Recurrent Self-Organization](https://arxiv.org/abs/1805.10966)<br>
**Code:** [GDM: Growing Dual-Memory Self-Organizing Networks](https://github.com/giparisi/GDM)<br>
**Dataset:** [Iris Species](https://www.kaggle.com/uciml/iris)<br>
![](https://www.frontiersin.org/files/Articles/401624/fnbot-12-00078-HTML/image_m/fnbot-12-00078-g001.jpg)

---
### CAT
**Paper:** [Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](https://arxiv.org/abs/2112.10017)<br>
**Code:** [CAT (Continual learning with forgetting Avoidance and knowledge Transfer)](https://github.com/ZixuanKe/CAT)<br>
![](https://github.com/ZixuanKe/CAT/raw/main/CAT.png)

---
## [The Emotion Machine](https://drive.google.com/file/d/0BxwvD5jbCicTZ0UzVGxXbVI2Y2M/view?resourcekey=0-Q1ko9cRcptzWWYqXqZfZgg)
[Video Lectures](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-868j-the-society-of-mind-fall-2011/video-lectures/) by Marvin Minsky<br>

**Six-level Model of Mind**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_six-level_model.png?raw=true)

**Attachments and Goals**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_Attachments_and_Goals.png?raw=true)

**From Pain to Suffering**<br>
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_critic-selector-based_machine.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_critic_selecting_resources.png?raw=true"></td>
</tr>
</table>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_spbbading_cascade.png?raw=true)

**Consciousness**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_trouble-detecting_critic.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_Immanence_Illusigjv.png?raw=true)

**Self**<br>
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_multiple_models_of_self.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EmotionMachine_is_mind_a_human_community.png?raw=true"></td>
</tr>
</table>

---
### Enhancing Cognitive Models of Emotions with Representation Learning
**Paper:** [Enhancing Cognitive Models of Emotions with Representation Learning](https://arxiv.org/abs/2104.10117)<br>
**Code:** [emorynlp/CMCL-2021](https://github.com/emorynlp/CMCL-2021)<br>
![](https://www.researchgate.net/publication/351019526/figure/fig1/AS:1014977343283201@1619000864697/Emotion-wheel-auto-derived-by-our-approach.png)
The 2D plot from the PAD values of 32 emotions predicted by regression models
![](https://www.researchgate.net/publication/351019526/figure/fig2/AS:1014977343258626@1619000864726/The-2D-plot-from-the-PAD-values-of-32-emotions-predicted-by-our-regression-models.png)

Emotion wheel proposed by Plutchik (1980).
![](https://www.researchgate.net/publication/351019526/figure/fig3/AS:1014977343262722@1619000864753/Emotion-wheel-proposed-by-Plutchik-1980.png)

---
### Machine Consciousness Architecture
**Paper:** [A Machine Consciousness architecture based on Deep Learning and Gaussian Processes](https://arxiv.org/abs/2002.00509)<br>
![](https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-61705-9_29/MediaObjects/500677_1_En_29_Fig3_HTML.png)

---
## Cognitive Architecture

### Cognitive Psychology
**Paper:** [Cognitive Psychology for Deep Neural Networks: A Shape Bias Case Study](https://arxiv.org/abs/1706.08606)<br>
![](https://d3i71xaburhd42.cloudfront.net/39fb9fa2615620f043084a2ecbbdb1a1f8c707c9/5-Figure1-1.png)
---
**Paper:** [The Cognitive Structure of Emotion]()<br>

---
### MECA
**Paper:** [An Overview of the Multipurpose Enhanced Cognitive Architecture (MECA)](https://www.sciencedirect.com/science/article/pii/S1877050918300267)<br>
![](https://www.researchgate.net/profile/Ricardo-Gudwin/publication/320342187/figure/fig1/AS:614298002927619@1523471465465/An-Overview-of-the-MECA-Cognitive-Architecture.png)

---
### Whole-Brain Probabilistic Generative Model (WB-PGM)
**Paper:** [Hippocampal formation-inspired probabilistic generative model](https://arxiv.org/abs/2103.07356)<br>
![](https://wba-initiative.org/wp-content/uploads/2021/03/9e614dedda0c6b2906da170aae8397a8.jpg)

**Paper:** [The whole brain architecture approach: Accelerating the development of artificial general intelligence by referring to the brain](https://arxiv.org/abs/2103.06123)<br>
![](https://ars.els-cdn.com/content/image/1-s2.0-S0893608021003543-gr1.jpg)
![](https://ars.els-cdn.com/content/image/1-s2.0-S0893608021003543-gr5.jpg)

**Paper:** [A Whole Brain Probabilistic Generative Model: Toward Realizing Cognitive Architectures for Developmental Robots](https://arxiv.org/abs/2103.08183)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/WB-PGM.png?raw=true)

---
### Integrated Cognitive Model
**Paper:** [Integrated Cognitive Architecture for Robot Learning of Action and Language](https://www.frontiersin.org/articles/10.3389/frobt.2019.00131/full)<br>
![](https://www.ncbi.nlm.nih.gov/pmc/articles/instance/7805838/bin/frobt-06-00131-g0001.jpg)
![](https://www.ncbi.nlm.nih.gov/pmc/articles/instance/7805838/bin/frobt-06-00131-g0003.jpg)
![](https://www.ncbi.nlm.nih.gov/pmc/articles/instance/7805838/bin/frobt-06-00131-g0006.jpg)

---
### Integrated PGM
**Paper:** [Neuro-SERKET: Development of Integrative Cognitive System through the Composition of Deep Probabilistic Generative Models](https://arxiv.org/abs/1910.08918)<br>
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs00354-019-00084-w/MediaObjects/354_2019_84_Fig2_HTML.png?as=webp)
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs00354-019-00084-w/MediaObjects/354_2019_84_Fig5_HTML.png?as=webp)

---
### Analogical Concept Memory
**Paper:** [Characterizing an Analogical Concept Memory for Architectures Implementing the Common Model of Cognition](https://arxiv.org/abs/2006.01962)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Advanced_Cognitive_Learning_for_embodied_comprehension.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Analogical_Concept_Memory.png?raw=true)

---
### A review of possible effects of cognitive biases
**Paper:** [A review of possible effects of cognitive biases on interpretation of rule-based machine learning models](https://www.sciencedirect.com/science/article/pii/S0004370221000096#fg0040)<br>
![](https://ars.els-cdn.com/content/image/1-s2.0-S0004370221000096-gr004.jpg)

---
### Spatial Navigation
**Paper:** [The Cognitive Architecture of Spatial Navigation: Hippocampal and Striatal Contributions](https://www.sciencedirect.com/science/article/pii/S0896627315007783)<br>
<img width="70%" height="70%" src="https://ars.els-cdn.com/content/image/1-s2.0-S0896627315007783-gr1_lrg.jpg">
 A Minimal Cognitive Architecture for Spatial Navigation
![](https://ars.els-cdn.com/content/image/1-s2.0-S0896627315007783-gr4.jpg)
![](https://ars.els-cdn.com/content/image/1-s2.0-S0896627315007783-gr3.jpg)

---
### Patterns of Cognitive Appraisal in Emotion
**Paper:** [Patterns of Cognitive Appraisal in Emotion](https://www.researchgate.net/publication/19274815_Patterns_of_Cognitive_Appraisal_in_Emotion)<br>

---
### Fusion Architecture for Learning and Cognition 
**Paper:** [Fusion Architecture for Learning and COgnition (FALCON)]()
FALCON is a three-channel fusion Adaptive Resonance Theory (ART) network
* Imitative learning
* Reinforcement learning
* Dual-Stage learning 

### CRAA
The Cognitive Regulated Affective Architecture 

### Affective User Model in Behaviroal Health
The Case for Cognitive-Affective Architectures as Affective User Models in Behavioral Health Technologies

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


