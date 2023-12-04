---
layout: post
title: Recurrent Neural Networks
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes RNN, LSTM, Transformers, GPT.

---
## RNN (循環神經網路) Introduction

### Recurrent Neural Networks (RNN)
**Blog:** [循環神經網路(Recurrent Neural Network, RNN)](https://ithelp.ithome.com.tw/articles/10193469)<br>
![](https://ithelp.ithome.com.tw/upload/images/20171210/20001976y5kxBTjmM7.jpg)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RNN_cell_model.png?raw=true)

**Ref.**[深度學習(3)--循環神經網絡(RNN, Recurrent Neural Networks)](https://arbu00.blogspot.com/2017/05/3-rnn-recurrent-neural-networks.html)<br>
![](https://4.bp.blogspot.com/-U-b50pPNd_s/WQ8C-5g0siI/AAAAAAAAI8c/VCpWPA-z3Y8iTvLIhX6hlO83BqogFMu5ACLcB/s640/RNN2.png)
![](https://3.bp.blogspot.com/-9cz6YIf-3Wk/WQ8C-7QNnOI/AAAAAAAAI8g/iFhXR9t3ii0UE9ZRXs425wR_HYJk9i7WgCLcB/s640/RNN1.png)
![](https://4.bp.blogspot.com/-xgt_nqA75Kw/WQ8iY0-73xI/AAAAAAAAI84/7lPr7Xwbt9cCHBpIYaHN2E6J7UsXNCSQgCEw/s1600/qr1.png)
![](https://2.bp.blogspot.com/-DAFSVtTbvjE/WQ8iYztOuEI/AAAAAAAAI88/JubT4oji4FIw7Tnvl4occhiQxykuNITWwCEw/s1600/qr2.png)

**Ref.**[IBM Recurrent Neural Networks](https://www.ibm.com/cloud/learn/recurrent-neural-networks)<br>
<table>
<tr>
<td><p align="center">one-to-one</p><img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_13C-RecurrentNeuralNetworks-WHITEBG.png"></td>
<td><p align="center">one-to-many</p><img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_13D-RecurrentNeuralNetworks-WHITEBG.png"></td>
<td><p align="center">many-to-one</p><img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_13E-RecurrentNeuralNetworks-WHITEBG.png"></td>
<td><p align="center">many-to-many</p><img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_13F-RecurrentNeuralNetworks-WHITEBG.png"></td>
<td><p align="center">many-to-many</p><img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_13G-RecurrentNeuralNetworks-WHITEBG.png"></td>
</tr>
</table>

### Long Short Term Memory (LSTM)
**Blog:** [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)<br>

![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

---
### Timeseries Weather Forecast
**Blog:** [Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)<br>
**Kaggle:** [rkuo2000/weather-lstm](https://www.kaggle.com/rkuo2000/weather-lstm)<br>
![](https://keras.io/img/examples/timeseries/timeseries_weather_forecasting/timeseries_weather_forecasting_6_1.png)
<table>
<tr>
<td><img src="https://keras.io/img/examples/timeseries/timeseries_weather_forecasting/timeseries_weather_forecasting_26_0.png"></td>
<td><img src="https://keras.io/img/examples/timeseries/timeseries_weather_forecasting/timeseries_weather_forecasting_26_1.png"></td>
<td><img src="https://keras.io/img/examples/timeseries/timeseries_weather_forecasting/timeseries_weather_forecasting_26_2.png"></td>
</tr>
<tr>
<td><img src="https://keras.io/img/examples/timeseries/timeseries_weather_forecasting/timeseries_weather_forecasting_26_3.png"></td>
<td><img src="https://keras.io/img/examples/timeseries/timeseries_weather_forecasting/timeseries_weather_forecasting_26_4.png"></td>
</tr>
</table>

---
### Stock Price Forecast
**Paper:** [Stock price forecast with deep learning](https://arxiv.org/abs/2103.14081)<br>
The input layer represents index value and volume from the previous **14** trading days
while the output layer represents the predicted index value on the next day. **rnn2** produces the lowest test error (MAE).
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_Price_Forecast_models.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_Price_Forecast_MAE.png?raw=true"></td>
</tr>
</table>

**Code:** [rkuo2000/stock-lstm](https://kaggle.com/rkuo2000/stock-lstm)<br>
```
model = Sequential()
model.add(Input(shape=(50, 5)))
model.add(LSTM(50))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_LSTM.png?raw=true)

**Code:** [rkuo2000/stock-cnn-lstm](https://kaggle.com/rkuo2000/stock-cnn-lstm)<br>
```
model = Sequential()
model.add(Input(shape=(50, 5)))
model.add(Conv1D(50, 5, activation='sigmoid'))
#model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_CNN_LSTM.png?raw=true)

**Code:** [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models)<br>
![](https://github.com/huseinzol05/Stock-Prediction-Models/raw/master/output/evolution-strategy.png)

---
### GPS+IMU Fusion
**Paper:** [IMU Data and GPS Position Information Direct Fusion Based on LSTM](https://www.mdpi.com/1424-8220/21/7/2500)<br>
![](https://www.mdpi.com/sensors/sensors-21-02500/article_deploy/html/images/sensors-21-02500-g007.png)
![](https://www.mdpi.com/sensors/sensors-21-02500/article_deploy/html/images/sensors-21-02500-g001-550.jpg)
![](https://www.mdpi.com/sensors/sensors-21-02500/article_deploy/html/images/sensors-21-02500-g006.png)
![](https://www.mdpi.com/sensors/sensors-21-02500/article_deploy/html/images/sensors-21-02500-g008.png)

---
### Urban Traffic Flow Prediction by Graph Convolutional Network
**Paper:** [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320)<br>
**Code:** [lehaifeng/T-GCN](https://github.com/lehaifeng/T-GCN)<br>
**Blog:** [T-GCN導讀](https://iter01.com/512968.html)

![](https://i.iter01.com/images/e69f0018c17c67939695fb91e6803138a2c95fa744297670daaa2610f0a8c69b.png)

**paper:** [A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://arxiv.org/abs/2006.11583)<br>

![](https://www.mdpi.com/ijgi/ijgi-10-00485/article_deploy/html/images/ijgi-10-00485-g001-550.jpg)

**Paper:** [AST-GCN: Attribute-Augmented Spatiotemporal Graph Convolutional Network for Traffic Forecasting](https://arxiv.org/abs/2011.11004)<br>
**Code:** [guoshnBJTU/ASTGCN-r-pytorch](https://github.com/guoshnBJTU/ASTGCN-r-pytorch)<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AST-GCN.png?raw=true)

---
### WLASL
**Paper:** [Word-level Deep Sign Language Recognition from Video](https://arxiv.org/abs/1910.11006)<br>
**Code:** [dxli94/WLASL](https://github.com/dxli94/WLASL)<br>
<iframe width="823" height="310" src="https://www.youtube.com/embed/wG-uaee4mJ4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Word Embedding
**Ref.** [Unsupervised Learning - Word Embedding](https://hackmd.io/@shaoeChen/B1CoXxvmm/https%3A%2F%2Fhackmd.io%2Fs%2Fr1iKfZu2Q)<br>

![](https://i.imgur.com/J2n4MtL.png)
![](https://i.imgur.com/0ouCbt5.png)
![](https://i.imgur.com/9fBXOpD.png)

---
### Sentiment Analysis
**Dataset:** [First GOP Debate Twitter Sentiment](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)<br>
**Code:** [Sentiment-LSTM](https://www.kaggle.com/rkuo2000/sentiment-lstm) vs [Sentiment-NLTK](https://www.kaggle.com/rkuo2000/sentiment-nltk)<br>
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-pos.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-neg.png?raw=true"></td>
</tr>
</table>


```
embed_dim = 128
units = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
```

---
## Neural Machine Translation 

### seq2seq
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/01/architecture.png)
![](https://miro.medium.com/max/1400/1*x4wsJobiSC7zlTkP8yv40A.png)
<p><img src="https://i.stack.imgur.com/YjlBt.png" width="50%" height="50%"></p>

```
model = Sequential()
model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
model.add(LSTM(units))
model.add(RepeatVector(out_timesteps))
model.add(LSTM(units, return_sequences=True))
model.add(Dense(out_vocab, activation='softmax'))
```

---
### Neural Decipher
![](https://upload.wikimedia.org/wikipedia/commons/0/04/22_alphabet.jpg)
Ugaritic is an extinct Northwest Semitic language, classified by some as a dialect of the Amorite language and so the only known Amorite dialect preserved in writing.
<img width="50%" height="50%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Ugaritic_Chart_of_Letters.svg/800px-Ugaritic_Chart_of_Letters.svg.png">

**Paper:** [Neural Decipherment via Minimum-Cost Flow: from Ugaritic to Linear B](https://arxiv.org/abs/1906.06718)<br>
**Code:** [j-luo93/NeuroDecipher](https://github.com/j-luo93/NeuroDecipher)<br>
![](https://media.arxiv-vanity.com/render-output/4869057/x1.png)

---
### ParlAI
**Paper:** [ParlAI: A Dialog Research Software Platform](https://arxiv.org/abs/1705.06476)<br>
**Code:** [facebookresearch/ParlAI](https://github.com/facebookresearch/ParlAI)<br>
![](https://raw.githubusercontent.com/facebookresearch/ParlAI/main/docs/source/_static/img/parlai_example.png)

---
### Hand Writing Synthesis
**Paper:** [Generating Sequences With Recurrent Neural Networks]()<br>
**Code:** [Grzego/handwriting-generation](https://github.com/Grzego/handwriting-generation)<br>
![](https://github.com/Grzego/handwriting-generation/blob/master/imgs/example-2.gif?raw=true)

---
## Transformer
**Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>
**Code:** [huggingface/transformers](https://github.com/huggingface/transformers)<br>
![](https://miro.medium.com/max/407/1*3pxDWM3c1R_WSW7hVKoaRA.png)
<table>
<tr>
<td><iframe width="400" height="300" src="https://www.youtube.com/embed/n9TlOhRjYoc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
<td><iframe width="400" height="300" src="https://www.youtube.com/embed/N6aRv06iv2g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></td>
</tr>
</table>

---
### New Understanding about Transformer
**Blog:** <br>
* [Researchers Gain New Understanding From Simple AI](https://www.quantamagazine.org/researchers-glimpse-how-ai-gets-so-good-at-language-processing-20220414/)
* [Transformer稱霸的原因找到了？OpenAI前核心員工揭開注意力頭協同工作機理](https://bangqu.com/A76oX7.html)

**Papers:**<br>
* [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
* [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

---
## Self-supervised Learning
**Blog:** [自督導式學習 (Self-supervised Learning)](https://medium.com/@ze_lig/%E8%87%AA%E7%9D%A3%E5%B0%8E%E5%BC%8F%E5%AD%B8%E7%BF%92-self-supervised-learning-%E6%9D%8E%E5%BC%98%E6%AF%85-ml2021-8-51b5c9fde97)<br>
![](https://miro.medium.com/max/1400/1*nlHlqcwHL193X4va0JDpkg.png)
模型參數比例比較: <br>
* **ELMO** (94M)
* **BERT** (340M)
* **GPT-2** (1542M)
* **Megatron** (8B)
* **T5** (11B)
* **Turing-NRG** (17B)
* **GPT-3** (175B)
* **Megatron-Turing NRG** (530B)
* **Switch Transfer** (1.6T)

![](https://leemeng.tw/images/gpt2/elmo-bert-gpt2.jpg)
![](https://leemeng.tw/images/bert/bert_elmo_gpt.jpg)

---
### BERT
**Paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)<br>
**Blog:** [進擊的BERT：NLP 界的巨人之力與遷移學習](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)<br>
![](https://leemeng.tw/images/bert/bert-intro.jpg)
![](https://leemeng.tw/images/bert/bert-2phase.jpg)
<iframe width="640" height="480" src="https://www.youtube.com/embed/gh0hewYkjgo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://leemeng.tw/images/bert/bert-pretrain-tasks.jpg)
![](https://leemeng.tw/images/bert/bert_fine_tuning_tasks.jpg)
<iframe width="640" height="480" src="https://www.youtube.com/embed/ExXA05i8DEQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### GPT (Generative Pre-Training Transformer)
**Paper:** [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)<br>
**Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)<br>
**Code:** [openai/gpt-2](https://github.com/openai/gpt-2)<br>

<iframe width="640" height="480" src="https://www.youtube.com/embed/WY_E0Sd4K80" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**[Transformer Demo](https://app.inferkit.com/demo)**<br>
**[GPT-2 small](https://minimaxir.com/apps/gpt2-small/)**<br>

<iframe width="666" height="300" src="https://www.youtube.com/embed/jz78fSnBG0s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**Blog:** [直觀理解GPT2語言模型並生成金庸武俠小說](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)<br>
![](https://leemeng.tw/images/gpt2/gpt2-model-comparison.jpg)

---
### Megatron (by Nvidia)
**Paper:** [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)<br>
**Code:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)<br>
**Blog:** [MegatronLM: Training Billion+ Parameter Language Models Using GPU Model Parallelism](https://nv-adlr.github.io/MegatronLM)<br>
![](https://nv-adlr.github.io/images/megatronlm/MLP_SelfAttention.jpg)
![](https://github.com/NVIDIA/Megatron-LM/blob/main/images/cases_april2021.png?raw=true)
![](https://1.bp.blogspot.com/-SllNg6Q4DEE/Xk7ZRCtzXaI/AAAAAAAAFVY/PaaM-FEgyFIdSn7VeT_XhvG9PXQdSC3_wCLcBGAsYHQ/s640/t5-trivia-lrg.gif)

---
### T5: Text-To-Text Transfer Transformer (by Google)
**Paper:** [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)<br>
**Code:** [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)<br>
**Dataset:** [C4](https://www.tensorflow.org/datasets/catalog/c4)<br>
**Blog:** [Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)<br>
![](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s640/image3.gif)
![](https://1.bp.blogspot.com/-89OY3FjN0N0/XlQl4PEYGsI/AAAAAAAAFW4/knj8HFuo48cUFlwCHuU5feQ7yxfsewcAwCLcBGAsYHQ/s640/image2.png)
![](https://1.bp.blogspot.com/-SllNg6Q4DEE/Xk7ZRCtzXaI/AAAAAAAAFVY/PaaM-FEgyFIdSn7VeT_XhvG9PXQdSC3_wCLcBGAsYHQ/s640/t5-trivia-lrg.gif)

---
### Turing-NLG (by Microsoft)
**Blog:** [Turing-NLG: A 17-billion-parameter language model by Microsoft](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)<br>

---
### GPT-3
**Code:** [openai/gpt-3](https://github.com/openai/gpt-3)<br>
**[GPT-3 Demo](https://gpt3demo.com/)**<br>

![](https://dzlab.github.io/assets/2020/07/20200725-gpt3-model-architecture.png)
![](https://i.stack.imgur.com/dthvC.png)

---
### Megatron-Turing NLG (by Nvidia)
**Blog:** [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)<br>

![](https://developer-blogs.nvidia.com/wp-content/uploads/2021/10/Model-Size-Chart.png)

---
### Switch Transformer (by Google)
**Paper:** [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)<br>
*The Switch Transformer is a switch feed-forward neural network (FFN) layer that replaces the standard FFN layer in the transformer architecture.*<br>
![](https://miro.medium.com/max/700/1*QMLNn7AFlwuirryw1Qdpaw.png)

* Switch-XXL : 395 billion parameters
* Switch-C : 1.571 trillion parameters

---
### Gopher
**Blog:** [DeepMind推出2800亿参数模型；剑桥团队首次检测到量子自旋液体](https://posts.careerengine.us/p/61b57eb572cc975e511fbae7)<br>
**Paper:** [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)<br>
* Six models of Transformer:
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Gopher_model_architecture.png?raw=true)

---
### RETRO - Retrieval-Enhanced Transformer (by DeepMind)
**Blog:** [Improving Language Models by Retrieving from Trillions of Tokens](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)<br>
**Paper:** [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/RETRO_architecture.png?raw=true)

---
### Chinchilla (by DeepMind)
**Blog:** [An empirical analysis of compute-optimal large language model training](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training)<br>
**Paper:** [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)<br>
![](https://miro.medium.com/max/700/1*NXYjc3qCfTtp9tB1dU-rIw.png)
![](https://assets-global.website-files.com/621e749a546b7592125f38ed/62557f7626b9e103db549c7b_tokens_vs_flops%20(1).png)
![](https://assets-global.website-files.com/621e749a546b7592125f38ed/62557f43672f48833d2088c1_chinchilla.performance.image.png)

---
### PaLM (by Google)
**Blog:** [Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)<br>
**Paper:** [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)<br>
**Code:** [lucidrains/PaLM-pytorch](https://github.com/lucidrains/PaLM-pytorch)
![](https://github.com/lucidrains/PaLM-pytorch/blob/main/palm.gif?raw=true)

`$ pip install PaLM-pytorch`<br>
**Usage:**
```
import torch
from palm_pytorch import PaLM

palm = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64,
)

tokens = torch.randint(0, 20000, (1, 2048))
logits = palm(tokens) # (1, 2048, 20000)
```

---
### [CKIP Transformers](https://ckip.iis.sinica.edu.tw/)
CKIP (CHINESE KNOWLEDGE AND INFORMATION PROCESSING) 繁體中文詞庫小組
繁體中文的 transformers 模型（包含 ALBERT、BERT、GPT2）及自然語言處理工具。
[CKIP Lab 下載軟體與資源](https://ckip.iis.sinica.edu.tw/resource)<br>
* [CKIP Transformers](https://github.com/ckiplab/ckip-transformers)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CKIP_Transformers_comparison.png?raw=true)
![](https://dreambo4.github.io/2021/09/26/%E8%AA%9E%E6%96%99%E5%BA%AB%E6%A8%A1%E5%9E%8B-04-%E6%96%B7%E8%A9%9E%E5%B7%A5%E5%85%B7%E6%AF%94%E8%BC%83-Jieba-vs-CKIP/%E9%95%B7%E7%85%A7%E6%96%B7%E8%A9%9E.jpg)

* [Demo server](https://ckip.iis.sinica.edu.tw/service/transformers)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CKIP_Transformers_demo.png?raw=true)

---
### Transformers Chatbot
**Code:** [https://github.com/demi6od/ChatBot](https://github.com/demi6od/ChatBot)<br>

**Chatbot based on Transformer and BERT**<br>
![](https://github.com/demi6od/ChatBot/blob/master/image/ChatBotBertTransformer.jpg?raw=true)

**Chatbot based on BERT and GPT**<br>
![](https://github.com/demi6od/ChatBot/raw/master/image/ChatBotBertGPT.jpg)

---
### Instruct GPT
**Paper:** [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)<br>
**Blog:** [Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/)<br>

Methods:<br>
![](https://cdn.openai.com/instruction-following/draft-20220126f/methods.svg)


---
### ChatGPT
[ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)<br>
ChatGPT is fine-tuned from a model in the GPT-3.5 series, which finished training in early 2022.<br>

Methods:<br> 
![](https://cdn.openai.com/chatgpt/draft-20221129c/ChatGPT_Diagram.svg)

[AI機器人Chat GPT爆紅　專家點出重大影響](https://www.digitimes.com.tw/iot/article.asp?id=0000652798_E7C62PT4L9WAVR7E2V4LY)<br>
<iframe width="640" height="455" src="https://www.youtube.com/embed/e0aKI2GGZNg" title="Chat GPT (可能)是怎麼煉成的 - GPT 社會化的過程" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Video Representation Learning
### SimCLR
**Paper:** [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)<br>
**Code:** [google-research/simclr](https://github.com/google-research/simclr)<br>
![](https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s640/image4.gif)

---
### BYOL
**Paper:** [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)<br>
**Code:** [lucidrains/byol-pytorch](https://github.com/lucidrains/byol-pytorch)<br>

![](https://github.com/lucidrains/byol-pytorch/blob/master/diagram.png?raw=true)

---
### VideoMoCo
**Paper:** [VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples](https://arxiv.org/abs/2103.05905)<br>
**Code:** [tinapan-pt/VideoMoCo](https://github.com/tinapan-pt/VideoMoCo)<br>
**Dataset:** [Kinetics400](https://deepmind.com/research/open-source/kinetics), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)<br>
**Blog:** [VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples (CVPR2021)](https://giveme2486.medium.com/videomoco-contrastive-video-representation-learning-with-temporally-adversarial-examples-336a5b5dae8)<br>

![](https://miro.medium.com/max/2000/1*oP_OmgFq3BilRrP8syNw_A.png)

---
### PolyViT
**Blog:** [PolyViT: Co-training Vision Transformers on Images, Videos and Audio](https://towardsdatascience.com/polyvit-co-training-vision-transformers-on-images-videos-and-audio-f5e81bee9491)<br>
**Paper:** [PolyViT: Co-training Vision Transformers on Images, Videos and Audio](https://arxiv.org/abs/2111.12993)<br>
* Architecture
![](https://miro.medium.com/max/1400/1*uXqoUlZA4rnKN7f0Yk1Q8g.jpeg)
* Co-training: Mini batch sampling methods
![](https://miro.medium.com/max/996/1*xvwtAtZbYJokNPz9gO-AlA.jpeg)

---
### Uni-Perceiver
**Paper:** [Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks](https://arxiv.org/abs/2112.01522)<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Uni-Perceiver_architecture.png?raw=true)

---
## Question Answering
### [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) - The Stanford Question Answering Dataset<br>
**Paper:** [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822)<br>

![](https://miro.medium.com/max/1400/1*Tqibs5z0zCntcK6kCpziaA.png)

### IE-Net
**Paper:** [Intra-Ensemble in Neural Networks](https://arxiv.org/abs/1904.04466)<br>

![](https://d3i71xaburhd42.cloudfront.net/b331b55b237e378f5f7a6745617fccd3d3fd32ff/250px/1-Figure1-1.png)

---
### ELECTRA
**Paper:** [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)<br>
![](https://github.com/lucidrains/electra-pytorch/blob/master/electra.png?raw=true)

**Code:** [google-research/electra](https://github.com/google-research/electra)<br>
On Kaggle, using GPU will run out of memory, using TPU or CPU ETA > 2 days

**Code:** [renatoviolin/electra-squad-8GB](https://github.com/renatoviolin/electra-squad-8GB)<br>
Fine tuning Electra large model using RTX2080 8 GB on Squad 2

**Code:** [lucidrains/electra-pytorch](https://github.com/lucidrains/electra-pytorch)<br>

---
### Chinese ELECTRA
**Github:** [ymcui/Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)<br>
* [CMRC 2018 (Cui et al., 2019)：篇章片段抽取型阅读理解（简体中文](https://github.com/ymcui/cmrc2018)
* [DRCD (Shao et al., 2018)：篇章片段抽取型阅读理解（繁体中文）](https://github.com/DRCKnowledgeTeam/DRCD)
* [XNLI (Conneau et al., 2018)：自然语言推断](https://github.com/google-research/bert/blob/master/multilingual.md)
* [ChnSentiCorp：情感分析](https://github.com/pengming617/bert_classification)
* [LCQMC (Liu et al., 2018)：句对匹配](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
* [BQ Corpus (Chen et al., 2018)：句对匹配](http://icrc.hitsz.edu.cn/Article/show/175.html)

---
## Speech Datasets
### General Voice Recognition Datasets
* [The LJ Speech Dataset/](https://keithito.com/LJ-Speech-Dataset/) : This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.
* [The M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
* [Speech Accent Archive](https://www.kaggle.com/rtatman/speech-accent-archive): The speech accent archive was established to uniformly exhibit a large set of speech accents from a variety of language backgrounds. As such, the dataset contains 2,140 English speech samples, each from a different speaker reading the same passage. Furthermore, participants come from 177 countries and have 214 different native languages. 
* [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio): RAVDESS contains 24 professional actors (12 female and 12 male), vocalizing the same statements. Not only this, but the speech emotions captured include calm, happy, sad, angry, fearful, surprise, and disgust at two levels of intensity.
* [TED-LIUM Release 3](https://www.openslr.org/51): The TED-LIUM corpus is made from TED talks and their transcriptions available on the TED website. It consists of 2,351 audio samples, 452 hours of audio. In addition, the dataset contains 2,351 aligned automatic transcripts in STM format.
* [Google Audioset](https://research.google.com/audioset/): This dataset contains expanding ontology of 635 audio event classes and a collection of over 2 million 10-second sound clips from YouTube videos. Moreover, Google used human labelers to add metadata, context and content analysis.
* [LibriSpeech ASR Corpus](https://www.openslr.org/12): This corpus contains over 1,000 hours of English speech derived from audiobooks. Most of the recordings are based on texts from Project Gutenberg. [Kaggle librispeech-clean](https://www.kaggle.com/victorling/librispeech-clean), [Tensorflow librispeech](https://www.tensorflow.org/datasets/catalog/librispeech)

### Speaker Identification Datasets
* [Gender Recognition by Voice](https://www.kaggle.com/primaryobjects/voicegender): This database’s goal is to help systems identify whether a voice is male or female based upon acoustic properties of the voice and speech. Therefore, the dataset consists of over 3,000 recorded voice samples collected from male and female speakers. 
* [Common Voice](https://commonvoice.mozilla.org/en/datasets): This dataset contains hundreds of thousands of voice samples for voice recognition. It includes over 500 hours of speech recordings alongside speaker demographics. To build the corpus, the content came from user submitted blog posts, old movies, books, and other public speech.
* [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb): VoxCeleb is a large-scale speaker identification dataset that contains over 100,000 phrases by 1,251 celebrities. Similar to the previous datasets, VoxCeleb includes a diverse range of accents, professions and age. 

### Speech Command Datasets
* [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands): Created by the TensorFlow and AIY teams, this dataset contains 65,000 clips, each one second in duration. Each clip contains one of the 30 different voice commands spoken by thousands of different subjects.
* [Synthetic Speech Commands Dataset](https://www.kaggle.com/jbuchner/synthetic-speech-commands-dataset): Created by Pete Warden, the Synthetic Speech Commands Dataset is made up of small speech samples. For example, each file contains single-word utterances such as yes, no, up, down, on, off, stop, and go. 
* [Fluent Speech Commands Dataset](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/): This comprehensive dataset contains over 30,000 utterances from nearly 100 speakers. In this dataset, each .wav file contains a single utterance used to control smart-home appliances or virtual assistants. For example, sample recordings include “put on the music” or “turn up the heat in the kitchen”. In addition, all audio contains action, object, and location labels.

### Conversational Speech Recognition Datasets
* [The CHiME-5 Dataset](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/data.html): This dataset is made up of the recordings of 20 separate dinner parties that took place in real homes. Each file is a minimum of 2 hours and includes audio recorded in the kitchen, living and dining room. 
* [2000 HUB5 English Evaluation Transcripts](https://catalog.ldc.upenn.edu/LDC2002T43): Developed by the Linguistic Data Consortium (LDC), HUB5 consists of transcripts of 40 English telephone conversations. The HUB5 evaluation series focuses on conversational speech over the telephone with the task of transcribing conversational speech into text. 
* [CALLHOME American English Speech](https://catalog.ldc.upenn.edu/LDC97S42): Developed by the Linguistic Data Consortium (LDC), this dataset consists of 120 unscripted 30-minute telephone conversations in English. Due to the conditions of the study, most participants called family members or close friends.

### Multilingual Speech Datasets
* [CSS10](https://github.com/Kyubyong/css10): A collection of single speaker speech datasets for 10 languages. The dataset contains short audio clips in German, Greek, Spanish, French, Finnish, Hungarian, Japanese, Dutch, Russian and Chinese.
* [BACKBONE Pedagogic Corpus of Video-Recorded Interviews](https://github.com/Jakobovski/free-spoken-digit-dataset): A web-based pedagogic corpora of video-recorded interviews with native speakers of English, French, German, Polish, Spanish and Turkish as well as non-native speakers of English.
* [Arabic Speech Corpus](http://en.arabicspeechcorpus.com/): This speech corpus contains phonetic and orthographic transcriptions of more than 3.7 hours of Modern Standard Arabic (MSA) speech. 
* [Nijmegen Corpus of Casual French](http://www.mirjamernestus.nl/Ernestus/NCCFr/index.php): Another single speech dataset, the Nijmegen corpus includes 35 hours of high-quality recordings. In this case, it features 46 French speakers conversing among friends, orthographically annotated by professional transcribers.
* [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset): This simple dataset contains recordings of spoken digits trimmed so that they are silent at the beginnings and ends.
* [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/): This is a corpus of aligned spoken articles from Wikipedia. In addition to English, the data is also available in German and Dutch.

---
## Text-To-Speech (using TTS)

### WaveNet
**Paper:** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)<br>
**Code:** [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)<br>
With a pre-trained model provided here, you can synthesize waveform given a mel spectrogram, not raw text.<br>
You will need mel-spectrogram prediction model (such as Tacotron2) to use the pre-trained models for TTS.<br>
**Demo:** [An open source implementation of WaveNet vocoder](https://r9y9.github.io/wavenet_vocoder/)<br>
**Blog:** [WaveNet: A generative model for raw audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)<br>
![](https://lh3.googleusercontent.com/Zy5xK_i2F8sNH5tFtRa0SjbLp_CU7QwzS2iB5nf2ijIf_OYm-Q5D0SgoW9SmfbDF97tNEF7CmxaL-o6oLC8sGIrJ5HxWNk79dL1r7Rc=w1440-rw-v1)

---
### Tacotron-2
**Paper:** [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)<br>
**Code:** [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)<br>
![](https://camo.githubusercontent.com/d6c3e238b30a49a31c947dd0c5b344c452b53ab5eb735dc79675b67c92a2cf96/68747470733a2f2f707265766965772e6962622e636f2f625538734c532f5461636f74726f6e5f325f4172636869746563747572652e706e67)

**Code:** [Tacotron 2 (without wavenet)](https://github.com/NVIDIA/tacotron2)<br>
![](https://github.com/NVIDIA/tacotron2/blob/master/tensorboard.png?raw=true)

---
### Forward Tacotron
**Code:** [as-ideas/ForwardTacotron](https://github.com/as-ideas/ForwardTacotron)<br>
**Blog:** [利用 ForwardTacotron 創造穩健的神經語言合成](https://blogs.nvidia.com.tw/2021/03/31/creating-robust-neural-speech-synthesis-with-forwardtacotron/)<br>

![](https://github.com/as-ideas/ForwardTacotron/blob/master/assets/model.png?raw=true)

---
### Meta-TTS
**Paper:** [Meta-TTS: Meta-Learning for Few-Shot Speaker Adaptive Text-to-Speech](https://arxiv.org/abs/2111.04040)<br>
**Code:** [https://github.com/SungFeng-Huang/Meta-TTS](https://github.com/SungFeng-Huang/Meta-TTS)<br>
![](https://github.com/SungFeng-Huang/Meta-TTS/blob/main/evaluation/images/meta-FastSpeech2.png?raw=true)

---
## Speech Seperation

### Looking to Listen
**Paper:** [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation](https://arxiv.org/abs/1804.03619)<br>
**Blog:** [Looking to Listen: Audio-Visual Speech Separation](https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)<br>

![](https://3.bp.blogspot.com/-i8yGQmRfu6k/Ws03pWxgp2I/AAAAAAAACiM/3KgklbbHIvsYo4Tyw3N1TKa7Eywagr4eACLcBGAs/s640/image6.jpg)
<iframe width="640" height="360" src="https://www.youtube.com/embed/Z_ogAiVoE1g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="640" height="360" src="https://www.youtube.com/embed/uKwUL7vt03M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="640" height="360" src="https://www.youtube.com/embed/_7aMiqXubWo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### VoiceFilter
**Paper:** [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826)<br>
**Code:** [mindslab-ai/voicefilter](https://github.com/mindslab-ai/voicefilter)<br>
Training took about 20 hours on AWS p3.2xlarge(NVIDIA V100)<br>
**Code:** [jain-abhinav02/VoiceFilter](https://github.com/jain-abhinav02/VoiceFilter)<br>
The model was trained on Google Colab for 30 epochs. Training took about 37 hours on NVIDIA Tesla P100 GPU.<br>

![](https://github.com/jain-abhinav02/VoiceFilter/raw/master/assets/images/model_workflow.PNG)
<iframe width="600" height="338" src="https://www.youtube.com/embed/2BF_1X7bmds" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### VoiceFilter-Lite
**Paper:** [VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition](https://arxiv.org/abs/2009.04323)<br>
**Blog:** [](https://google.github.io/speaker-id/publications/VoiceFilter-Lite/)<br>

![](https://google.github.io/speaker-id/publications/VoiceFilter-Lite/resources/architecture.png)
<iframe width="800" height="450" src="https://www.youtube.com/embed/BiWMZdnHuVs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### QCNN ASR
**Paper:** [Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition](https://arxiv.org/abs/2010.13309)<br>
**Code:** [huckiyang/QuantumSpeech-QCNN](https://github.com/huckiyang/QuantumSpeech-QCNN)<br>

![](https://github.com/huckiyang/QuantumSpeech-QCNN/blob/main/images/QCNN_Sys_ASR.png?raw=true)
![](https://github.com/huckiyang/QuantumSpeech-QCNN/blob/main/images/cam_sp_0.png?raw=true)

---
### Voice Filter
**Paper:** [Voice Filter: Few-shot text-to-speech speaker adaptation using voice conversion as a post-processing module](https://arxiv.org/abs/2202.08164)<br>
![]()

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

