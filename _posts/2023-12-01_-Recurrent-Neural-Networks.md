---
layout: post
title: Recurrent Neural Networks
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes RNN, LSTM, Neural Machine Translation, seq2seq, Transformers, Electra, Chinese-Electra

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
**Code:** [Sentiment-NLTK](https://www.kaggle.com/rkuo2000/sentiment-nltk)<br>
*wordcloud of NaiveBayes Classifier*<br>
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-pos.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-neg.png?raw=true"></td>
</tr>
</table>

**Code:** [Sentiment-LSTM](https://www.kaggle.com/rkuo2000/sentiment-lstm)<br>
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
**Blog.** [A Must-Read NLP Tutorial on Neural Machine Translation](https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/)<br>
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/01/architecture.png)
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/01/enc_dec_2.png)

### seq2seq
![](https://miro.medium.com/max/1400/1*x4wsJobiSC7zlTkP8yv40A.png)

```
model = Sequential()
model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
model.add(LSTM(units))
model.add(RepeatVector(out_timesteps))
model.add(LSTM(units, return_sequences=True))
model.add(Dense(out_vocab, activation='softmax'))
```

![](https://i.stack.imgur.com/YjlBt.png)

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
### Mianbot
**Code:** [zake7749/Chatbot](https://github.com/zake7749/Chatbot)<br>
![](https://raw.githubusercontent.com/zake7749/Chatbot/master/docs/demo.png)

---
### Japanese Chatbot
**Code:** [reppy4620/Dialog](https://github.com/reppy4620/Dialog)<br>

![](https://github.com/reppy4620/Dialog/raw/master/result/result.png)

To solve low diversity of dialogs: **Inverse Token Frequency (ITF) loss**<br>
**Paper:** [Another Diversity-Promoting Objective Function for Neural Dialogue Generation](https://arxiv.org/abs/1811.08100)<br>

---
### Hand Writing Synthesis
**Paper:** [Generating Sequences With Recurrent Neural Networks]()<br>
**Code:** [Grzego/handwriting-generation](https://github.com/Grzego/handwriting-generation)<br>
![](https://github.com/Grzego/handwriting-generation/blob/master/imgs/example-2.gif?raw=true)

---
### Smart Reply
**paper:** [Gmail Smart Compose: Real-Time Assisted Writing](https://arxiv.org/abs/1906.00080)<br>
![](https://www.weak-learner.com/assets/img/blog/personal/sc1.png)
![](https://www.weak-learner.com/assets/img/blog/personal/sc2.png)

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


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

