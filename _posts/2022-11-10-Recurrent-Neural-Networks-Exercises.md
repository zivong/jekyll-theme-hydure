---
layout: post
title: Recurrent Neural Networks Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes Text Generation, Stock Prediction, Sentiment Analysis, Neural Machine Translation, Transformers Question-Answering & Text-to-SQL, GPT2 Novel Generation, Reading Comprehension, Voice Filter, Speech Command Recognition.

---
## LSTM 

### Text Generation
**Kaggle:** [textgen-lstm](https://www.kaggle.com/rkuo2000/textgen-lstm)<br>
**Dataset:** ~/tf/datasets/text/shakespear.txt<br>
**Model**<br>
```
input_shape = (max_seq_len, vocab_size)

model = Sequential()
model.add(LSTM(64, input_shape=input_shape))
model.add(Dense(len(chars), activation='softmax'))
```
**Generated-Text**<br>
```
----- diversity: 0.2
----- Generating with seed: "this should be the house.
being holiday, the beggar's shop is shut.
what, ho! apothecary!

apothecar"
this should be the house.
being holiday, the beggar's shop is shut.
what, ho! apothecary!
```

---
### Stock Prediction
**Kaggle:** [rkuo2000/stock-lstm](https://www.kaggle.com/rkuo2000/stock-lstm)<br>
**Dataset:** download from [https://www.alphavantage.co/](https://www.alphavantage.co/)<br>
`pip install alpha_vantage`<br>
`python download_stock_quote.py GOOGL` <br>
`python stock_lstm.py GOOGL`<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_LSTM.png?raw=true)

---
### Stock LSTM with MACD + Buy/Sell Strategy
**Kaggle:** [rkuo2000/stock-lstm-macd](https://www.kaggle.com/rkuo2000/stock-lstm-macd)<br>

*technical indicator = Moving Average Convergence Divergence (MACD)*<br>

<img width="50%" height="50%" src="https://miro.medium.com/max/1269/0*dcNNGSOyOpE6Pby2.png">
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Stock_LSTM_MACD.png?raw=true)

---
### Sentiment Analysis
**Dataset:** [First GOP Debate Twitter Sentiment](https://www.kaggle.com/crowdflower/first-gop-debate-twitter-sentiment)<br>
**Kaggle:** [rkuo2000/sentiment-nltk](https://kaggle.com/rkuo2000/sentiment-nltk)<br>
**Kaggle:** [rkuo2000/sentiment-lstm](https://kaggle.com/rkuo2000/sentiment-lstm)<br>
*wordcloud of NaiveBayes Classifier*<br>
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-pos.png?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sentiment-NLTK-neg.png?raw=true"></td>
</tr>
</table>
> 2041/2041 - 1s - loss: 0.9442 - acc: 0.8305<br>
> score: 0.94<br>
> acc: 0.83<br>

---
### Neural Machine Translation
**Kaggle:** [rkuo2000/nmt-seq2seq](https://www.kaggle.com/rkuo2000/nmt-seq2seq)<br>
**Model**<br>
```
model = models.Sequential()
model.add(layers.Embedding(in_vocab, 512, input_length=in_timesteps, mask_zero=True))
model.add(layers.LSTM(512))
model.add(layers.RepeatVector(out_timesteps))
model.add(layers.LSTM(512, return_sequences=True))
model.add(layers.Dense(out_vocab, activation='softmax'))
```
**Translation Result**<br>
```
                      actual              predicted
2408  i have lots of friends    i have many friends    
4882      tom has no enemies  tom doesnt no enemies    
3374     tom needs your help    tom needs your help    
4345           do it with me           do it for me    
4869  his music is too noisy        a music is sour    
1148    youre very skeptical      youre very smart     
1678     youre a filthy liar    youre a filthy liar    
4923             ill do this           ill do that     
4371          tom isnt witty             tom isnt      
4353       well let you know      well let you know    
2274   he cant stop laughing          he cant stop     
2003          i am a student          im a student     
4884   lets do what tom said          lets tom  his    
2038      ill do it tomorrow            ill can it     
4098      would he like that          would you  it
```

---
### BERT Translate
**Kaggle:** [rkuo2000/bert-translate](https://www.kaggle.com/rkuo2000/bert-translate)<br>
input &emsp;&emsp;: "[CLS] I go to school by bus [SEP]"<br>
expected : "我搭公車上學"<br>

---
### Transformer pipelines
**Kaggle:** [rkuo2000/transformers-pipelines](https://www.kaggle.com/rkuo2000/transformers-pipelines)<br>

`pip install transformers`<br>

#### 1. Sentence Classification - Sentiment Analysis
```
from transformers import pipeline
```
```
nlp_sentence_classif = pipeline('sentiment-analysis')
nlp_sentence_classif('Such a nice weather outside !')
```
{'label': 'POSITIVE', 'score': 0.9997655749320984}

#### 2. Token Classification - Named Entity Recognition
```
nlp_token_class = pipeline('ner')
nlp_token_class('Hugging Face is a French company based in New-York.')
```
[{'entity': 'I-ORG',  'score': 0.9970938,  'index': 1,  'word': 'Hu',  'start': 0,  'end': 2},<br>
 {'entity': 'I-ORG',  'score': 0.93457514,  'index': 2,  'word': '##gging',  'start': 2,  'end': 7},<br>
 {'entity': 'I-ORG',  'score': 0.978706,  'index': 3,  'word': 'Face',  'start': 8,  'end': 12},<br>
 {'entity': 'I-MISC',  'score': 0.9981996,  'index': 6,  'word': 'French',  'start': 18,  'end': 24},<br>
 {'entity': 'I-LOC',  'score': 0.9983047,  'index': 10,  'word': 'New',  'start': 42,  'end': 45},<br>
 {'entity': 'I-LOC',  'score': 0.8913456,  'index': 11,  'word': '-',  'start': 45,  'end': 46},<br>
 {'entity': 'I-LOC',  'score': 0.99795234,  'index': 12,  'word': 'York',  'start': 46,  'end': 50}]<br>
    
#### 3. Question Answering
```
from transformers import pipleine
nlp_qa = pipeline('question-answering`)
nlp_qa(context="Hugging Face is a French company based in New-York.", questions="Where is Hugging Face based?")
```
{'score': 0.8327597379684448, 'start': 39, 'end': 49, 'answer': 'n New-York'}

#### 4. Text Generation - Mask Filling
```
nlp_fill = pipeline('fill-mask')
nlp_fill('Hugging Face is a French company based in <mask>')
```
[{'sequence': 'Hugging Face is a French company based in Paris',
  'score': 0.27758949995040894,
  'token': 2201,
  'token_str': ' Paris'},<br>
 {'sequence': 'Hugging Face is a French company based in Lyon',
  'score': 0.14941278100013733,
  'token': 12790,
  'token_str': ' Lyon'},<br>
 {'sequence': 'Hugging Face is a French company based in Geneva',
  'score': 0.045764125883579254,
  'token': 11559,
  'token_str': ' Geneva'},<br>
 {'sequence': 'Hugging Face is a French company based in France',
  'score': 0.04576260223984718,
  'token': 1470,
  'token_str': ' France'},<br>
 {'sequence': 'Hugging Face is a French company based in Brussels',
  'score': 0.040675751864910126,
  'token': 6497,
  'token_str': ' Brussels'}]<br>

#### 5. Projection - Features Extraction  
```
import numpy as np
nlp_features = pipeline('feature-extraction')
output = nlp_features('Hugging Face is a French company based in Paris')
np.array(output).shape   # (Samples, Tokens, Vector Size)
```
(1, 12, 768)

---
### CKIP Transformers (斷詞工具）
**Code:** [CKIP Transformers](https://github.com/ckiplab/ckip-transformers)<br>
**[PyPi](https://pypi.org/project/ckip-transformers)**<br>

`pip install -U transformers`<br>
`pip install -U ckip_transformers`<br>

* import library

```
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
```

* initialize drivers

```
# Initialize drivers
ws_driver  = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
ner_driver = CkipNerChunker(model="bert-base")
```

* One may also load their own checkpoints using our drivers

```
# Initialize drivers with custom checkpoints
ws_driver  = CkipWordSegmenter(model_name="path_to_your_model")
pos_driver = CkipPosTagger(model_name="path_to_your_model")
ner_driver = CkipNerChunker(model_name="path_to_your_model")
```

* use CPU / GPU

```
# Use CPU
ws_driver = CkipWordSegmenter(device=-1)

# Use GPU:0
ws_driver = CkipWordSegmenter(device=0)
```

* Input text
```
# Input Text
text = [
   "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
   "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
   "空白 也是可以的～",
]
```

* Run pipeline
  - 斷詞與實體辨識的輸入必須是 list of sentences。
  - 詞性標記的輸入必須是 list of list of words。
```
# Run pipeline
ws  = ws_driver(text)
pos = pos_driver(ws)
ner = ner_driver(text)
```
詞性標記工具會自動用 '，,。：:；;！!？?' 等字元在執行模型前切割句子<br>

* use_delim=True (可設定 delim_set 參數使用別的字元做切)

```
# Enable sentence segmentation
ws  = ws_driver(text, use_delim=True)
ner = ner_driver(text, use_delim=True)
```
* use_delim=False (另外可指定 use_delim=False 已停用此功能，或於斷詞、實體辨識時指定 use_delim=True 已啟用此功能)

```
# Disable sentence segmentation
pos = pos_driver(ws, use_delim=False)

# Use new line characters and tabs for sentence segmentation
pos = pos_driver(ws, delim_set='\n\t')
```

* set `batch size` and `maximum sentence length` to better utilize you machine resources.<br>

```
# Sets the batch size and maximum sentence length
ws = ws_driver(text, batch_size=256, max_length=128)
```

* Show results

```
# Pack word segmentation and part-of-speech results
def pack_ws_pos_sentece(sentence_ws, sentence_pos):
   assert len(sentence_ws) == len(sentence_pos)
   res = []
   for word_ws, word_pos in zip(sentence_ws, sentence_pos):
      res.append(f"{word_ws}({word_pos})")
   return "\u3000".join(res)

# Show results
for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
   print(sentence)
   print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
   for entity in sentence_ner:
      print(entity)
   print()
```


* Exericse:
 
cd ~/tf<br>
python3 [ckip.py](https://github.com/rkuo2000/tf/blob/master/ckip.py)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CKIP_Transformers_exercise.png?raw=true)

---
### Transformer Text-to-SQL 
**Kaggle:** [rkuo2000/transformers-text2sql](https://www.kaggle.com/rkuo2000/transformers-text2sql)<br>
```
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

def get_sql(query):
    input_text = "translate English to SQL: %s </s>" % query
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
    return tokenizer.decode(output[0])

text = "How many models were finetuned using BERT as base model?"
get_sql(text)
```
SELECT COUNT Model fine tuned FROM table WHERE Base model = BERT

---
### GPT2 Text Generation

**Kaggle:** [rkuo2000/gpt2-textgen](https://www.kaggle.com/rkuo2000/gpt2-textgen)<br>
GPT2 PyTorch<br>
======================================== SAMPLE 1 ========================================<br>
 It was about the year 1000, and its inhabitants were called the 'wild man,' the 'pure forest,' because the flowers they brought with them were no longer seen. It was like a dream. In my mind, I thought, 'What is this?' And it was a dream. At that time I had no idea of the nature of nature.

I was a little child with a very bad case of epilepsy. I had been told that my parents were very pious and were very pious, and that I would prove myself to them. But I didn't know what to do. I had a bad case of epilepsy. I was very hungry and had to eat the whole day before I could get enough. I was in a state of complete confusion and I thought I was going mad. When I got to the hospital when I was twenty, I was immediately cured.

After that I was very sick. I was very sick. I was sick because I was sick with what I called the 'foul smell,' the 'foul smell' of the flowers in the flowers. And what did I do? I began to think I was going crazy. I thought that I was going mad. I kept thinking that I was going mad. I kept thinking that my friends were going crazy. I kept thinking that I was going mad. I kept thinking that I was going mad.

Then I was a little girl. My parents had been very good to me. My mother had taught me many things that I had never understood. My sister had been very good to me.

I knew, in fact, that what I had done was wrong, and that I could not understand how God could have made me stupid so easily by making me stupid. I had heard that God had made me stupid in a certain way. I knew that he had made me stupid. I had seen things that I had never seen before. It was like a dream. And then I said, 'What is this dream?'

I said, 'This is a dream in my head.' And I came to realize that I had been told that it was a dream.

I knew that God had made me stupid. I knew that he had made me stupid because I was like a child. I knew that I had been told that I was going crazy. I knew that I was going crazy. And I had learned that there was no God and that the only way to be a good man was to be kind to others, and that

---
### GPT2 Text Generation
**Kaggle:** [rkuo2000/gpt2-text-generation](https://www.kaggle.com/rkuo2000/gpt2-text-generation)<br>
[{'generated_text': 'I am thrilled to be here deep in the heart of Texas. We have great fans. I think there\'s a lot of respect for them."\n\nWinnipeg\'s Peter Dierkes said it was another huge day for them both at the'},<br>
 {'generated_text': 'I am thrilled to be here deep in the heart of Texas. With a reputation for quality food and healthy services, we want our community to have the best meals in their communities. We are ready to serve as many people as possible. We hope to'},<br>
 {'generated_text': "I am thrilled to be here deep in the heart of Texas. This has been a privilege to be here for so many years. It's amazing I'm back and I can't wait to say farewell to the people who've helped make this dream come"},<br>
 {'generated_text': 'I am thrilled to be here deep in the heart of Texas. I will see these four players in their own home after our game."'},<br>
 {'generated_text': 'I am thrilled to be here deep in the heart of Texas. It\'s like my father\'s time! It\'s so much more than a trip down memory lane, it\'s the culmination of my life," she says. "And it\'s been the'}]<br>
 
---
### GPT2 Chinese Poem
**Kaggle:** [rkuo2000/gpt2-chinese-poem](https://www.kaggle.com/rkuo2000/gpt2-chinese-poem)<br>
[{'generated_text': '[CLS]梅 山 如 积 翠 ， 了 了 复 攒 攒 。 云 接 天 开 画 ， 舟 移 岸 带 盘 。 滩 声 和 雨 听 ， 石 势 趁 风 看 。 明 日 重 来 路 ， 依 前 六 月 寒 。 名 何 代 下 ， 古 寺 暮 钟 残 。 僧 向 烟 中 老 ， 人 归 雨 外 看 。 山 光 翠 欲 滴 ， 秋 气 冷 全 乾 。 一 夜 猿 啼 急 ， 西 岩 瀑 布 寒 。 竺 前'}]

### GPT2 Chinese Lyrics
**Kaggle:** [rkuo2000/gpt2-chinese-lyrics](https://www.kaggle.com/rkuo2000/gpt2-chinese-lyrics)<br>
[{'generated_text': '最美的不是下雨天，是曾与你躲过雨的屋檐 ， 还 是 曾 经 最 爱 的 那 本 书 签 ， 如 今 我 已 不 在 身 边 ， 而 我 却 还 在 想 念 ， 我 们 牵 手 的 那 个 街 角 ， 我 们 笑 的 那 么 甜 ， 你 说 天 色 那 么 好 ， 是 该 和 我 去 拥 抱 ， 每 天 都 能 幸 福 的 在 画 面 上 画 满 幸 福 的 符'}]

### GPT2 Novel
**Kaggle:** [rkuo2000/gpt2-novel](https://www.kaggle.com/rkuo2000/gpt2-novel)<br>
**Blog:** [直觀理解GPT2語言模型並生成金庸武俠小說](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)<br>

![](https://leemeng.tw/images/gpt2/4_%E5%A4%A9%E9%BE%8D%E5%85%AB%E9%83%A8.jpg)

---
### GPT-2 Fine Tuning
**Kaggle:** [rkuo2000/gpt2-german-recipes](https://www.kaggle.com/rkuo2000/gpt2-german-recipes)<br>
`chef('Die Nudeln Kochen, Fleisch anbraten')`<br>
[{'generated_text': 'Die Nudeln Kochen, Fleisch anbraten und zusammen mit den Paprika in die Sauce rühren. Mit Salz, Pfeffer, Paprika, Koriander, schwarzem Pfeffer, Paprika, Chili und Paprikapulver abschmecken'}]<br>

**Kaggle:** [rkuo2000/gpt2-trump-s-rallies](https://www.kaggle.com/rkuo2000/gpt2-trump-s-rallies)<br>

**Kaggle:** [rkuo2000/gpt2-twitterbot](https://www.kaggle.com/rkuo2000/gpt2-twitterbot)<br>

---
## Chatbot
### Dialog in Japanese
**Code:** [https://github.com/reppy4620/Dialog](https://github.com/reppy4620/Dialog)<br>

![](https://github.com/reppy4620/Dialog/raw/master/result/result.png)

---
### Transformers Chatbot
**Code:** [https://github.com/demi6od/ChatBot](https://github.com/demi6od/ChatBot)<br>

---
### ELECTRA SQuAD2.0
**Kaggle:** [rkuo2000/electra-squad2-0](https://www.kaggle.com/rkuo2000/electra-squad2-0)<br>
**Dataset:** [buildformacarov/squad-20](https://www.kaggle.com/buildformacarov/squad-20)<br>
*running GPU will be out of memory, running TPU or CPU : ETA > 2 days*<br>

10/8162 = 0.1%, SPS: 0.0, ELAP: 4:27, ETA: 2 days, 12:27:13 - loss: 183.2872

---
### Text-To-Speech
**TorchAudio:** [TTS with torchaudio](https://pytorch.org/tutorials/intermediate/text_to_speech_with_torchaudio.html)<br>

**Kaggle:** [rkuo2000/forwardtacotron-tts](https://www.kaggle.com/rkuo2000/forwardtacotron-tts)<br>
<audio controls="controls">
  <source type="audio/wav" src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ForwardTacotron_TTS.wav?raw=true"></source>
</audio>

---
### Voice Filter
**Code:** [mindslab-ai/voicefilter](https://github.com/mindslab-ai/voicefilter)<br>
*Training took about 20 hours on AWS p3.2xlarge(NVIDIA V100)*<br>
**Code:** [jain-abhinav02/VoiceFilter](https://github.com/jain-abhinav02/VoiceFilter)<br>
*The model was trained on Google Colab for 30 epochs. Training took about 37 hours on NVIDIA Tesla P100 GPU.*<br>

---
### QCNN ASR
**Kaggle:** [rkuo2000/qcnn-asr](https://www.kaggle.com/rkuo2000/qcnn-asr)<br>
**Dataset:** Google Speech Commands [v0.01](https://www.kaggle.com/neehakurelli/google-speech-commands) / [v0.02](https://www.kaggle.com/guntherneumair1/speechcommandv02-cleaned)<br>
Labels = ['left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop’,]<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/QCNN_ASR.png?raw=true)

QCNN U-Net Bi-LSTM Attention Model<br>
![](https://github.com/huckiyang/QuantumSpeech-QCNN/blob/main/images/QCNN_Sys_ASR.png?raw=true)

* Neural Saliency by Class Activation Mapping<br>
`python cam_sp.py`<br>
<img width="50%" height="50%" src="https://github.com/huckiyang/QuantumSpeech-QCNN/blob/main/images/cam_sp_0.png?raw=true">

* evaluation with a CTC model w WER<br>
`python qsr_ctc_wer.py`<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

