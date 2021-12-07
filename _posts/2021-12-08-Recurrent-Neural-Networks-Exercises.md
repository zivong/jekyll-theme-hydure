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
`pip install alpha_vantage`
`python download_stock_quote.py GOOGL` 
`python stock_lstm.py GOOGL`

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
nlp_qa(context="Hugging Face is a French company based n New-York.", questions="Where is Hugging Face based?")
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
### Transformer Text Generation
**Kaggle:** [rkuo2000/transformer-gpt2](https://www.kaggle.com/rkuo2000/transformer-gpt2)<br>

```
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
gentxt = generator("It was a bright cold day in April,", max_length=30, num_return_sequences=5)
[print(gentxt[i]['generated_text']) for i in range(len(gentxt))]
```
1. It was a bright cold day in April, when I walked out the door. I'd just started doing some work at the company's new construction facility
2. It was a bright cold day in April, in Mexico City. The skies above Mexico were dim but dark by day. With no light in their eyes
3. It was a bright cold day in April, and the sky was blue, with a single moon. For a few more seconds, you felt like a
4. It was a bright cold day in April, the third week of the World Cup. It's that day when the world champions go to their match before
5. It was a bright cold day in April, so I was heading home from my job at the Air Force base in Las Vegas. When I came back
 
---
### Transformer Text-to-SQL 
**Kaggle:** [rkuo2000/transformer-text2sql](https://www.kaggle.com/rkuo2000/transformer-text2sql)<br>
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
### Fine Tuning GPT-2 model
**[Fine-tune a non-English GPT-2 Model with Huggingface](https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb)**

**Dataset:** [Donal Trump rallies](https://www.kaggle.com/christianlillelund/donald-trumps-rallies)<br>

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
### GPT2 Novel
**Kaggle:** [rkuo2000/gpt2-novel](https://www.kaggle.com/rkuo2000/gpt2-novel)<br>
**Blog:** [直觀理解GPT2語言模型並生成金庸武俠小說](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)<br>

![](https://leemeng.tw/images/gpt2/4_%E5%A4%A9%E9%BE%8D%E5%85%AB%E9%83%A8.jpg)

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
**Kaggle:** [rkuo2000/forwardtacotron-tts](https://www.kaggle.com/rkuo2000/forwardtacotron-tts)<br>

---
### Voice Filter
**Code:** [mindslab-ai/voicefilter](https://github.com/mindslab-ai/voicefilter)<br>
Training took about 20 hours on AWS p3.2xlarge(NVIDIA V100)<br>
**Code:** [jain-abhinav02/VoiceFilter](https://github.com/jain-abhinav02/VoiceFilter)<br>
The model was trained on Google Colab for 30 epochs. Training took about 37 hours on NVIDIA Tesla P100 GPU.<br>

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

