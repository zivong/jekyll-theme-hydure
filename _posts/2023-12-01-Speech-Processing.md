---
layout: post
title: Speech Processing
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Speech Datasets, Text-To-Speech, Speech Seperation, Voice Clonging, etc.

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
**Blog:** [利用 ForwardTacotron 創造穩健的神經語言合成](https://blogs.nvidia.com.tw/2021/03/31/creating-robust-neural-speech-synthesis-with-forwardtacotron/)<br>
**Code:** [https://github.com/as-ideas/ForwardTacotron](https://github.com/as-ideas/ForwardTacotron)<br>
![](https://github.com/as-ideas/ForwardTacotron/blob/master/assets/model.png?raw=true)

---
### Few-shot Transformer TTS
**Paper:** [Multilingual Byte2Speech Models for Scalable Low-resource Speech Synthesis](https://arxiv.org/abs/2103.03541)<br>
**Code:** [https://github.com/mutiann/few-shot-transformer-tts](https://github.com/mutiann/few-shot-transformer-tts)<br>

---
### Meta-TTS
**Paper:** [Meta-TTS: Meta-Learning for Few-Shot Speaker Adaptive Text-to-Speech](https://arxiv.org/abs/2111.04040)<br>
**Code:** [https://github.com/SungFeng-Huang/Meta-TTS](https://github.com/SungFeng-Huang/Meta-TTS)<br>
![](https://github.com/SungFeng-Huang/Meta-TTS/blob/main/evaluation/images/meta-FastSpeech2.png?raw=true)

---
### SeamlessM4T
**Paper:** [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)<br>
**Code:** [https://github.com/facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication)<br>
**Colab:** [seamless_m4t_colab](https://github.com/camenduru/seamless-m4t-colab)<br>
[app.py](https://huggingface.co/spaces/facebook/seamless_m4t/blob/main/app.py)<br>

SeamlessM4T models support the tasks of:
* Speech-to-speech translation (S2ST)
* Speech-to-text translation (S2TT)
* Text-to-speech translation (T2ST)
* Text-to-text translation (T2TT)
* Automatic speech recognition (ASR)

---
### coqui TTS
**Code:** [https://www.kaggle.com/code/rkuo2000/coqui-tts](https://github.com/coqui-ai/TTS)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/coqui-tts](https://www.kaggle.com/code/rkuo2000/coqui-tts)<br>

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
### Voice Filter
**Paper:** [Voice Filter: Few-shot text-to-speech speaker adaptation using voice conversion as a post-processing module](https://arxiv.org/abs/2202.08164)<br>

---
## Voice Conversion

### Review of Deep Learning Voice Conversion 
**Paper:** [Overview of Voice Conversion Methods Based on Deep Learning](https://www.mdpi.com/2076-3417/13/5/3100)<br>
**Paper:** [Reimagining Speech: A Scoping Review of Deep Learning-Powered Voice Conversion](https://arxiv.org/abs/2311.08104)<br>

---
### Voice Cloning
**Paper:** [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558.abs)<br>
**Kaggle:** [https://github.com/CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)<br>
![](https://camo.githubusercontent.com/52ad48c214ea4126fc823420629940a0cccbf638ff386709056de790edf8bd1b/68747470733a2f2f692e696d6775722e636f6d2f386c46556c677a2e706e67)<br>

---
###
**Paper:** [PERSONALIZED LIGHTWEIGHT TEXT-TO-SPEECH: VOICE CLONING WITH ADAPTIVE STRUCTURED PRUNING]()<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

