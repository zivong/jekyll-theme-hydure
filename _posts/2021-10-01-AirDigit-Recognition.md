---
layout: post
title: AirDigit Recognition
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

以手機所錄取之手勢動作資料上傳至Kaggle平台訓練AI模型後,交由電腦啟動辨識服務器, 再以AirDigit App傳送手勢動作之資料至電腦進行辨識。

---
## 操作流程
使用 AirDigit App 將空中手寫數字之手勢動作錄取三軸加速器資料儲存, 然後上傳至Kaggle平台建立資料集來訓練AI模型,
再交由電腦執行辨識服務器程式, 之後打開空中手寫數字App, 傳送動作資料至PC進行辨識, 並回覆數字顯示於App上。

---
## 手機 App開發平台: MIT App Inventor 2
### Android 系統
以Google Chrome瀏覽器打開網址 **[http://ai2.appinventor.mit.edu](http://ai2.appinventor.mit.edu)**, 即可使用App Inventor 2 開發環境

### iOS 系統
* 操作步驟請參考 [https://youtu.be/z9YmWY8FeII](https://youtu.be/z9YmWY8FeII)<br />
 - iPhone 安裝MIT App Inventor, 啟動後產生QR碼
 - 電腦以Chrome開啟**ai2.appinventor.mit.edu**上之計畫，上方選單選**Connect**，後選 **AI companion** ，產生**QR碼**及文字碼
 - iPhone執行MIT App Inventor掃描QR碼或輸入文字碼，AI2計畫會傳送應用程式至iPhone執行
<iframe width="811" height="456" src="https://www.youtube.com/embed/z9YmWY8FeII" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## 開發 AirDigit 手機應用程式
使用個人電腦或筆電進行以下步驟：<br />
1. 下載 [AirDigit.aia](https://drive.google.com/file/d/1bKW38dGtjk3XkeMbiFFcalfw-CyR1rTM/view?usp=sharing)於電腦<br />
2. 於 [https://ai2.appinventor.mit.edu](https://ai2.appinventor.mit.edu)上方選 **import project(.aia) from my computer**<br />
![](https://github.com/rkuo2000/Robotics/blob/gh-pages/images/AI2_import_aia.png?raw=true)

3. App設計畫面如下
![](https://github.com/rkuo2000/Robotics/blob/gh-pages/images/AI2_AirDigit_Design.png?raw=true)

4. App程式畫面如下
![](https://github.com/rkuo2000/Robotics/blob/gh-pages/images/AI2_AirDigit_Block.png?raw=true)

5. 修改檔案路徑 file:///storage/emulated/0/Android/data/appinventor.ai_**rkuo2000**.AirDigit/files/
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AI2_AirDigit_initial_filepath.png?raw=true)
   - 修改rkuo2000為yourname (查找手機應用程式的檔案路徑, 如果名稱為中文, 建議先改成英文)<br>
   - 須再手機上先用檔案管理員搜尋0_000.csv檔案, 再修改路徑
6. 按MIT AppInventor上方選單**Build**會產生QR碼, 用手機掃描QR碼可下載安裝應用程式

---
## 使用AirDigit App 紀錄手勢動作以建立資料集
### 空中手寫0~9數字, 10為手機不動
![](https://github.com/rkuo2000/Robotics/blob/gh-pages/images/AirDigit_dataset.png?raw=true)
*手寫數字需按筆劃動作進行, 可有效提昇辨識率！*<br />

### 資料集之檔案
* 錄取三軸加速器之資料檔案如下：<br />
> **Train: 0~10 各20筆訓練資料**<br />
> 0_000.csv, 0_001.csv, …, 0_019.csv<br />
> 1_000.csv, 1_001.csv, …, 1_019.csv<br />
> …<br />
> 9_000.csv, 9_001.csv, …, 9_019.csv<br />
> 10_000.csv, 10_001.csv, …, 10_019.csv<br />
> <br />
> **Test: 0~10 各2筆訓練資料**<br />
> 0_000.csv, 0_001.csv, 1_000.csv, 1_001.csv, …, 9_000.csv, 9_001.csv, 10_000.csv, 10_001.csv <br />

### 使用AirDigit App進行以下資料集取樣步驟：
1. 輸入**[檔案名]**: `0_000` (10_000 持手機不動, 平放亦可）
 - 按[取樣儲存]時會進行取樣,並儲存資料檔`0_000.csv`至手機
 
2. 輸入[取樣速度]: `50` (初設取樣速度每50ms一次 = 每秒20次)

3. 輸入[感測時間]: `1500`（初設取樣時間長度1500ms = 1.5秒, 如動作較大較長, 可延長至２或３秒）

4. 按下**[取樣儲存]**, 接著馬上搖動手機做出手勢動作 (例如: 0~9之數字)
 - 如果檔案名沒有修改, 此檔案之資料會被新取樣之資料覆蓋<br />
<br />
5. 手勢動作前要先設不同[檔案名], 才按下[開始取樣]
 - 每個數字動作需至少做**20次** (如:0_000, 0_001, ..., 0_019)
 
6. 完成0~10手勢動作的各**20次**取樣 (如:0_xxx, 1_xxx, ..., 10_xxx)

7. 開手機上之[檔案管理員], 尋找含有0_000.csv之airdigit資料夾, 將其壓縮成.zip, 並命名為**AirDigit.zip**
 - Android手機之路徑為internal storage>Android>data>appinventor_yourID.airdigit<br />
<br />
8. 上傳此AirDigit.zip至你的Kaggle Datasets (https://kaggle.com/your_name/datasets) 
 - 將會建出一個你的資料集airdigit (如:`https://kaggle.com/your_name/airdigit`)

**Note:** 請注意手寫數字之筆畫順序，需固定筆畫順序進行

---
## 於Kaggle平台使用資料集訓練AI模型

0. 請先確認電腦上之Tensorflow版本
- 開啟 python3
> import tensorflow as tf<br>
> print(tf. _ _ version _ _)

1. 於電腦上用Chrome開**AirDigit-CNN**計畫網址
- https://kaggle.com/rkuo2000/airdigit-cnn (為tensorflow==2.4.1)
- https://kaggle.com/rkuo2000/airdigit-classification (為tensorflow==2.6.2)

2. 右上方按[Edit & Copy]一鍵複製計畫, 會產生 https://kaggle.com/your_name/airdigit-cnn

3. 於Kaggle上執行AirDigit-CNN進行模型訓練（點選上方[Run-All]來執行所有程式）

4. 執行程式後會產生一模型檔 **airdigit_cnn.h5** (點選下載至電腦）

---
## 於電腦上開啟 AI辨識服務器

### 安裝環境
* 於電腦安裝[Annaconda 3](https://www.anaconda.com/products/individual) (提供一方便之python及相關packages的打包安裝)
 
* 於Winwdows上啟動**Anaconda Prompt**, 或於MAC / Ubuntu 上開啟Terminal

* 以下安裝python3所之packages 含opencv-python, tensorflow<br />
`$ conda create -n tensor`<br />
`$ conda activate tensor`<br />
`$ conda install matplotlib pillow`<br />
`$ pip install pandas requests`<br />
`$ pip install opencv-python`<br />

* 檢查tensorflow版本, 並安裝與kaggle平台相同之tensorflow版本<br />
`$ python -c 'import tensorflow as tf; print(tf.__version__)'`<br />
`$ pip install tensorflow`==2.6.0 (安裝與kaggle相同之tensorflow版本)<br />

* 取得程式範例<br />
`$ git clone https://github.com/rkuo2000/tf`<br />

* 將**airdigit_cnn.h5**複製到 ~/tf/models

* 執行 AI辨識服務器之程式 [airdigit_cnn.py](https://github.com/rkuo2000/tf/blob/master/airdigit_cnn.py)<br />
`$ cd ~/tf`<br />
`$ python airdigit_cnn.py`<br />

![](https://github.com/rkuo2000/Robotics/blob/gh-pages/images/airdigit_cnn_server1.png?raw=true)

顯示服務器工作埠在192.168.1.7:5000

![](https://github.com/rkuo2000/Robotics/blob/gh-pages/images/airdigit_cnn_server2.png?raw=true)

最後顯示辨識出之數字為**none**及回傳"POST /predict HTTP/1.1 200"訊息至手機<br />

---
## 進行手勢動作辨識

* 首先讓電腦與手機連到同一網路 (可由手機開熱點分享給電腦)

<p align="center"><img src="https://github.com/rkuo2000/Robotics/blob/gh-pages/images/AirDigit_App.png?raw=true"></p><br />

* 以手機啟動**AirDigit** App

* 於[URL] : `192.168.1.7:5000/predict` (輸入已啟動 AI辨識服務器之電腦IP位址)<br />
  顯示IP位址: 
  - **Windows** 開Command prompt : `ipconfig`
  - **MAC** 開Terminal : `ipconfig`
  - **Ubuntu** 開Terminal : `ip addr`

* 按下**[取樣上傳]**錄取動作資料並上傳至辨識服務器, 然後回傳辨識結果
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirDigit_App_result.jpg?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirDigit_airdigit_cnn.png?raw=true)

<br />
<br />

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

