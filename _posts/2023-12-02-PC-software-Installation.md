---
layout: post
title: PC Software Installation
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

PC Software Installation: Editor, GitBash, Python3, Anaconda3, CUDA & CuDNN, Tensorflow & PyTorch installation. 

---
## List of PC Software to install
1. Editor
2. OS
3. GPU
4. Python
5. Tensorflow/Pytorch

---
### Editor

#### For Windows: install [Notepad++](https://notepad-plus-plus.org/downloads/)

### For Ubuntu / MacOS: no intallation needed, use built-in editors
* **nano** (for Ubuntu / MacOS)<br>
* **vim** (for Ubuntu / MacOS)<br>

---
### Terminal

#### install [Git for Windows](https://gitforwindows.org/)

**[Linux Command 命令列指令與基本操作入門教學](https://blog.techbridge.cc/2017/12/23/linux-commnd-line-tutorial/)**<br>
* `ls -l` (列出目錄檔案)<br>
* `cd ~` (換目錄)<br>
* `mkdir new` (產生新檔案夾)<br>
* `rm file_name` (移除檔案)<br>
* `rm –rf directory_name` (移除檔案夾)<br>
* `df .` (顯示SD卡已用量)<br>
* `du –sh directory` (查看某檔案夾之儲存用量)<br>
* `free` (檢查動態記憶體用量)<br>
* `ps –a`   (列出正在執行的程序)<br>
* `kill -9 567`  (移除程序 id=567)<br>
* `cat /etc/os-release` (列出顯示檔案內容，此檔案是作業系統版本)<br>
* `vi file_name` (編輯檔案)<br>
* `nano file_name` (編輯檔案)<br>
* `clear` (清除螢幕顯示)<br>
* `history` (列出操作記錄)<br>

---
### install Python3 on Windows PC

**Python3.11.7 for Windows**<br>
1. Go to [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Download Windows installer (64-bit)<br>
3. Customomize installation to set directory to `C:\Python3`
![](https://img-blog.csdnimg.cn/e6f8a219eefa4bc29b3c120bc4afdbc3.png)

---
### checking Python version on Ubuntu PC
**Ubuntu 20.04 LTS**<br>
`$ python3 -V`<br>
Python 3.8.10

**Ubuntu 22.04 LTS**<br>
`$ python3 -V`<br>
Python 3.10.4

---
### List of Python packages for installation
* Open GitBash / Ubuntu Terminal<br>
`python3 -V`<br>
`python3 –m pip install --upgrade pip`<br>
`pip -V`<br>
`pip install jupyter`<br>
`pip install pandas`<br>
`pip install matplotlib pillow imutils`<br>
`pip install opencv-python`<br>
`pip install scikit-learn`<br>
`git clone https://github.com/rkuo2000/cv2`<br>
`git clone https://github.com/rkuo2000/tf`<br>

---
### install Tensorflow
1. using pip to install tensorflow<br>
`pip install tensorflow`<br>

2. using anaconda to install tensorflow<br>
`$ conda activate tensor`<br>
`(tensor) $ conda install tensorflow`<br>

---
### install PyTorch
* [PyTorch get-started](https://pytorch.org/get-started/locally/)<br>
`pip install torch torchvision`<br>

---
## Learn Programming

### [Python Programming](https://www.programiz.com/python-programming)

### [Tensorflow Turorials](https://www.tensorflow.org/tutorials)

### [PyTorch Tutorials](https://pytorch.org/tutorials/)

---
## Supplement

### Anaconda3  
*(用於安裝一串版本相容的Python packages)*<br>

**Anaconda3 on Windows**<br>
* Download [Distribution](https://www.anaconda.com/products/distribution)
* [Python 初學者的懶人包 Anaconda 下載與安裝](https://walker-a.com/archives/6260)<br>

**Anaconda3 on Ubuntu**<br>
[How to Install Anaconda on Ubuntu 18.04 and 20.04](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04)<br>
* download Anaconda3<br>
`$ curl -O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh`<br>
* install Anaconda3<br>
`$ bash Anaconda3-2021.11-Linux-x86_64.sh`<br>
* create env<br>
`(base) $ conda create -n tensor python=3.9`<br>
* activate env<br>
`(base) $ conda activate tensor`<br>
* deactivate env<br>
`(tensor) $ conda deactivate`<br>
* remove an env<br>
`(base) $ conda-env remove -n tensor`<br> 

---
### GPU acceleration
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
  - [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
* [CuDNN](https://developer.nvidia.com/cudnn)
  - [cuDNN installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

---
## Google Colab

### Google Colab 教學
[Google Colab 教學 (1)｜Python 雲端開發環境安裝與快速導覽](https://medium.com/python4u/google-colab-%E6%95%99%E5%AD%B8-1-python-%E9%9B%B2%E7%AB%AF%E9%96%8B%E7%99%BC%E7%92%B0%E5%A2%83%E5%AE%89%E8%A3%9D%E8%88%87%E5%BF%AB%E9%80%9F%E5%B0%8E%E8%A6%BD-78942200525f)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

