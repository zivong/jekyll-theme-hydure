---
layout: post
title: AI Platform Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

AI Hardware, AI chips, ML Benchrmark, Framework, Open platforms

---
## AI Hardware

### Google TPU Cloud
[Google’s Cloud TPU v4 provides exaFLOPS-scale ML with industry-leading efficiency](https://cloud.google.com/blog/topics/systems/tpu-v4-enables-performance-energy-and-co2e-efficiency-gains)
![](https://storage.googleapis.com/gweb-cloudblog-publish/images/1_Cloud_TPU_v4.max-1100x1100.jpg)
One eighth of a TPU v4 pod from Google's world’s largest publicly available ML cluster located in Oklahoma, which runs on ~90% carbon-free energy.<br>
<br>
<p><img src="https://storage.googleapis.com/gweb-cloudblog-publish/images/2_Cloud_TPU_v4.max-1400x1400.jpg" width="50%" height="50%"></p>
TPU v4 is the first supercomputer to deploy a reconfigurable OCS. OCSes dynamically reconfigure their interconnect topology
Much cheaper, lower power, and faster than Infiniband, OCSes and underlying optical components are <5% of TPU v4’s system cost and <5% of system power.

---
## Nvidia

### CUDA & CuDNN
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
* [CuDNN](https://developer.nvidia.com/cudnn)
![](https://developer.nvidia.com/sites/default/files/akamai/cudnn/cudnn_chart.png)

---
### AI SuperComputer
**DGX GH200**<br>
![](https://www.storagereview.com/wp-content/uploads/2023/06/storagereview-nvidia-dgx-gh200-1.jpg)
![](https://www.storagereview.com/wp-content/uploads/2023/06/Screenshot-2023-06-26-at-11.28.32-AM.png)

### AI Data Center
**HGX H200**<br>
<p><img src="https://cdn.videocardz.com/1/2023/11/NVIDIA-H200-Overview-1536x747.jpg"></p>

### AI Workstatione/Server (for Enterprise)
**DGX H100**<br>
系統每個 H100 Tensor Core GPU 性能平均比以前 GPU 高約 6 倍，搭載 8 個 GPU，每個 GPU 都有一個 Transformer Engine，加速生成式 AI 模型。8 個 H100 GPU 透過 NVIDIA NVLink 連接，形成巨大 GPU，也可擴展 DGXH100 系統，使用 400 Gbps 超低延遲 NVIDIA Quantum InfiniBand 將數百個 DGX H100 節點連線一台 AI 超級電腦，速度是之前網路的兩倍。
![](https://img.technews.tw/wp-content/uploads/2022/03/23023452/NVIDIA-H100-1-JP.jpg)

---
### AI HPC
**HGX A100**<br>
[搭配HGX A100模組，華碩發表首款搭配SXM形式GPU伺服器](https://www.ithome.com.tw/review/147911)<br>
<p><img src="https://s4.itho.me/sites/default/files/images/ESC%20N4A-E11_2D%20top%20open.jpg"></p>

---
### GPU
[GeForce RTX-4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889)
<p><img src="https://www.pcworld.com/wp-content/uploads/2023/04/geforce-rtx-4090-jensen.jpg" width="50%" height="50%"></p>

---
### AI PC/Notebook
**NPU**: [三款AI PC筆電搶先看！英特爾首度在臺公開展示整合NPU的Core Ultra筆電，具備有支援70億參數Llama 2模型的推論能力](https://www.ithome.com.tw/news/159673)<br>
![](https://s4.itho.me/sites/default/files/images/PXL_20231107_003902969.jpg)
宏碁在現場展示用Core Ultra筆電執行圖像生成模型，可以在筆電桌面螢幕中自動生成動態立體的太空人桌布，還可以利用筆電前置鏡頭來追蹤使用者的臉部輪廓，讓桌布可以朝著使用者視角移動。此外，還可以利用工具將2D平面圖像轉為3D裸眼立體圖。

---
### Edge AI
![](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-agx-orin-family-4c25-p@2x.jpg)

![](https://www.seeedstudio.com/blog/wp-content/uploads/2022/07/NVIDIA-Jetson-comparison_00.png)

![](https://global.discourse-cdn.com/nvidia/original/3X/0/b/0bfc1897e20cca7f4eac2966f2ad5829d412cbeb.jpeg)
![](https://global.discourse-cdn.com/nvidia/optimized/3X/5/a/5af686ee3f4ad71bc44f22e4a9323fe68ed94ba8_2_690x248.jpeg)

---
## AI chips

### [Top 10 AI Chip Makers of 2023: In-depth Guide](https://research.aimultiple.com/ai-chip-makers/)

#### [Tesla AI](https://www.tesla.com/AI)
[Enter Dojo: Tesla Reveals Design for Modular Supercomputer & D1 Chip](https://www.hpcwire.com/2021/08/20/enter-dojo-tesla-reveals-design-for-modular-supercomputer-d1-chip/)<br>

[Teslas will be 'more intelligent' than HUMANS by 2033 as their microchips already have 36% the capacity of the brain, study reveals](https://www.dailymail.co.uk/sciencetech/article-11206325/Teslas-smarter-humans-2033-microchips-handle-362T-operations-second.html)<br>

---
### [Collections of MPU for Edge AI applications](https://makerpro.cc/2023/08/collections-of-mcu-for-edge-ai-applications/)
#### [天璣 9300](https://www.mediatek.tw/products/smartphones-2/mediatek-dimensity-9300)
* 單核性能提升超過 15%
* 多核性能提升超過 40%
* 4 個 Cortex-X4 CPU 主頻最高可達 3.25GHz
* 4 個 Cortex-A720 CPU 主頻為 2.0GHz
* 內置 18MB 超大容量緩存組合，三級緩存（L3）+ 系統緩存（SLC）容量較上一代提升 29%

#### [天璣 8300](https://www.mediatek.tw/products/smartphones-2/mediatek-dimensity-8300)
* 八核 CPU 包括 4 個 Cortex-A715 大核和 4 個 Cortex-A510 能效核心
* Mali-G615 GPU
* 支援 LPDDR5X 8533Mbps 記憶體
* 支援 UFS 4.0 + 多循環隊列技術（Multi-Circular Queue，MCQ）
* 高能效 4nm 製程

#### ADI MAX78000
![](https://i0.wp.com/makerpro.cc/wp-content/uploads/2023/08/EdgeAI_MCU_P1.jpg?resize=1024%2C414&ssl=1)

#### TI MPU: AM62A、AM68A、AM69A
![](https://i0.wp.com/makerpro.cc/wp-content/uploads/2023/08/1691657090157.jpg?resize=768%2C607&ssl=1)

---
### Kneron 耐能智慧
* KNEO300 EdgeGPT
<p><img src="https://image-cdn.learnin.tw/bnextmedia/image/album/2023-11/img-1701333658-39165.jpg" width="50%" height="50%"></p>

* KL530 AI SoC
![](https://www.kneron.com/tw/_upload/image/solution/large/938617699868711f.jpg)
  - 基於ARM Cortex M4 CPU内核的低功耗性能和高能效設計。
  - 算力達1 TOPS INT 4，在同等硬件條件下比INT 8的處理效率提升高達70%。
  - 支持CNN,Transformer，RNN Hybrid等多種AI模型。
  - 智能ISP可基於AI優化圖像質量，強力Codec實現高效率多媒體壓縮。
  - 冷啟動時間低於500ms，平均功耗低於500mW。

* KL720 AI SoC (算力可達0.9 TOPS/W)
![](https://www.kneron.com/tw/_upload/image/solution/large/95f4758c9cfd08.png)

---
### Realtek AmebaPro2
[AMB82-MINI](https://www.amebaiot.com/en/amebapro2/#rtk_amb82_mini)<br>
<p><img src="https://www.amebaiot.com/wp-content/uploads/2023/03/amb82_mini.png" width="50%" height="50%"></p>
* MCU
  - Part Number: RTL8735B
  - 32-bit Arm v8M, up to 500MHz
* MEMORY
  - 768KB ROM
  - 512KB RAM
  - 16MB Flash
  - Supports MCM embedded DDR2/DDR3L memory up to 128MB
* KEY FEATURES
  - Integrated 802.11 a/b/g/n Wi-Fi, 2.4GHz/5GHz
  - Bluetooth Low Energy (BLE) 5.1
  - Integrated Intelligent Engine @ 0.4 TOPS
<iframe width="580" height="327" src="https://www.youtube.com/embed/_Kzqh6JXndo" title="AIoT: AmebaPro2 vs ESP32" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## [mlplatform.org](https://www.mlplatform.org/)
The machine learning platform is part of the Linaro Artificial Intelligence Initiative and is the home for Arm NN and Compute Library – open-source software libraries that optimise the execution of machine learning (ML) workloads on Arm-based processors.
![](https://www.mlplatform.org/assets/images/assets/images/content/NN-frameworks20190814-800-1d11a6.webp)
<table>
  <tr><td>Project</td><td>Repository</td></tr>
  <tr><td>Arm NN</td><td>[https://github.com/ARM-software/armnn](https://github.com/ARM-software/armnn)</td></tr>
  <tr><td>Compute Library</td><td>[https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary](https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary)</td></tr>
  <tr><td>Arm Android NN Driver</td><td>https://github.com/ARM-software/android-nn-driver</td></tr>
</table>

---
### [ARM NN SDK](https://www.arm.com/zh-TW/products/silicon-ip-cpu/ethos/arm-nn)
免費提供的 Arm NN (類神經網路) SDK，是一組開放原始碼的 Linux 軟體工具，可在節能裝置上實現機器學習工作負載。這項推論引擎可做為橋樑，連接現有神經網路框架與節能的 Arm Cortex-A CPU、Arm Mali 繪圖處理器及 Ethos NPU。<br>

**[ARM NN](https://github.com/ARM-software/armnn)**<br>
Arm NN is the most performant machine learning (ML) inference engine for Android and Linux, accelerating ML on Arm Cortex-A CPUs and Arm Mali GPUs.

---
## Benchmark
### [MLPerf](https://mlcommons.org/en/)

### [MLPerf™ Inference Benchmark Suite](https://github.com/mlcommons/inference)
MLPerf Inference v3.1 (submission 04/08/2023)

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm-v2 | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail |

---
### [NVIDIA’s MLPerf Benchmark Results](https://www.nvidia.com/en-us/data-center/resources/mlperf-benchmarks/)
**NVIDIA H100 Tensor Core GPU**<br>

| Benchmark                                  | Per-Accelerator Records |
|--------------------------------------------|-------------------------|
| Large Language Model (LLM)                 | 548 hours (23 days) |
| Natural Language Processing (BERT)         | 0.71 hours |
| Recommendation (DLRM-dcnv2)                | 0.56 hours |
| Speech Recognition (RNN-T)                 | 2.2 hours | 
| Image Classification (ResNet-50 v1.5)      | 1.8 hours |
| Object Detection, Heavyweight (Mask R-CNN) | 2.6 hours |
| Object Detection, Lightweight (RetinaNet)  | 4.9 hours |
| Image Segmentation (3D U-Net)              | 1.6 hours |

---
## Framework
### [PyTorch](https://pytorch.org)

### [Tensorflow](https://www.tensorflow.org)

### [Keras 3.0](https://keras.io/keras_3/)
<p><img src="https://s3.amazonaws.com/keras.io/img/keras_3/cross_framework_keras_3.jpg" width="50%" height="50%"></p>
All 40 Keras Applications models (the keras.applications namespace) are available in all backends. The vast array of pretrained models in KerasCV and KerasNLP also work with all backends. This includes: **BERT, OPT, Whisper, T5, StableDiffusion, YOLOv8, SegmentAnything, etc.**

### TinyML
[EloquentTinyML](https://github.com/eloquentarduino/EloquentTinyML)

### [Tensorflow.js](https://www.tensorflow.org/js/demos)

### [MediaPipe](https://google.github.io/mediapipe/)

---
## Open Platforms
<table>
  <tr>
    <td><a href="https://kaggle.com"><img src="https://www.kaggle.com/static/images/site-logo.png"></a></td>
    <td><a href="https://gym.openai.com/"><img src="https://i.pinimg.com/474x/a8/5e/11/a85e111c643e20543f1a5283a2de835c.jpg"></a></td>
  </tr>
</table>
<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

