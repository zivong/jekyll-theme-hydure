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
![](https://storage.googleapis.com/gweb-cloudblog-publish/images/2_Cloud_TPU_v4.max-1400x1400.jpg)
TPU v4 is the first supercomputer to deploy a reconfigurable OCS. OCSes dynamically reconfigure their interconnect topology
Much cheaper, lower power, and faster than Infiniband, OCSes and underlying optical components are <5% of TPU v4’s system cost and <5% of system power.

---
## Nvidia

### CUDA & CuDNN
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
* [CuDNN](https://developer.nvidia.com/cudnn)
![](https://developer.nvidia.com/sites/default/files/akamai/cudnn/cudnn_chart.png)

---
### AI Workstations/Servers
[生成式 AI 動力核心，NVIDIA 宣布 DGX H100 系統開始出貨](https://technews.tw/2023/05/02/nvidia-announces-shipping-of-dgx-h100-systems/)<br>
**DGX H100** 系統每個 H100 Tensor Core GPU 性能平均比以前 GPU 高約 6 倍，搭載 8 個 GPU，每個 GPU 都有一個 Transformer Engine，加速生成式 AI 模型。8 個 H100 GPU 透過 NVIDIA NVLink 連接，形成巨大 GPU，也可擴展 DGXH100 系統，使用 400 Gbps 超低延遲 NVIDIA Quantum InfiniBand 將數百個 DGX H100 節點連線一台 AI 超級電腦，速度是之前網路的兩倍。
![](https://img.technews.tw/wp-content/uploads/2022/03/23023452/NVIDIA-H100-1-JP.jpg)

---
### AI HPC
[搭配HGX A100模組，華碩發表首款搭配SXM形式GPU伺服器](https://www.ithome.com.tw/review/147911)
![](https://s4.itho.me/sites/default/files/images/Asus%20ESC%20N4A-E11-2022-09-4.jpg)

---
### GPU
[GeForce RTX-4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889)
![](https://tpucdn.com/gpu-specs/images/c/3889-back.jpg)

---
### Edge AI
![](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-agx-orin-family-4c25-p@2x.jpg)

![](https://www.seeedstudio.com/blog/wp-content/uploads/2022/07/NVIDIA-Jetson-comparison_00.png)

![](https://global.discourse-cdn.com/nvidia/original/3X/0/b/0bfc1897e20cca7f4eac2966f2ad5829d412cbeb.jpeg)
![](https://global.discourse-cdn.com/nvidia/optimized/3X/5/a/5af686ee3f4ad71bc44f22e4a9323fe68ed94ba8_2_690x248.jpeg)

---
## AI chips
### [Top 10 AI Chip Makers of 2023: In-depth Guide](https://research.aimultiple.com/ai-chip-makers/)
1. Nvidia
DGX™ A100 is the flagship AI chip of Nvidia which is also designed for data centers. Product integrates 8 GPUs and up to 640GB GPU memory.

2. Intel
Intel® NCS2 is the latest AI chip from Intel and was developed specifically for deep learning. 

3. Google Alphabet
Google Cloud TPU is the purpose-built machine learning accelerator chip that powers Google products like Translate, Photos, Search, Assistant, and Gmail

4. AMD
AMD launched MI300 for AI training workloads in June 2023

5. IBM
IBM launched its “neuromorphic chip” TrueNorth AI in 2014. TrueNorth contains 5.4 billion transistors, 1 million neurons, and 256 million synapses, so it can efficiently perform deep network inference and deliver high-quality data interpretation.

![](https://research.aimultiple.com/wp-content/webp-express/webp-images/uploads/2022/03/AI-Chip-Companies-total-fundings.png.webp)

6. SambaNova Systems
founded in 2017, the company has developed the SN10 processor chip and raised more than $1.1 billion in funding.

7. Cerebras Systems
founded in 2015, In April 2021, the company announced its new AI chip model, Cerebras WSE-2, which has 850,000 cores and 2.6 trillion transistors.

8. Graphcore
Graphcore is a British company founded in 2016. The company announced its flagship AI chip as IPU-POD256. Graphcore has already been funded with around $700 million.

9. Groq
Groq has been founded by former Google employees. The company represents a new model for AI chip architecture that aims to make it easier for companies to adopt their systems. The startup has already raised around $350 million and produced its first models such as GroqChip™ Processor, GroqCard™ Accelerator, etc.

10. Mythic
Mythic was founded in 2012. It developed products such as M1076 AMP, MM1076 key card, etc., and has already raised about $150 million in funding.

---
### Tesla D1 chip
[Enter Dojo: Tesla Reveals Design for Modular Supercomputer & D1 Chip](https://www.hpcwire.com/2021/08/20/enter-dojo-tesla-reveals-design-for-modular-supercomputer-d1-chip/)
With each D1 chip providing 22.6 teraflops of FP32 performance, <br>
each training tile will provide 565 teraflops and each cabinet (containing 12 tiles) will provide 6.78 petaflops - <br>
meaning that one ExaPOD alone will deliver a maximum theoretical performance of 67.8 FP32 petaflops. <br>

[Tesla details Dojo supercomputer, reveals Dojo D1 chip and training tile module](https://www.datacenterdynamics.com/en/news/tesla-details-dojo-supercomputer-reveals-dojo-d1-chip-and-training-tile-module/)
![](https://media.datacenterdynamics.com/media/images/training_tiles_III.width-358.png)

---
### Kneron 耐能智慧
* KL530 AI SoC
![](https://www.kneron.com/tw/_upload/image/solution/large/938617699868711f.jpg)
  - 基於ARM Cortex M4 CPU内核的低功耗性能和高能效設計。
  - 算力達1 TOPS INT 4，在同等硬件條件下比INT 8的處理效率提升高達70%。
  - 支持CNN,Transformer，RNN Hybrid等多種AI模型。
  - 智能ISP可基於AI優化圖像質量，強力Codec實現高效率多媒體壓縮。
  - 冷啟動時間低於500ms，平均功耗低於500mW。
<br>
<br>
* KL720 AI SoC (算力可達0.9 TOPS/W)
![](https://www.kneron.com/tw/_upload/image/solution/large/95f4758c9cfd08.png)
  - 基於ARM Cortex M4 CPU内核的低功耗性能和高能效設計
  - 可適配高端IP攝像頭，智能電視，AI眼鏡、耳機以及AIoT網絡的終端設備。 
  - 可處理高達4K圖像，全高清影音和3D感應，實現精準的臉部識別以及手勢控制。 
  - 可為翻譯機和AI助手等產品提供自然語言處理。 
  - 以上各種功能以及其它邊緣AI — 例如感熱 — 均可實時處理。 
<br>

---
### Realtek AmebaPro2
[AMB82-MINI](https://www.amebaiot.com/en/amebapro2/#rtk_amb82_mini)<br>
![](https://www.amebaiot.com/wp-content/uploads/2022/06/AMB82-MINI-2048x1489.jpg)
* MCU
  - Part Number: RTL8735B
  - 32-bit Arm v8M, up to 500MHz
* MEMORY
  - 768KB ROM
  - 512KB RAM
  - Supports MCM embedded DDR2/DDR3L memory up to 128MB
  - External Flash up to 64MB
* KEY FEATURES
  - Integrated 802.11 a/b/g/n Wi-Fi, 2.4GHz/5GHz
  - Bluetooth Low Energy (BLE) 4.2
  - Integrated Intelligent Engine @ 0.4 TOPS
  - Ethernet Interface
  - USB Host/Device
  - SD Host
  - ISP
  - Audio Codec
  - H.264/H.265
  - Secure Boot
  - Crypto Engine
* OTHER FEATURES
  - 2 SPI interfaces
  - 1 I2C interface
  - 8 PWM interfaces
  - 3 UART interfaces
  - 3 ADC interfaces
  - 2 GDMA interfaces
  - Max 23 GPIO
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

### [Tensorflow Lite](https://www.tensorflow.org/lite)

### [Tensorflow Lite for Microcontroller](https://www.tensorflow.org/lite/microcontrollers)

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

