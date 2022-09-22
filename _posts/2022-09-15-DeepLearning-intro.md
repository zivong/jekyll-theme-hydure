---
layout: post
title: Deep Learning Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Deep Learning is a broader family of machine learning methods based on artificial neural networks.

---
## AI Hardware Acceleration

* Google TPU Cloud
<p align="center"><img src="https://miro.medium.com/max/588/0*tchhZPBigPBYbWRp.png"></p>

* GPU Server
<p align="center"><img src="https://www.leadtek.com/images/news/20190527_1_en.jpg" width="50%" height="50%"></p>

* GPU workstation
<p align="center"><img src="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/dgx-station-a100/nvidia-dgx-station-og.jpg" width="80%" height="80%"></p>

* GPU Computer
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Quad-GeForce-GTX-Titan-Black-Ultimate-GPU-Gaming-Computer.png" width="30%" height="50%"></p>

### Embedded Board
* **Nvidia Jetson**
![](https://www.nvidia.com/content/dam/en-zz/Solutions/intelligent-machines/embedded-systems/jetson-commercial-journey-og.jpg)

* **Cannan Kendryte**
<table>
<tr>
<td><img src="https://a.rimg.com.tw/s0/0bf/4e8/avqscf/c/00/cb/22031982595275_696.jpg"></td>
<td><img src="https://img.alicdn.com/imgextra/i3/4161557473/O1CN01E0vdMs254gYFHg5vV_!!4161557473.png"></td>
</tr>
</table>

### USB Doggle
<table>
  <tr>
  <td><img src="https://imgur.com/2g7eTms.png"></td>
  <td><img src="https://lh3.googleusercontent.com/vvBAqSnXyg3h9yS0JLyVehhV-e__3NFbZ6q7Ft-rEZp-9wDTVZ49yjuYJwfa4jQZ-RVnChHMr-DDC0T_fTxVyQg3iBMD-icMQooD6A=w630-rw"></td>
  </tr>
</table>

* CAM Stick 
M5Stack StickV K210 AI攝像機開發板
<img width="35%" height="35%" src="https://cf.shopee.tw/file/e42a372f356174a1e5261d3a16aaf078">

---
## AI chips
### [10 coolest AI chips of 2021](https://www.crn.com/slide-shows/components-peripherals/the-10-coolest-ai-chips-of-2021-so-far-/1)
* Ambarella CV52S
* Atlazo AZ-N1
* AWS Trainium
* Cerebras Wafer Scale Engine 2 - 850,000 cores and 40GB of on-chip memory
* Google TPU v4 : 4096 TPU v4 cores, 1.1exaflops
* 3rd-Gen Intel Xeon Scalable (Ice Lake)
* Mythic M1076 Analog Matrix Processor : 25 trillion operations per seconds
* Nvidia A100
* Syntiant NDP120 - supports more than 7 million parameters 
* Xilinx Versal AI Edge

---
### Tesla D1 chip
[Enter Dojo: Tesla Reveals Design for Modular Supercomputer & D1 Chip](https://www.hpcwire.com/2021/08/20/enter-dojo-tesla-reveals-design-for-modular-supercomputer-d1-chip/)
<table>
  <tr>
    <td><img src="https://6lli539m39y3hpkelqsm3c2fg-wpengine.netdna-ssl.com/wp-content/uploads/2021/08/d1-chip-tesla-300x204.png"/></td>
    <td><img src="https://6lli539m39y3hpkelqsm3c2fg-wpengine.netdna-ssl.com/wp-content/uploads/2021/08/training-tile-tesla-300x206.png"/></td>
  </tr>
</table>
![](https://6lli539m39y3hpkelqsm3c2fg-wpengine.netdna-ssl.com/wp-content/uploads/2021/08/exapod-768x250.png)

<font size="3">
With each D1 chip providing 22.6 teraflops of FP32 performance, <br>
each training tile will provide 565 teraflops and each cabinet (containing 12 tiles) will provide 6.78 petaflops - <br>
meaning that one ExaPOD alone will deliver a maximum theoretical performance of 67.8 FP32 petaflops. <br>
</font>

[Tesla details Dojo supercomputer, reveals Dojo D1 chip and training tile module](https://www.datacenterdynamics.com/en/news/tesla-details-dojo-supercomputer-reveals-dojo-d1-chip-and-training-tile-module/)
<table>
  <tr>
  <td><img src="https://i0.wp.com/semianalysis.com/wp-content/uploads/2021/08/training-tile-2.png?resize=800%2C445&ssl=1" /></td>
  <td><img src="https://media.datacenterdynamics.com/media/images/training_tiles_III.original.png" /></td>
  </tr>
</table>

---
### Google TPU
* [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
* Cloud TPU VM architectures
![](https://cloud.google.com/tpu/docs/images/tpu-pod-architecture.png)
* Ref. [Hardware for Deep Learning Part4: ASIC](https://blog.inten.to/hardware-for-deep-learning-part-4-asic-96a542fe6a81)
![](https://miro.medium.com/max/2000/1*iOWhkTnD0uXnpQ2Y_viHdA.png)
* TPU v4 
![](https://1.bp.blogspot.com/-E_4XkiTpfik/YLInSqPOY6I/AAAAAAAAAYo/h9FR2niT-yMZJgLUcLi03C2w-4yFCUwFgCLcBGAsYHQ/s16000/Google-TPU-v4.png)
* TPU v3 Block Diagram
![](https://1.bp.blogspot.com/-eVdyyxSonCM/YLIp1D90koI/AAAAAAAAAZo/yn_8_Ku9ReglUf_-UKVxQh8Nidpi1iFJwCLcBGAsYHQ/w640-h350/TPU-v3-Block-Diagram.png)
* TPU Block Diagram
![](https://miro.medium.com/max/700/1*9uNlFIx5Uic2hoC4jIV6hg.png)

---
### Nvida GPUs
* V100 (DataCenter GPU)
![](https://www.nvidia.com/content/dam/en-zz/es_em/es_em/Solutions/Data-Center/tesla-v100/data-center-tesla-v100-pcie-625-ud@2x.jpg)
* A100 (DataCenter GPU)
![](https://s4.itho.me/sites/default/files/images/Nvidia%20A100%20PCIe.jpg)
![](https://images.contentstack.io/v3/assets/blt71da4c740e00faaa/blt2d1a709708601dc9/6021bac59a7bfd14d273254d/Slide3-1-1024x576.png)
* DXG2 (Workstation)
![](https://images.anandtech.com/doci/12587/screenshot_61_575px.png)
* H100 

**Blog:** [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)<br>
![](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/GTC2022_SXM5_01_v001_DL-1536x864.png)
![](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Perf-Main-FINAL-768x325.jpg)

---
### RTX4090
![](https://www.pcworld.pl/g1/news/thumbnails/4/1/415701_RTX_4090_webp_80_resize_750x329.webp)
<iframe width="750" height="422" src="https://www.youtube.com/embed/cJlo2I7CiD0" title="Microsoft Flight Simulator | NVIDIA DLSS 3 - Exclusive First-Look" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### CUDA & CuDNN
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
* [CuDNN](https://developer.nvidia.com/cudnn)
![](https://developer.nvidia.com/sites/default/files/akamai/cudnn/cudnn_chart.png)

---
### Nivida Jetson 
![](https://www.fastcompression.com/img/jetson/nvidia-jetson-modules2.png)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Jetson_modules_Hardware_comparison.png?raw=true)
* [Jetson AGX Orin 32GB module](https://www.nvidia.com/zh-tw/autonomous-machines/embedded-systems/jetson-orin/) 275 TOPS
![](https://www.nvidia.com/content/dam/en-zz/Solutions/intelligent-machines/jetson-orin/jetson-orin-modules-2c50-d.jpg)
* [Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
* [Getting the Best Performance on MLPerf Inference 2.0](https://developer.nvidia.com/blog/getting-the-best-performance-on-mlperf-inference-2-0/)

---
### Jetson Orin Nano
It has up to eight streaming multiprocessors (SMs) composed of 1024 CUDA cores and up to 32 Tensor Cores for AI processing.
![](https://developer-blogs.nvidia.com/wp-content/uploads/2022/09/Block-diagram-of-Jetson-Orin-Nano-625x508.png)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Jetson_Orin_Nano.png?raw=true)

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
## [AI on Chip Taiwan Alliance (台灣人工智慧晶片聯盟)](https://www.aita.org.tw/)
![](https://www.aita.org.tw/images/sig2-1.png)
<br>
![](https://www.aita.org.tw/images/sig2-2.png)

---
## ML Benchmark: 
<p align="center"><img src="https://www.servethehome.com/wp-content/uploads/2021/06/Google-TPU-v4-MLPerf-v1.0-Tope-Line-Results-1536x1536.png" width="70%" height="70%"></p>

### [MLPerf](https://mlcommons.org/en/)
* MLPerf Inference v1.1 Results<br>
&emsp;[inference-datacenter v1.1 results](https://mlcommons.org/en/inference-datacenter-11/)<br>
&emsp;[inference-edge v1.1 results](https://www.mlcommons.org/en/inference-edge-11/)<br>
* [MLPerf Training v1.0 Results](https://mlcommons.org/en/training-normal-10/)<br>
* [MLPerf Tiny Inference Benchmark](https://mlcommons.org/en/inference-tiny-05)<br>

---
## Framework
### [PyTorch](https://pytorch.org)
![](https://miro.medium.com/max/1400/1*agu5YjWbY1RXhWHZJLZwbw.png)

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

