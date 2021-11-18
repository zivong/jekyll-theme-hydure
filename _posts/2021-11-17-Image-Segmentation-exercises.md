---
layout: post
title: Image Segmentation Exercises
author: [Richard Kuo]
category: [Example]
tags: [jekyll, ai]
---

*Projects includes DeeplabV3+, Semantic Segmentation in PyTorch, Semantic Segmentation for VTON, Panoptic Driving Perception.*

---
## Semantic Segmentation
### DeepLabV3 plus
**kaggle:** [rkuo2000/deeplabv3-plus](https://kaggle.com/rkuo2000/deeplabv3-plus)
![](https://www.kaggleusercontent.com/kf/80043630/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..TaRXVZa2nWY3RM9FNo1wew.ksZ4zo01xRQOqEEuW0UNLOUj-TIfNEPANY5NMCMvmInhMAOPUm7UQJEYF-lifqiVt_cglG_1-7Pp27c-IIfo5FaqsjBfRMs4qfUaQndh-jDCyAe_EzNjJcrNyn4xsX3z7pvwm9EV39KGgXK3f_AEa48hvgNvwlv8rmR5lJp8mpsdlx2ogj2ohI3OhgsCGooSH6NtUo4cPYlQBJvfnFC7I-MdVaGMEqrXSJvLgvEcMWjBagC2eRM7w7VwYjyvZn7RqKQkN9K4pxFcqirxPb4fd2hf7Hs5kyUzFnpMvWv9n-JzgOZnPOrJJ52xc9FtAXyBzsOIRkjk7jZEfQLGT5sowSWCrwYfg1_kQY7zbf85LzZWm2ZtUvQZvHNf2RC6EDNp8-SC8lOz7DoA9kKyOhYKkPG8KO0TbjF1KJ0pQOIyyt_Uhuv_lJjTf47WNsCWiry9Bf3QTPjJi9tOpEqTPEk1ibNv2vXDQfOotkt9p_JWITxPBqsORSsRm1Ae9Bz0F97HDRAaG4hEf-9axgMQzHhpiAPyXCbQ8xSL7rR5SmgJnMJFsyUtKMBNYzTp8iooB2DY_2v1vwtVgEvD4uqvVOEfkIAezqjWjsNLOqqKxm3dytWwfPBn3yOi3xbmXxcb6WMALkWD7HjDTyLGCaNwwsa_3A.CVKYm1Iq4cjvOeH12JQ2Ww/__results___files/__results___23_0.png)

---
### Image Segmentation Keras (PSPNet)
**Kaggle:** [rkuo2000/image-segmentation-keras](https://kaggle.com/rkuo2000/image-segmentation-keras)
![](https://www.kaggleusercontent.com/kf/80046633/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..HLC6AKGc2RD6xdvM7bipug.cOacCE4v_dk50nX5dWl7-KHdKqH3jZFUU0RXyH80UH4r9OdfYlw-wIghcEBxz1XDBZpSJedApXr_s6CfsntQeMOmP_fcoTZ1vEIHFWL_3cE7Azf_7_26NjyKjQSwTd_UWTC-xCIyvykQ28R6RIOsiCXhew7ca7jQRboyQ07gDHoQk8KCsjcL-6Yaj2lxAtCxswwj6x_hzJIzjkSxpMKJTaSruYZ6DLOQbn0vHIap3-GGLWAk2pO_aVSREQLnM4RB_3Mo25A7Stw8ZcYI-OkEOkcOdRAJn5D5KHr6X8X7sLgCCz-pFWaHGjMjsQ2hgga4AEl-BtLYIZ7A1SA_hQj0YuiHZ9DRAwDnrxmT6uW8GBjAa5JejGzQKRDLGbE7nZJm0nc2FiH5uu6ngcDPCb9S9atxScKrnAbVaUwgvL4yuhiZhQ6_oO9TDeiM_5RJAPEQ8N9V_HAVMz5cTSQel2ti7c0hRYLzf-u9mZiLjwSojSaZ7N5ycrSDNpA_mBbB_U0RYBbG0FrcFjQArVvsOTKbQD1GsNjot_TH6tmON4o1dK7deeo5D-thAOjz7gEkIMvviiuTHSkcLec5Tz0RZcDDp-UeJgGrLVPIUrm0HzNws0Rp0-M8rMeQi1fU3rJA8fePe6TYQuaSAJSI-hqK9eBm-Iy53FFCPytyluhIw7LyBr0.OVLRT9TfU8BeByRijddOMg/__results___files/__results___12_0.png)
![](https://www.kaggleusercontent.com/kf/80046633/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..HLC6AKGc2RD6xdvM7bipug.cOacCE4v_dk50nX5dWl7-KHdKqH3jZFUU0RXyH80UH4r9OdfYlw-wIghcEBxz1XDBZpSJedApXr_s6CfsntQeMOmP_fcoTZ1vEIHFWL_3cE7Azf_7_26NjyKjQSwTd_UWTC-xCIyvykQ28R6RIOsiCXhew7ca7jQRboyQ07gDHoQk8KCsjcL-6Yaj2lxAtCxswwj6x_hzJIzjkSxpMKJTaSruYZ6DLOQbn0vHIap3-GGLWAk2pO_aVSREQLnM4RB_3Mo25A7Stw8ZcYI-OkEOkcOdRAJn5D5KHr6X8X7sLgCCz-pFWaHGjMjsQ2hgga4AEl-BtLYIZ7A1SA_hQj0YuiHZ9DRAwDnrxmT6uW8GBjAa5JejGzQKRDLGbE7nZJm0nc2FiH5uu6ngcDPCb9S9atxScKrnAbVaUwgvL4yuhiZhQ6_oO9TDeiM_5RJAPEQ8N9V_HAVMz5cTSQel2ti7c0hRYLzf-u9mZiLjwSojSaZ7N5ycrSDNpA_mBbB_U0RYBbG0FrcFjQArVvsOTKbQD1GsNjot_TH6tmON4o1dK7deeo5D-thAOjz7gEkIMvviiuTHSkcLec5Tz0RZcDDp-UeJgGrLVPIUrm0HzNws0Rp0-M8rMeQi1fU3rJA8fePe6TYQuaSAJSI-hqK9eBm-Iy53FFCPytyluhIw7LyBr0.OVLRT9TfU8BeByRijddOMg/__results___files/__results___16_0.png)

---
### Semantic Segmentation on PyTorch
**Kaggle:** [rkuo2000/semantic-segmentation-on-pytorch](https://www.kaggle.com/rkuo2000/semantic-segmentation-on-pytorch)<br />

![](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/weimar_000091_000019_gtFine_color.png?raw=True)

**Github:** [Tramac/awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)<br />
**Datasets:** [[ADE20K]](http://groups.csail.mit.edu/vision/datasets/ADE20K/), [[CityScapes]](https://www.cityscapes-dataset.com/), [[Pascal VOC]](http://host.robots.ox.ac.uk/pascal/VOC/), [[SBU-shadow]](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)<br />
**Models:** [Model & Backbone](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/DETAILS.md)

---
### Semantic Segmentation for VITON
Train PSPNet with VITON dataset & Test it

**Kaggle**: [rkuo2000/semantic-segmentation-viton-train](https://www.kaggle.com/rkuo2000/semantic-segmentation-viton-train)<br />
&emsp;&emsp;&emsp;&emsp;[rkuo2000/semantic-segmentation-viton-detect](https://www.kaggle.com/rkuo2000/semantic-segmentation-viton-detect)<br />

**Github:** [IanTaehoonYoo/semantic-segmentation-pytorch](https://github.com/IanTaehoonYoo/semantic-segmentation-pytorch)<br />

---
## Instance Segmentation
**Kaggle:** [rkuo2000/yolact](https://www.kaggle.com/rkuo2000/yolact)<br />
**Github:** [dbolya/yolact](https://github.com/dbolya/yolact)<br />

![](https://www.kaggleusercontent.com/kf/51709015/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..QjhSkbIf490iF5pCpaCTKw.6oLuok2y4E4tx2Q5WCBCKgoBRNNxsEm4BWOcAnrA8omXw32qRuZgvD3j0o03ILnM4G7JsT9I9qP-Ldpg2LpiRc98G3ENLAf2UV5e4OKZ98caYa8BBK_DhVvbRk0UDm8F9k_HMhDnEgmyQNZ7BsYkIfZiA78XpDtTbVBzcI-bbK7pTsiS7ez6qFbRYbxWzHT-LK69WEFvmWgJlr2B4twtgPjxwivUmBql5JbQICcTjs8hbPq3iOxRSvmXfwLVpLU30ldjAwBwb73XBmoEy_tF1lI9DgbrcCTkrxAjynTB4g3wrss70BQ7MLBiMdqOVC85pzn3kb4as3Wi_QKTYPVbMLTHUk3loL4I8qcyBPkL40PGX14LF3ykZjaXhSi5R7njg9QUSyMEe_UTGPuN0lsbAfM1yabIrDz5FSt22zj6FKsXhndhVWNs33emoaGgX-xt8gBxQDFfz0d0ZOXU3_GDWdDkZ86F1724o5-GXw3KhX3i5hU2hPwxo212dSGA3KZ8-W0w4IuE-mJ65RcPyWFkSGOVTjhJSNO9AIhbTC9VFyrXJNKHHXWtVRdtQh9ZHNWBCoRoWhKvm-ReTpiAbucE9J7uWORAfbNJCvBevyRlcfwgjrupIL3M0DkX4Q6A0QW0YMOOgMcOqwiSLd0i0iQpUA.3eCyP7OzEEmxtyF2CM_hKw/__results___files/__results___12_0.jpg)

---
## Video Object Segmentation
N/A

---
## Panoptic Segmentation

### Panoptic Driving Perception
**Kaggle**: [rkuo2000/yolop](https://www.kaggle.com/rkuo2000/yolop)<br />

![](https://mdimg.wxwenku.com/getimg/ccdf080c7af7e8a10e9b88444af98393d1f7b49c5e9d65ef2cd827532f32de1fa52314f1ea7a53ff4a598fa8606fdabf.jpg)

<iframe width="560" height="315" src="https://www.youtube.com/embed/4f9YHyqnq0A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br />

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

