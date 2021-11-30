---
layout: post
title: Face Recognition Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises include Facial Emotion Detection, Face Mask Detection, and Face Identification.

---
## Facial Emotion Detection (表情偵測)
Using MTCNN to detect face, then use CNN model to detect emotion
1. Train a CNN model for facial expression recognition (FER2013 dataset)
2. load CNN model to combine with MTCNN face detection

### check Tensorflow version
<u>On Kaggle:</u><br> 
`import tensorflow as tf`<br>
`print(tf.__version__)`<br>
2.6.0

<u>On PC:</u><br>
`$python -c 'import tensorflow as tf; print(tf.__version__)' `<br>
2.6.0

*Make sure to use same version of Tensorflow on Kaggle and PC for model file compatibilty.*

### FER2013-CNN
![](https://static-01.hindawi.com/articles/cin/volume-2018/7208794/figures/7208794.fig.003.svgz)
**Kaggle:** [rkuo2000/fer2013-cnn](https://kaggle.com/rkuo2000/fer2013-cnn)<br>
**Dataset:** [FER-2013](https://www.kaggle.com/msambare/fer2013)<br>
1. [Edit & Copy] FER2013-CNN 
2. [Run All] to Train model
3. Download model file `fer2013_cnn.h5`

### Emotion Detection
1. Clone sample codes<br>
`$git clone https://github.com/rkuo2000/tf`<br>
2. Copy **fer2013_cnn.h5** to ~/tf/models<br>
3. To detect using image file<br>
`$python mtcnn_emotion_detection.py images/facemask1.jpg`<br>
4. To detect using webcam<br>
`$python mtcnn_emotion_detection_cam.py`<br>

[mtcnn_emotion_detection_cam.py](https://github.com/rkuo2000/tf/blob/master/mtcnn_emotion_detection_cam.py)

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/facial_emotion_detection.png?raw=true)

---
## Face Mask Detection (口罩偵測)
Using MTCNN to detect face, then use CNN model to detect facemask
1. Train a CNN model for facemask detection
2. load CNN model to combine with MTCNN face detection

### check Tensorflow version
<u>On Kaggle:</u><br> 
`import tensorflow as tf`<br>
`print(tf.__version__)`<br>
2.6.0

<u>On PC:</u><br>
`$python -c 'import tensorflow as tf; print(tf.__version__)' `<br>
2.6.0

*Make sure to use same version of Tensorflow on Kaggle and PC for model file compatibilty.*

### FaceMask Detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/facemask_detection_12k_images_dataset.png?raw=true)
**Kaggle:** [rkuo2000/faskmask-cnn](https://kaggle.com/rkuo2000/facemask-cnn)<br>
**Dataset:** [Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset<br>
1. [Edit & Copy] FaceMask-CNN 
2. [Run All] to Train model
3. Download model file `facemask_cnn.h5`

### Face Mask Detection
1. Clone sample codes<br>
`$git clone https://github.com/rkuo2000/tf`<br>
2. Copy **facemask_cnn.h5** to ~/tf/models<br>
3. To detect using image file<br>
`$python mtcnn_facemask_detection.py images/facemask1.jpg`<br>
4. To detect using webcam<br>
`$python mtcnn_facemask_detection_cam.py`<br>

[mtcnn_facemask_detection_cam.py](https://github.com/rkuo2000/tf/blob/master/mtcnn_facemask_detection_cam.py)

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/face_mask_detection.png?raw=true)

---
## Face Identification (人臉辨識)
### FaceNet-PyTorch
**Kaggle:** [FaceNet-PyTorch](https://kaggle.com/rkuo2000/FaceNet-PyTorch)<br>
[Edit&Copy] https://kaggle.com/rkuo2000/facenet-pytorch<br>
[Run-All]
![](https://middleeast.in-24.com/entertainment/temp/resized/medium_2021-09-28-7fdde6cee2.jpg)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/FaceNet_PyTorch_embeddings.png?raw=true)
```
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in test_embeddings]
for dist in dists:
    if any(e<1 for e in dist):
        print(names[np.argmin(dist)])
    else:
        print('unknown')
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/FaceNet_PyTorch_result.png?raw=true)

---
### FaceNet
**Github:** [facenet-pytorch](https://github.com/rkuo2000/facenet-pytorch)<br>
**Kaggle:** [FaceNet-PyTorch](https://kaggle.com/rkuo2000/FaceNet-PyTorch)<br>

1. [Fork] https://github.com/rkuo2000/facenet-pytorch
2. [Edit&Copy] https://kaggle.com/rkuo2000/facenet-pytorch<br>
* face detection : models/mtcnn.py<br>
* facenet model : models/inception_resnet_v1.py<br>
* example: examples/infer.ipynb<br>
3. upload 1.jpg for each name to **https://github.com/your_name/facenet-pytoch**  data/test_images
4. upload yourself photo 1.jpg to **https://github.com/your_name/facenet-pytoch** data/test_images/your_name
5. Open **https://kaggle.com/your_name/facenet-pytorch**
6. modify git clone path<br>
`!git clone https://github.com/your_name/facenet-pytorch`<br>
7. modify test image path<br>
`img = plt.imread("facenet_pytorch/data/test_images/your_name/1.jpg")`<br>
8. Run-All

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

