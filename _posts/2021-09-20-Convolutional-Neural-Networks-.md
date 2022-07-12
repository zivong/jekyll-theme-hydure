---
layout: post
title: Convolutional Neural Networks
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Convolutional Neural Network (CNN) includes an overview, examples, and models.

---
* A more biologically accurate representation of a neuron
![](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/assets/tfdl_0403.png)

---
## Convolutional Neural Network (CNN)

### [Overview](https://www.analyticsvidhya.com/blog/2022/01/convolutional-neural-network-an-overview/)

**Blog:** [Basic Introduction to Convolutional Neural Network in Deep Learning](https://www.analyticsvidhya.com/blog/2022/03/basic-introduction-to-convolutional-neural-network-in-deep-learning/)<br>
* Image Classification
![](https://editor.analyticsvidhya.com/uploads/804084200125366Convolutional_Neural_Network_to_identify_the_image_of_a_bird.png)
* Typical CNN
![](https://editor.analyticsvidhya.com/uploads/59954intro%20to%20CNN.JPG)
![](https://editor.analyticsvidhya.com/uploads/94787Convolutional-Neural-Network.jpeg)
* Convolutional Layers
![](https://editor.analyticsvidhya.com/uploads/18707neural-networks-layers-visualization.jpg)

* Convolutional Operation
![](https://www.researchgate.net/profile/Hiromu-Yakura/publication/323792694/figure/fig1/AS:615019968475136@1523643595196/Outline-of-the-convolutional-layer.png)
* Max-Pooling
![](https://cdn-images-1.medium.com/max/1600/1*ODDBelSSa1drUjCHGgPt2w.png)
* [Activation Fuctions](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)<br>
Sigmoid vs ReLU (Rectified Linear Unit) 
![](https://miro.medium.com/max/1400/1*XxxiA0jJvPrHEJHD4z893g.png)
Tanh or hyperbolic tangent
![](https://miro.medium.com/max/1190/1*f9erByySVjTjohfFdNkJYQ.jpeg)
Leaky ReLU
![](https://miro.medium.com/max/1400/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg)
* [Softmax Activation function](https://towardsdatascience.com/softmax-activation-function-how-it-actually-works-d292d335bd78)
![](https://i.stack.imgur.com/0rewJ.png)
![](https://miro.medium.com/max/875/1*KvygqiInUpBzpknb-KVKJw.jpeg)
* Dropout<br>
[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)<br>
在模型訓練時隨機讓網絡某些隱含層節點的權重不工作，不工作的那些節點可以暫時認為不是網絡結構的一部分，但是它的權重得保留下来(只是暫時不更新而已)<br>
![](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/assets/tfdl_0408.png)
* Early Stopping
![](https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/assets/tfdl_0409.png)

---
## Examples
### MNIST dataset (手寫數字資料集）
![](https://miro.medium.com/max/495/1*G8jKIPXjoI_WivkDFUPlZQ.png)
60000筆28x28灰階數字圖片之訓練集<br>
10000筆28x28灰階數字圖片之測試集<br>

### [MNIST-CNN](https://www.kaggle.com/rkuo2000/mnist-cnn)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/CNN_MNIST.png?raw=true)
[mnist_cnn.py](https://github.com/rkuo2000/tf/blob/master/mnist_cnn.py)<br>
```
from tensorflow.keras import models, layers, datasets
import matplotlib.pyplot as plt

# Load Dataset
mnist = datasets.mnist # MNIST datasets
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()

# data normalization
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0 
 
print('x_train shape:', x_train.shape)
print('train samples:', x_train.shape[0])
print('test samples:', x_test.shape[0])

# reshape for input
x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Build Model
num_classes = 10 # 0~9

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(5, 5),activation='relu', padding='same',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary() 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train Model
epochs = 12 
batch_size = 128

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Save Model
models.save_model(model, 'models/mnist_cnn.h5')

# Evaluate Model
score = model.evaluate(x_train, y_train, verbose=0) 
print('\nTrain Accuracy:', score[1]) 
score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest  Accuracy:', score[1])
print()

# Show Training History
keys=history.history.keys()
print(keys)

def show_train_history(hisData,train,test): 
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
	
show_train_history(history, 'loss', 'val_loss')
show_train_history(history, 'accuracy', 'val_accuracy')
```
### [hiraganamnist](https://www.kaggle.com/rkuo2000/hiraganamnist)
**Dataset:** [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist)<br>
![](https://github.com/rois-codh/kmnist/raw/master/images/kmnist_examples.png)

---
### [Sign-Language MNIST](https://www.kaggle.com/code/rkuo2000/sign-language-mnist)
**Dataset:** [Sign-Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sign_Language_MNIST.png?raw=true)
21964筆28x28彩色手勢字母圖片之訓練集<br>
5491筆28x28彩色手勢字母圖片之測試集<br>

---
### [FashionMNIST-CNN](https://www.kaggle.com/rkuo2000/fashionmnist-cnn)
**Dataset:** [FashionMNIST](https://www.kaggle.com/zalando-research/fashionmnist)<br>
![](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)
28x28 grayscale images<br>
10 classes: [T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot]<br>
60000 train data<br>
10000 test data<br>

---
### [AirDigit CNN](https://www.kaggle.com/rkuo2000/airdigit-cnn)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/AirDigit_dataset.png?raw=true)

---
### [ECG Classification](https://www.kaggle.com/rkuo2000/ecg-classification)
[心電圖診斷理論基礎與系統](http://rportal.lib.ntnu.edu.tw:8080/server/api/core/bitstreams/9ae9fc6a-fa31-4bdf-b3ed-486881f61af8/content)<br>
ECG 心電圖分類：<br>
1. **Normal (正常心跳)**
2. **Artial Premature (早發性心房收縮)**
  - 早發性心房收縮就是心房在收到指令前提早跳動，由於打出的血量少而造成心跳空虛的症狀，再加上舒張期變長，回到心臟的血流增加使下一次的心跳力道較強，造成心悸的感覺，當心臟長期處於耗能異常的狀態，就容易併發心臟衰竭。
3. **Premature ventricular contraction (早發性心室收縮)**
  - 早發性心室收縮是心律不整的一種，是指病人的心室因為電位不穩定而提早跳動，病人會有心跳停格的感覺。因為病人心跳的質量差，心臟打出的血液量不足，卻仍會消耗能量，長期下來就可能衍生出心臟衰竭等嚴重疾病。早發性心室收縮的症狀包括心悸、頭暈、冷汗和胸悶等。
4. **Fusion of ventricular and normal (室性融合心跳)**
  - 室性融合波是由於兩個節律點發出的衝動同時激動心室的一部分形成的心室綜合波，是心律失常的干擾現象範疇
5. **Fusion of paced and normal (節律器融合心跳)**

**Paper:** [ECG Heartbeat Classification: A Deep Transferable Representation](https://arxiv.org/abs/1805.00794)<br>
![](https://d3i71xaburhd42.cloudfront.net/0997b7e7aa68708414fdb3257263f81f9d9c33ae/2-Figure1-1.png)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ECG_CNN.png?raw=true)

---
### [PPG2ABP](https://www.kaggle.com/code/rkuo2000/ppg2abp)
**Paper:** [PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks](https://arxiv.org/abs/2005.01669)<br>
![](https://d3i71xaburhd42.cloudfront.net/cf8a20a4ce19797c4ea03534505a369277f63da2/5-Figure1-1.png)
![](https://d3i71xaburhd42.cloudfront.net/cf8a20a4ce19797c4ea03534505a369277f63da2/10-Figure2-1.png)
**Code:** [nibtehaz/PPG2ABP](https://github.com/nibtehaz/PPG2ABP)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

