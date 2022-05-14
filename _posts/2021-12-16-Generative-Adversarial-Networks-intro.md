---
layout: post
title: Generative Adversarial Networks Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Style Transfer, VAE, GAN, Image Inpainting, DeepFaceDrawing, Toonify, PoseGAN, Pose2Pose, Virtual Try On, Face Swap, Talking Head, VQ-VAE, Music Seperation, Deep Singer, Voice Conversion.

---
## Style Transfer

### [DeepDream](https://deepdreamgenerator.com/)

![](https://b2h3x3f6.stackpathcdn.com/assets/landing/img/gallery/4.jpg)

---
### Nerual Style Transfer
**Paper:** [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)<br>
**Code:** [ProGamerGov/neural-style-pt](https://github.com/ProGamerGov/neural-style-pt)<br>
**Tutorial:** [Neural Transfer using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

![](https://pytorch.org/tutorials/_images/neuralstyle.png)
![](https://miro.medium.com/max/700/1*sBNwIsv5pPQQqHHEAkBHIw.png)

### Fast Style Transfer
**Paper:**[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) & [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)<br>
**Code:** [lengstrom/fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)<br>

<table>
<tr>
<td><img src="https://github.com/lengstrom/fast-style-transfer/raw/master/examples/style/udnie.jpg?raw=true"></td>
<td><img src="https://github.com/lengstrom/fast-style-transfer/blob/master/examples/content/stata.jpg?raw=true"></td>
<td><img src="https://github.com/lengstrom/fast-style-transfer/blob/master/examples/results/stata_udnie.jpg?raw=true"></td>
</tr>
</table>

## Variational AutoEncoder
### VAE
**Blog:** [VAE(Variational AutoEncoder) 實作](https://ithelp.ithome.com.tw/articles/10226549)<br>
**Paper:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)<br>
**Code:** [rkuo2000/fashionmnist-vae](https://www.kaggle.com/rkuo2000/fashionmnist-vae)<br>

![](https://github.com/timsainb/tensorflow2-generative-models/blob/master/imgs/vae.png?raw=1)
![](https://i.imgur.com/ZN6MyTx.png)
![](https://ithelp.ithome.com.tw/upload/images/20191009/20119971nNxkMbzOB8.png)

---
### Arbitrary Style Transfer
**Paper:** [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)<br>
**Code:** [elleryqueenhomels/arbitrary_style_transfer](https://github.com/elleryqueenhomels/arbitrary_style_transfer)<br>
![](https://user-images.githubusercontent.com/13844740/33978899-d428bf2e-e0dc-11e7-9114-41b6fb8921a7.jpg)

<table>
<tr>
<td><img width="50%" height="50%" src="https://github.com/elleryqueenhomels/arbitrary_style_transfer/raw/master/images/style_thumb/udnie_thumb.jpg?raw=true"></td>
<td><img width="50%" height="50%" src="https://github.com/elleryqueenhomels/arbitrary_style_transfer/raw/master/outputs/udnie-lance-2.0.jpg?raw=true"></td>
</tr>
<tr>
<td><img width="50%" height="50%" src="https://github.com/elleryqueenhomels/arbitrary_style_transfer/raw/master/images/style_thumb/escher_sphere_thumb.jpg?raw=true"></td>
<td><img width="50%" height="50%" src="https://github.com/elleryqueenhomels/arbitrary_style_transfer/raw/master/outputs/escher_sphere-lance-2.0.jpg?raw=true"></td>
</tr>
<tr>
<td><img width="50%" height="50%" src="https://github.com/elleryqueenhomels/arbitrary_style_transfer/raw/master/images/style_thumb/mosaic_thumb.jpg?raw=true"></td>
<td><img width="50%" height="50%" src="https://github.com/elleryqueenhomels/arbitrary_style_transfer/raw/master/outputs/mosaic-lance-2.0.jpg?raw=true"></td>
</tr>
</table>

### zi2zi
**Blog:** [zi2zi: Master Chinese Calligraphy with Conditional Adversarial Networks](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html)<br>
**Paper:** [Generating Handwritten Chinese Characters using CycleGAN](https://arxiv.org/abs/1801.08624)<br>
**Code:** [kaonashi-tyc/zi2zi](https://github.com/kaonashi-tyc/zi2zi)<br>

![](https://github.com/kaonashi-tyc/zi2zi/blob/master/assets/intro.gif?raw=true)
![](https://kaonashi-tyc.github.io/assets/network.png)

---
### DALL.E
**Blog:** [https://openai.com/blog/dall-e/](https://openai.com/blog/dall-e/)<br>
DALL·E is a 12-billion parameter version of GPT-3 trained to generate images from text descriptions, using a dataset of text–image pairs. <br>
**Paper:** [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)<br> 
**Code:** [openai/DALL-E](https://github.com/openai/DALL-E)<br>
![](https://pic1.zhimg.com/80/v2-4a1a29db75492620aaa06d98cdc66fb0_720w.jpg)

### [DALL.E-2](https://openai.com/dall-e-2/)
DALL·E 2 is a new AI system that can create realistic images and art from a description in natural language.<br>
**Paper:** [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)<br>
<iframe width="731" height="365" src="https://www.youtube.com/embed/qTgPSKKjfVg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://pic3.zhimg.com/80/v2-e096e3cf8a1e7a9f569b18f658da574e_720w.jpg)

---
## GAN
**Blog:** [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://wiki.pathmind.com/generative-adversarial-network-gan)<br>
**Paper:** [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)<br>
G是生成的神經網路，它接收一個隨機的噪訊z，通過這個噪訊生成圖片，為G(z)<br>
D是辨别的神經網路，辨别一張圖片夠不夠真實。它的輸入參數是x，x代表一張圖片，輸出D(x)代表x為真實圖片的機率<br>
![](https://developers.google.com/machine-learning/gan/images/gan_diagram.svg)

```
class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)
```

---
### DCGAN - Deep Convolutional Generative Adversarial Network
**Paper:** [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)<br>
**Code:** [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)<br>

![](https://github.com/carpedm20/DCGAN-tensorflow/raw/master/assets/result_16_01_04_.png)

**Generator**<br>
![](https://editor.analyticsvidhya.com/uploads/2665314.png)

**[DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)**<br>
```
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```

```
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```

---
### [DCGAN_LSGAN_WGAN_WGAN-GP_SNGAN_RSGAN_BEGAN_ACGAN_PGGAN_pix2pix_BigGAN](https://github.com/MingtaoGuo/DCGAN_WGAN_WGAN-GP_LSGAN_SNGAN_RSGAN_BEGAN_ACGAN_PGGAN_TensorFlow)<br>

![](https://raw.githubusercontent.com/MingtaoGuo/DCGAN_WGAN_WGAN-GP_LSGAN_SNGAN_RSGAN_BEGAN_ACGAN_PGGAN_TensorFlow/master/Image/GAN.jpg)

---
### MrCGAN
**Paper:** [Compatibility Family Learning for Item Recommendation and Generation](https://arxiv.org/abs/1712.01262)<br>
**Code:** [appier/compatibility-family-learning](https://github.com/appier/compatibility-family-learning)<br>

![](https://github.com/appier/compatibility-family-learning/blob/master/images/MrCGAN.jpg?raw=true)

---
### pix2pix
**Paper:** [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)<br>
**Code:** [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)<br>

![](https://camo.githubusercontent.com/c10e6bc28b817a8741c2611e685eec2f6e2634587227699290dece8dd7e13d0c/68747470733a2f2f7068696c6c6970692e6769746875622e696f2f706978327069782f696d616765732f7465617365725f76332e706e67)
![](https://www.researchgate.net/profile/Satoshi-Kida/publication/333259964/figure/fig1/AS:761197632712705@1558495066472/The-architecture-of-a-pix2pix-and-b-CycleGAN-a-pix2pix-requires-perfectly-aligned.png)

---
### CycleGAN
**Paper:** [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)<br>
**Code:** [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)<br>

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)

**Tutorial:** [CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan)<br>
![](https://i0.wp.com/neptune.ai/wp-content/uploads/input-and-predicted-image.png?resize=601%2C296&ssl=1)
```
# Generator G translates X -> Y
# Generator F translates Y -> X.
fake_y = generator_g(real_x, training=True)
cycled_x = generator_f(fake_y, training=True)

fake_x = generator_f(real_y, training=True)
cycled_y = generator_g(fake_x, training=True)

# same_x and same_y are used for identity loss.
same_x = generator_f(real_x, training=True)
same_y = generator_g(real_y, training=True)

disc_real_x = discriminator_x(real_x, training=True)
disc_real_y = discriminator_y(real_y, training=True)

disc_fake_x = discriminator_x(fake_x, training=True)
disc_fake_y = discriminator_y(fake_y, training=True)

# calculate the loss
gen_g_loss = generator_loss(disc_fake_y)
gen_f_loss = generator_loss(disc_fake_x)

total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y,cycled_y)

# Total generator loss = adversarial loss + cycle loss
total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
```

---
### pix2pixHD
**Paper:** [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585)<br>
**Code:** [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)<br>

![](https://github.com/NVIDIA/pix2pixHD/blob/master/imgs/teaser_720.gif?raw=true)

---
### vid2vid
**Paper:** [Video-to-Video Synthesis](https://arxiv.org/abs/1808.06601)<br>
**Code:** [NVIDIA/vid2vid](https://github.com/NVIDIA/vid2vid)<br>

![](https://github.com/NVIDIA/vid2vid/blob/master/imgs/teaser.gif?raw=true)
![](https://github.com/NVIDIA/vid2vid/blob/master/imgs/face.gif?raw=true)
![](https://github.com/NVIDIA/vid2vid/blob/master/imgs/pose.gif?raw=true)

---
### Recycle-GAN
**Paper:** [Recycle-GAN: Unsupervised Video Retargeting](https://arxiv.org/abs/1808.05174)<br>
**Code:** [aayushbansal/Recycle-GAN](https://github.com/aayushbansal/Recycle-GAN)<br>

![](https://img.technews.tw/wp-content/uploads/2018/09/20161417/teaser-624x313.png)
<iframe width="853" height="480" src="https://www.youtube.com/embed/IkmhU2UmgqM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### StarGAN
**Paper:** [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)<br>
**Code:** [yunjey/stargan](https://github.com/yunjey/stargan)<br>

![](https://github.com/yunjey/stargan/blob/master/jpg/main.jpg?raw=true)
![](https://miro.medium.com/max/1400/0*nkdR_l-XxMpx7oOJ.png)
![](https://github.com/hoangthang1607/StarGAN-Keras/blob/master/assests/overview.PNG?raw=true)
![](https://miro.medium.com/max/2400/1*N8DEcZsa8Qqp7nx927oajw.png)

---
### Glow
**Blog:** [https://openai.com/blog/glow/](https://openai.com/blog/glow/)<br>
**Paper:** [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)<br>
**Code:** [openai/glow](https://github.com/openai/glow)<br>

<video autoplay="" muted="" playsinline="" width="50%" height="50%" loop="">
  <source src="https://cdn.openai.com/research-covers/glow/videos/both_loop_new.mp4" type="video/mp4">
</video>
<video autoplay="" muted="" playsinline="" width="50%" height="50%" loop="">
  <source src="https://cdn.openai.com/research-covers/glow/videos/prafulla_people_loop.mp4" type="video/mp4">
Your browser does not support video
</video>
![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-28_at_8.43.24_PM_tNckkOB.png)

---
### StyleGAN
**Paper:** [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)<br>
**Code:** [NVlabs/stylegan](https://github.com/NVlabs/stylegan)<br>

<iframe width="574" height="323" src="https://www.youtube.com/embed/kSLJriaOumA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### StyleGAN 2
**Blog:** [Understanding the StyleGAN and StyleGAN2 Architecture](https://medium.com/analytics-vidhya/understanding-the-stylegan-and-stylegan2-architecture-add9e992747d)<br>
**Paper:** [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1912.04958)<br>
**Code:** [hNVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)<br>

![](https://miro.medium.com/max/692/0*nzmrf7VMLsTWt8SX)
![](https://miro.medium.com/max/2400/0*vHNCiQGmbXaTAfH5)
<iframe width="574" height="323" src="https://www.youtube.com/embed/c-NJtV9Jvp0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### StyleGAN2-ADA
**Paper:** [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)<br>
**Code:** [hNVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)<br>

![](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/docs/stylegan2-ada-teaser-1024x252.png?raw=true)

<iframe width="450" height="323" src="https://www.youtube.com/embed/DVXX0tmVyco" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### StyleGAN2 Distillation
**Paper:** [StyleGAN2 Distillation for Feed-forward Image Manipulation](https://arxiv.org/abs/2003.03581)<br>
**Code:** [EvgenyKashin/stylegan2-distillation](https://github.com/EvgenyKashin/stylegan2-distillation)<br>

![](https://github.com/EvgenyKashin/stylegan2-distillation/blob/master/imgs/title.jpg?raw=true)
![](https://github.com/EvgenyKashin/stylegan2-distillation/raw/master/imgs/aging.jpg?raw=true)
![](https://github.com/EvgenyKashin/stylegan2-distillation/raw/master/imgs/style_mixing.jpg?raw=true)
![](https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-58542-6_11/MediaObjects/504482_1_En_11_Fig2_HTML.png)

---
### Toonify
**Blog:** [StyleGAN network blending](https://www.justinpinkney.com/stylegan-network-blending/)<br>

![](https://www.justinpinkney.com/static/af0a8323fc372141dbc899811d1f7dd2/93992/98.jpg)

**Paper:** [Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains](https://arxiv.org/abs/2010.05334)<br>
**Code:** [justinpinkney/toonify](https://github.com/justinpinkney/toonify)<br>

![](https://github.com/justinpinkney/toonify/blob/master/abe_toon.jpg?raw=true)

---
### Face Sketch Synthesis
**Paper:** [Face Sketch Synthesis via Semantic-Driven Generative Adversarial Network](https://arxiv.org/abs/2106.15121)<br>

![](https://media.arxiv-vanity.com/render-output/5336916/parsing_U2.png)
![](https://media.arxiv-vanity.com/render-output/5336916/Architecture.png)

---
### pix2style2pix
**Code:** [eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)<br>

![](https://github.com/eladrich/pixel2style2pixel/raw/master/docs/seg2image.png)

---
## Face Datasets
### [Celeb-A HQ Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
[tensorflow datasets](https://www.tensorflow.org/datasets/catalog/celeb_a_hq)<br>
![](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png)

### [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)

![](https://github.com/NVlabs/ffhq-dataset/blob/master/ffhq-teaser.png?raw=true)

---
### [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset)

![](https://github.com/NVlabs/metfaces-dataset/blob/master/img/metfaces-teaser.png?raw=true)

---
### [Animal Faces-HQ dataset (AFHQ)](https://www.kaggle.com/andrewmvd/animal-faces)
Animal Faces-HQ (AFHQ), consisting of 15,000 high-quality images at 512×512 resolution<br>
The dataset includes three domains of cat, dog, and wildlife, each providing about 5000 images.<br>
![](https://production-media.paperswithcode.com/datasets/Screenshot_2021-01-26_at_18.31.42.png)

---
### [Ukiyo-e Faces](https://www.justinpinkney.com/ukiyoe-dataset/)

![](https://www.justinpinkney.com/static/63be553a63ea6ea0f81719542f35410a/4b190/ukiyoe-dataset.jpg)

---
### [Cartoon Faces](https://www.kaggle.com/rkuo2000/cartoonfaces)

![](https://github.com/justinpinkney/toonify/blob/master/montage-small.jpg?raw=true)

---
### GANimation
**Blog:** [GANimation: Anatomically-aware Facial Animation from a Single Image](https://www.albertpumarola.com/research/GANimation/)<br>
**Paper:** [GANimation: Anatomically-aware Facial Animation from a Single Image](https://arxiv.org/abs/1807.09251)<br>
**Code:** [albertpumarola/GANimation](https://github.com/albertpumarola/GANimation)<br>

![](https://camo.githubusercontent.com/6fdd6c7b53aee10ca455af0c82fcf556c9c2e846daf2351f265172d5ec503bc5/687474703a2f2f7777772e616c6265727470756d61726f6c612e636f6d2f696d616765732f323031382f47414e696d6174696f6e2f7465617365722e706e67)
![](https://www.albertpumarola.com/images/2018/GANimation/model.png)
<br>
![](https://www.albertpumarola.com/images/2018/GANimation/face_gen_eq.png)
<br>
![](https://www.albertpumarola.com/images/2018/GANimation/results.png)

---
### Sefa 
**Blog:** [https://genforce.github.io/sefa/](https://genforce.github.io/sefa/)<br>
**Paper:** [Closed-Form Factorization of Latent Semantics in GANs](https://arxiv.org/abs/2007.06600)<br>
**Code:** [genforce/sefa](https://github.com/genforce/sefa)<br>

<table>
<tr>
<td>Pose</td><td>Mouth</td><td>Eye</td>
</tr>
<tr>
<td><img src="https://genforce.github.io/sefa/assets/stylegan_animeface_pose.gif"></td>
<td><img src="https://genforce.github.io/sefa/assets/stylegan_animeface_mouth.gif"></td>
<td><img src="https://genforce.github.io/sefa/assets/stylegan_animeface_eye.gif"></td>
</tr>
</table>
<iframe width="850" height="472" src="https://www.youtube.com/embed/OFHW2WbXXIQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### AniGAN
**Blog:** [博士後小姐姐把「二次元老婆生成器」升級了：這一次可以指定畫風](https://bangqu.com/5uWn8B.html)<br>
**Paper:** [AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation](https://arxiv.org/abs/2102.12593)<br>
![](https://d2ndd3gtcc6iwc.cloudfront.net/liang/news/20210302/v2-0d272ec5646191d4b0a11ffefb41c408_b.jpg)
![](https://d2ndd3gtcc6iwc.cloudfront.net/liang/news/20210302/v2-15685fdb3dbe7c63e12b5da8811c5d51_b.jpg)
![](https://d2ndd3gtcc6iwc.cloudfront.net/liang/news/20210302/v2-f716db3e91ba8587af8897d55e9e59d1_b.jpg)

---
### CartoonGAN
**Blog:** [Generate Anime using CartoonGAN and Tensorflow 2.0](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html)<br>
**Paper:** [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)<br>
**Code:** [mnicnc404/CartoonGan-tensorflow](https://github.com/mnicnc404/CartoonGan-tensorflow)<br>

![](https://github.com/mnicnc404/CartoonGan-tensorflow/blob/master/images/cover.gif?raw=true)
![](https://gmarti.gitlab.io/assets/cartoongan/architecture_cartoogan.png)

---
### Cartoon-GAN
**Paper:** [Generative Adversarial Networks for photo to Hayao Miyazaki style cartoons](https://arxiv.org/abs/2005.07702)<br>
**Code:** [FilipAndersson245/cartoon-gan](https://github.com/FilipAndersson245/cartoon-gan)<br>

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Cartoon-GAN_comparison.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Cartoon-GAN_arch.png?raw=true)

---
### White-box Cartoonization
**Paper:** [White-Box Cartoonization Using An Extended GAN Framework](https://arxiv.org/abs/2107.04551)<br>
**Code:** [SystemErrorWang/White-box-Cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization)<br>

![](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/shinjuku.jpg?raw=true)
![](https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/images/method.jpg?raw=true)

**Code:** [White-box facial image cartoonizaiton](https://github.com/SystemErrorWang/FacialCartoonization)<br>
![](https://github.com/SystemErrorWang/FacialCartoonization/blob/master/results/00050.jpg?raw=true)

---
### GAN Compression
**Paper:** [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936)<br>
**Code:** [mit-han-lab/gan-compression](https://github.com/mit-han-lab/gan-compression)<br>

![](https://github.com/mit-han-lab/gan-compression/raw/master/imgs/teaser.png)
![](https://github.com/mit-han-lab/gan-compression/blob/master/imgs/overview.png?raw=true)

---
### SRGAN
**Paper:** [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)<br>
**Code:** [tensorlayer/srgan](https://github.com/tensorlayer/srgan)<br>
![](https://github.com/tensorlayer/srgan/raw/master/img/SRGAN_Result2.png?raw=true)
![](https://github.com/tensorlayer/srgan/raw/master/img/model.jpeg?raw=true)

---
### SingleHDR
**Paper:** [Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline](https://arxiv.org/abs/2004.01179)<br>
**Code:** [alex04072000/SingleHDR](https://github.com/alex04072000/SingleHDR)<br>

![](https://github.com/alex04072000/SingleHDR/blob/master/teaser.png?raw=true)
![](https://github.com/alex04072000/SingleHDR/blob/master/Proposed.png?raw=true)

---
## Image Inpainting

### High-Resolution Image Inpainting
**Blog:** [Review: High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis](https://medium.com/analytics-vidhya/review-high-resolution-image-inpainting-using-multi-scale-neural-patch-synthesis-4bbda21aa5bc)<br>
**Paper:** [High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis](https://arxiv.org/abs/1611.09969)<br>
**Code:** [leehomyc/Faster-High-Res-Neural-Inpainting](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting)<br>
![](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting/blob/master/images/teaser.png?raw=true)
![](https://miro.medium.com/max/2400/1*FdZUwlYIAI_sEziOZS-Qjg.png)

---
### Image Inpainting for Irregular Holes
**Blog:** [Image Inpainting for Irregular Holes Using Partial Convolutions](https://medium.com/@neurohive/image-inpainting-for-irregular-holes-using-partial-convolutions-e46ab64f9570)<br>
**Paper:** [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)<br>
**Code:** [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv)<br>

![](https://miro.medium.com/max/2000/1*HUmj7An3CvGrJiTZAgiHBw.png)
![](https://miro.medium.com/max/2400/0*NTUlZCsCYHcRI7y4.)

---
### DeepFill V2
**Blog:** [A Practical Generative Deep Image Inpainting Approach](https://towardsdatascience.com/a-practical-generative-deep-image-inpainting-approach-1c99fef68bd7)<br>
**Paper:** [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)<br>
**Code:** [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting)<br>

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_raw.png"></td>
<td><img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_input.png"></td>
<td><img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_output.png"></td>
</tr>
</table>
![](https://miro.medium.com/max/1400/1*Q38k2RnxBkgWSJxzblzbJA.png)
![](https://miro.medium.com/max/1225/1*UaweIaCSL8HmFG9jh-KGlA.png)

---
### EdgeConnect
**Paper:** [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://arxiv.org/abs/1901.00212)<br>
**Code:** [knazeri/edge-connect](https://github.com/knazeri/edge-connect)<br>

![](https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png)
![](https://miro.medium.com/max/2640/1*KKiBBWo20W2BjrzEViWVVA.png)

---
### Deep Flow-Guided Video Inpainting
**Paper:** [Deep Flow-Guided Video Inpainting](https://arxiv.org/abs/1905.02884)<br>
**Code:** [nbei/Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)<br>

![](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/blob/master/gif/captain.gif?raw=true)
![](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/raw/master/gif/flamingo.gif?raw=true)
![](https://nbei.github.io/video-inpainting/framework.png)

---
### Flow-edge Guided Video Completion
**Paper:** [Flow-edge Guided Video Completion](https://arxiv.org/abs/2009.01835)<br>
**Code:** [vt-vl-lab/FGVC](https://github.com/vt-vl-lab/FGVC)<br>

<iframe width="585" height="329" src="https://www.youtube.com/embed/CHHVPxHT7rc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-58610-2_42/MediaObjects/504453_1_En_42_Fig2_HTML.png)

---
### GauGAN
**Paper:** [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)<br>
**Code:** [NVlabs/SPADE](https://github.com/NVlabs/SPADE)<br>

<table>
<tr>
<td><img src="https://nvlabs.github.io/SPADE/images/treepond.gif"></td>
<td><img src="https://nvlabs.github.io/SPADE/images/ocean.gif"></td>
</tr>
</table>

### DeepFaceDrawing
**Paper:**  [Deep Generation of Face Images from Sketches](https://arxiv.org/abs/2006.01047)<br>
**Code:** [franknb/Drawing-to-Face](https://github.com/franknb/Drawing-to-Face)<br>

![](https://github.com/franknb/Drawing-to-Face/raw/main/showcase/architecture.png)

---
### Pose-guided Person Image Generation
**Paper:** [Pose Guided Person Image Generation](https://arxiv.org/abs/1705.09368)<br>
**Code:** [charliememory/Pose-Guided-Person-Image-Generation](https://github.com/charliememory/Pose-Guided-Person-Image-Generation)<br>

![](https://raw.githubusercontent.com/charliememory/Pose-Guided-Person-Image-Generation/d6122a319a6f4af845883933cc12ffc1da09cb19/imgs/Paper-framework.svg)

---
### PoseGAN
**Paper:** [Deformable GANs for Pose-based Human Image Generation](https://arxiv.org/abs/1801.00055)<br>
**Code:** [AliaksandrSiarohin/pose-gan](https://github.com/AliaksandrSiarohin/pose-gan)<br>

![](https://github.com/AliaksandrSiarohin/pose-gan/blob/master/sup-mat/teaser.jpg?raw=true)
![](https://d3i71xaburhd42.cloudfront.net/74b9632e8c7bc7c96af5561a017b40b9613f196d/5-Figure2-1.png)

---
### Everybody Dance Now
**Blog:** [https://carolineec.github.io/everybody_dance_now/](https://carolineec.github.io/everybody_dance_now/)<br>
**Paper:** [Everybody Dance Now](https://arxiv.org/abs/1808.07371)<br>
**Code:** [carolineec/EverybodyDanceNow](https://github.com/carolineec/EverybodyDanceNow)

![](https://camo.githubusercontent.com/eda0581853e31c65a83d6da433efa50f47830f3f867269f8309322f613a74e9e/68747470733a2f2f6c61756768696e6773717569642e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031382f30382f4576657279626f64792d44616e63652d4e6f772e676966)
<iframe width="640" height="360" src="https://www.youtube.com/embed/PCBTZh41Ris" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Virtual Try On (VTON)
### VITON
**Paper:** [VITON: An Image-based Virtual Try-on Network](https://arxiv.org/abs/1711.08447)<br>

![](https://www.researchgate.net/profile/Hyunwoo-Hwangbo-2/publication/344765109/figure/fig1/AS:948614977318913@1603178843747/Image-based-virtual-try-on-network-42.png)

---
### MG-VTON
**Paper:** [Towards Multi-pose Guided Virtual Try-on Network](https://arxiv.org/abs/1902.11026)

![](https://www.researchgate.net/publication/353677925/figure/fig4/AS:1052925799460871@1628048481996/Visualized-comparison-between-MG-VTON-7-and-several-variants-of-our-method-on-the-MPV.jpg)
![](https://user-images.githubusercontent.com/25688193/63428303-3827b700-c452-11e9-8151-b9a6129d8364.png)

---
### Poly-GAN
**Paper:** [Poly-GAN: Multi-Conditioned GAN for Fashion Synthesis](https://arxiv.org/abs/1909.02165)<br>

![](https://www.researchgate.net/publication/335651443/figure/fig5/AS:799974707183625@1567740241650/Poly-GAN-results-shown-from-left-to-right-column-Model-image-Pose-Skeleton-Reference.ppm)
![](https://www.researchgate.net/publication/335651443/figure/fig2/AS:799974707191808@1567740241131/Poly-GAN-pipeline-Stage-1-Garment-transformation-with-Poly-GAN-conditioned-on-the-RGB.ppm)

---
### ACGPN
**Paper:** [Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content](https://arxiv.org/abs/2003.05863)<br>
**Code:** [switchablenorms/DeepFashion_Try_On](https://github.com/switchablenorms/DeepFashion_Try_On)<br>

<iframe width="1148" height="646" src="https://www.youtube.com/embed/BbKBSfDBcxI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-09-01_at_10.13.40_AM_V9osXdz.png)

---
### SMIS
**Blog:** [https://seanseattle.github.io/SMIS/](https://seanseattle.github.io/SMIS/)<br>
**Paper:** [Semantically Multi-modal Image Synthesis](https://arxiv.org/abs/2003.12697)<br>
**Code:** [Seanseattle/SMIS](https://github.com/Seanseattle/SMIS)<br>

![](https://seanseattle.github.io/SMIS/imgs/main.jpg)
<iframe width="784" height="500" src="https://www.youtube.com/embed/uarUonGi_ZU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://d3i71xaburhd42.cloudfront.net/5f678f828a3797540d2fba1517aa38570f2de0de/4-Figure2-1.png)

---
### CP-VTON+
**Paper:** [CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On](https://minar09.github.io/cpvtonplus/cvprw20_cpvtonplus.pdf)<br>
**Code:** [minar09/cp-vton-plus](https://github.com/minar09/cp-vton-plus)<br>

![](https://github.com/minar09/cp-vton-plus/raw/master/teaser.png)
![](https://d3i71xaburhd42.cloudfront.net/36d2174ceaf40317c79e33e1028c0ab94da2be61/3-Figure2-1.png)

---
### O-VITON
**Paper:** [Image Based Virtual Try-on Network from Unpaired Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Neuberger_Image_Based_Virtual_Try-On_Network_From_Unpaired_Data_CVPR_2020_paper.pdf)<br>
**Code:** [trinanjan12/Image-Based-Virtual-Try-on-Network-from-Unpaired-Data](https://github.com/trinanjan12/Image-Based-Virtual-Try-on-Network-from-Unpaired-Data)<br>

Inference Flow<br>
![](https://camo.githubusercontent.com/34e89ac87f584a41b1edbd78151794f4f6b213b668e7a40f75332204f6327837/68747470733a2f2f692e696d6775722e636f6d2f6550374c73576e2e706e67)

Training Flow<br>
![](https://camo.githubusercontent.com/6cf9f66c1e980925c440f60e11f80e5f12c21898d5fca3541e6f37f9b59b368a/68747470733a2f2f692e696d6775722e636f6d2f5a72664d477a652e706e67)

---
### PF-AFN
**Paper:** [Parser-Free Virtual Try-on via Distilling Appearance Flows](https://arxiv.org/abs/2103.04559)<br>
**Code:** [geyuying/PF-AFN](https://github.com/geyuying/PF-AFN)<br>

![](https://github.com/geyuying/PF-AFN/raw/main/show/compare_both.jpg?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PF-AFN.png?raw=true)

---
### CIT
**Paper:** [Cloth Interactive Transformer for Virtual Try-On](https://arxiv.org/abs/2104.05519)<br>
**Code:** [Amazingren/CIT](https://github.com/Amazingren/CIT)<br>

![](https://www.researchgate.net/publication/350834504/figure/fig2/AS:1011971105239040@1618284121641/Qualitative-comparisons-of-the-warped-cloths-by-the-proposed-CIT-based-geometric-matching.jpg)
![](https://www.researchgate.net/publication/350834504/figure/fig3/AS:1011971105230854@1618284121849/Qualitative-comparisons-of-different-state-of-the-art-methods.jpg)
![](https://www.researchgate.net/publication/350834504/figure/fig4/AS:1011971109437440@1618284122022/Qualitative-comparisons-of-ablation-studies-between-B3-and-B4.jpg)
![](https://www.researchgate.net/publication/350834504/figure/fig5/AS:1011971109425152@1618284122137/Several-failure-cases-of-our-proposed-CIT-for-virtual-try-on.jpg)
![](https://www.researchgate.net/publication/350834504/figure/fig1/AS:1011971105230852@1618284121602/The-overall-architecture-of-the-proposed-Cloth-Interactive-Transformer-CIT-for-virtual.png)

---
### pix2surf
**Paper:** [Learning to Transfer Texture from Clothing Images to 3D Humans](https://arxiv.org/abs/2003.02050)<br>
**Code:** [polobymulberry/pix2surf](https://github.com/polobymulberry/pix2surf)<br>

![](https://github.com/polobymulberry/pix2surf/blob/master/teaser_gif.gif?raw=true)

---
### TailorNet 
**Paper:** [TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style](https://arxiv.org/abs/2003.04583)<br>
**Code:** [chaitanya100100/TailorNet](https://github.com/chaitanya100100/TailorNet)<br>

![](https://virtualhumans.mpi-inf.mpg.de/tailornet/imgs/multi.gif)
<iframe width="784" height="441" src="https://www.youtube.com/embed/F0O21a_fsBQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## 3D Avatar

### PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization
**Blog:** [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://shunsukesaito.github.io/PIFuHD/)<br>
**Paper:** [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://arxiv.org/abs/2004.00452)<br>
**Code:** [facebookresearch/pifuhd](https://github.com/facebookresearch/pifuhd)<br>

![](https://shunsukesaito.github.io/PIFuHD/resources/images/teaser.png)
<iframe width="574" height="323" src="https://www.youtube.com/embed/-1XYTmm8HhE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://shunsukesaito.github.io/PIFuHD/resources/images/overview.png)

---
### [gDNA: Towards Generative Detailed Neural Avatars](https://xuchen-ethz.github.io/gdna/)
**Blog:** [PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization](https://shunsukesaito.github.io/PIFuHD/)<br>
**Paper:** [gDNA: Towards Generative Detailed Neural Avatars](https://arxiv.org/abs/2201.04123)<br>

<iframe width="763" height="429" src="https://www.youtube.com/embed/uOyoH7OO16I" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://ait.ethz.ch/projects/2022/gdna/downloads/assets/pipeline.png)
To generate diverse 3D humans, we build an implicit multi-subject articulated model. We model clothed human shapes and detailed surface normals in a pose-independent canonical space via a neural implicit surface representation, conditioned on latent codes.

---
## NeRF

### [NeRF:Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf)
**Paper:** [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
**Code:** [bmild/nerf](https://github.com/bmild/nerf)
**Colab:** [tiny_nerf](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)
**Kaggle:** [rkuo2000/tiny-nerf](https://www.kaggle.com/code/rkuo2000/tiny-nerf)<br>
![](https://github.com/bmild/nerf/raw/master/imgs/pipeline.jpg)
<iframe width="940" height="528" src="https://www.youtube.com/embed/JuH79E8rdKc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://uploads-ssl.webflow.com/51e0d73d83d06baa7a00000f/5e700a025ff238947d682a1f_pipeline_website-03.svg)
The algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location (x, y, z) and viewing direction (θ, φ)) and whose output is the volume density and view-dependent emitted radiance at that spatial location.
![](https://uploads-ssl.webflow.com/51e0d73d83d06baa7a00000f/5e700ef6067b43821ed52768_pipeline_website-01-p-800.png)
We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis.
<video autoplay><source src=="http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/website_renders/synth_grid_3.mp4"></video>

---
### [FastNeRF](https://microsoft.github.io/FastNeRF/)
**Paper**: [FastNeRF: High-Fidelity Neural Rendering at 200FPS](FastNeRF: High-Fidelity Neural Rendering at 200FPS)<br>
<iframe width="731" height="365" src="https://www.youtube.com/embed/mi5b142WEmw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [KiloNeRF](https://creiser.github.io/kilonerf/)
**Paper:** [KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs](https://arxiv.org/abs/2103.13744)<br>
**Code:** [creiser/kilonerf](https://github.com/creiser/kilonerf)<br>
![](https://github.com/creiser/kilonerf/blob/master/interactive-viewer.gif?raw=true)
![](https://pbs.twimg.com/media/ExXgi0UWYAACEs0?format=jpg&name=small)

---
### [PlenOctrees](https://alexyu.net/plenoctrees/)
**Paper:** [PlenOctrees for Real-time Rendering of Neural Radiance Fields](https://arxiv.org/abs/2103.14024)<br>
**Code:** [NeRF-SH training and conversion](https://github.com/sxyu/plenoctree) & [Volume Rendering](https://github.com/sxyu/volrend)<br>
<iframe width="696" height="408" src="https://www.youtube.com/embed/obrmH1T5mfI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://alexyu.net/plenoctrees/img/pipeline.png)

--- 
### [Sparse Neural Radiance Grids (SNeRG)](https://phog.github.io/snerg/)
**Paper:** [Baking Neural Radiance Fields for Real-Time View Synthesis](https://arxiv.org/abs/2103.14645)<br>
**Code:** [google-research/snerg](https://github.com/google-research/google-research/tree/master/snerg)<br>
<iframe width="746" height="418" src="https://www.youtube.com/embed/5jKry8n5YO8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
![](https://phog.github.io/snerg/img/breakdown.png)

---
### [Faster Neural Radiance Fields Inference](https://chrischoy.github.io/research/nerfs/)
* **NeRF Inference: Probabilistic Approach**<br>
![](https://miro.medium.com/max/770/1*uE6cqVbIdTGZMMYfMR1Rvw.png)
* **Faster Inference: Efficient Ray-Tracing + Image Decomposition**<br>
<table>
<tr><td>Method Render</td><td>time</td><td>Speedup</td></tr>
<tr><td>NeRF</td><td>56185 ms</td><td> – </td></tr>
<tr><td>NeRF + ESS + ERT</td><td>788 ms</td><td>71</td></tr>
<tr><td>KiloNeRF</td><td>22 ms</td><td>2548</td></tr>
</table>
<br>
* **Sparse Voxel Grid and Octrees: Spatial Sparsity**<br>
![](https://d3i71xaburhd42.cloudfront.net/17d7767a6ea87f4ab24d9cfaa5039160af9cad76/6-Figure3-1.png)
* **Neural Sparse Voxel Fields** proposed learn a sparse voxel grid in a progressive manner that increases the resolution of the voxel grid at a time to not just such represent explicit geomety but also to learn the implicit features per non-empty voxel.<br>
![](https://alexyu.net/plenoctrees/img/pipeline.png)
* **PlenOctree** also uses the octree structure to speed up the geometry queries and store the view-dependent color representation on the leaves.
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/8bd70d3dcfa295d9710922c34c1a9eeb0be48b94/3-Figure2-1.png)
* **KiloNeRF** proposes to decompose this large deep NeRF into a set of small shallow NeRFs that capture only a small portion of the space.
![](https://pbs.twimg.com/media/ExXgi0UWYAACEs0?format=jpg&name=small)
* **Baking Neural Radiance Fields (SNeRG)** proposes to decompose an image into the diffuse color and specularity so that the inference network handles a very simple task.
![](https://phog.github.io/snerg/img/breakdown.png)

---
### [Point-NeRF](https://xharlie.github.io/projects/project_sites/pointnerf/)
**Paper:** [Point-NeRF: Point-based Neural Radiance Fields](https://arxiv.org/abs/2201.08845)<br>
**Code:** [Xharlie/pointnerf](https://github.com/Xharlie/pointnerf)<br>
![](https://github.com/Xharlie/pointnerf/blob/master/images/pipeline.png?raw=true)
Point-NeRF uses neural 3D point clouds, with associated neural features, to model a radiance field.

---
### SqueezeNeRF
**Paper:** [SqueezeNeRF: Further factorized FastNeRF for memory-efficient inference](https://arxiv.org/abs/2204.02585)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/SqueezeNeRF_architecture.png?raw=true)

---
### [Nerfies: Deformable Neural Radiance Fields](https://nerfies.github.io/)
**Paper:** [Nerfies: Deformable Neural Radiance Fields](https://arxiv.org/abs/2011.12948)<br>
**Code:** [google/nerfies](https://github.com/google/nerfies)<br>
<video width="720" height="320"  autoplay><source src="https://homes.cs.washington.edu/~kpar/nerfies/videos/teaser.mp4" type="video/mp4"></video>
<iframe width="763" height="429" src="https://www.youtube.com/embed/MrKrnHhk8IA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [Light Field Networks](https://www.vincentsitzmann.com/lfns/)
**Paper:** [Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering](https://arxiv.org/abs/2106.02634)<br>
**Code:** [Light Field Networks](https://github.com/vsitzmann/light-field-networks)<br>
<iframe width="970" height="546" src="https://www.youtube.com/embed/x3sSreTNFw4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## Face Swap
### faceswap-GAN
**Code:** [https://github.com/shaoanlu/faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN)<br>

![](https://camo.githubusercontent.com/42a9fd0780d51da64af00f571823f8b2111c9bce62bca292d02ddba64e959a07/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f32346b31367674716b686c663133692f6175746f5f726573756c74732e6a70673f7261773d31)

---
### DeepFake
**Paper:** [DeepFaceLab: Integrated, flexible and extensible face-swapping framework](https://arxiv.org/abs/2005.05535)<br>
**Github:** [iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)<br>
**[DeepFake Detection Challenge](https://ai.facebook.com/blog/deepfake-detection-challenge)**<br>

<iframe width="574" height="323" src="https://www.youtube.com/embed/R9f7WD0gKPo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### ObamaNet
**Paper:** [ObamaNet: Photo-realistic lip-sync from text](https://arxiv.org/abs/1801.01442)<br>
**Code:** [acvictor/Obama-Lip-Sync](https://github.com/acvictor/Obama-Lip-Sync)<br>

<img width="50%" height="50%" src="https://github.com/acvictor/Obama-Lip-Sync/blob/master/results/5.png?raw=true">

---
### Talking Face
**Paper:** [Talking Face Generation by Adversarially Disentangled Audio-Visual Representation](https://arxiv.org/abs/1807.07860)<br>
**Code:** [Hangz-nju-cuhk/Talking-Face-Generation-DAVS](https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS)<br>

![](https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS/blob/master/misc/teaser.png?raw=true)

---
### Neural Talking Head
**Blog:** [Creating Personalized Photo-Realistic Talking Head Models](https://medium.com/ai%C2%B3-theory-practice-business/creating-personalized-photo-realistic-talking-head-models-34302d247f9b)<br>
**Paper:** [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](https://arxiv.org/abs/1905.08233)<br>
**Code:** [vincent-thevenin/Realistic-Neural-Talking-Head-Models](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models)<br>

![](https://miro.medium.com/max/1354/1*TqYFSRQx98biy5KlvlGDRA.png)
![](https://miro.medium.com/max/2400/1*FLMSUV23pGOuPgspPDPysA.png)

---
### First Order Model
**Blog:** [First Order Motion Model for Image Animation](https://aliaksandrsiarohin.github.io/first-order-model-website/)<br>
**Paper:** [First Order Motion Model for Image Animation](https://arxiv.org/abs/2003.00196)<br>
**Code:** [AliaksandrSiarohin/first-order-model](https://github.com/AliaksandrSiarohin/first-order-model)<br>

![](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/sup-mat/vox-teaser.gif?raw=true)
![](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/sup-mat/fashion-teaser.gif?raw=true)
![](https://aliaksandrsiarohin.github.io/first-order-model-website/pipeline.png)
<iframe width="606" height="341" src="https://www.youtube.com/embed/u-0cQ-grXBQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### VQ-AVE
**Blog:** [帶你認識Vector-Quantized Variational AutoEncoder - 理論篇](https://medium.com/ai-academy-taiwan/%E5%B8%B6%E4%BD%A0%E8%AA%8D%E8%AD%98vector-quantized-variational-autoencoder-%E7%90%86%E8%AB%96%E7%AF%87-49a1829497bb)<br>

**Paper:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)<br>

![](https://miro.medium.com/max/700/1*9GZoBSZPw4VelO2vV9KfDw.png)

**Paper:** [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)<br>

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-28_at_4.56.19_PM.png)


---
## Music Seperation
### Spleeter
**Paper:** [Spleeter: A FAST AND STATE-OF-THE ART MUSIC SOURCE
SEPARATION TOOL WITH PRE-TRAINED MODELS](https://archives.ismir.net/ismir2019/latebreaking/000036.pdf)<br>
**Code:** [deezer/spleeter](https://github.com/deezer/spleeter)<br>

---
### Wave-U-Net
**Paper:** [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185)<br>
**Code:** [f90/Wave-U-Net](https://github.com/f90/Wave-U-Net)<br>

![](https://github.com/f90/Wave-U-Net/blob/master/waveunet.png?raw=true)

---
### Hyper Wave-U-Net
**Paper:** [Improving singing voice separation with the Wave-U-Net using Minimum Hyperspherical Energy](https://arxiv.org/abs/1910.10071)<br>
**Code:** [jperezlapillo/hyper-wave-u-net](https://github.com/jperezlapillo/hyper-wave-u-net)<br>
**MHE regularisation:**<br>
![](https://github.com/jperezlapillo/Hyper-Wave-U-Net/blob/master/diagram_v2.JPG?raw=true)

---
### Demucs
**Paper:** [Music Source Separation in the Waveform Domain](https://arxiv.org/abs/1911.13254)<br>
**Code:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs)<br>

![](https://github.com/facebookresearch/demucs/blob/main/demucs.png?raw=true)

---
## Deep Singer

### [OpenAI Jukebox](https://jukebox.openai.com/)
**Blog:** [Jukebox](https://openai.com/blog/jukebox/)<br>
model modified from **VQ-VAE-2**
**Paper:** [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341)<br>
**Colab:** [Interacting with Jukebox](https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb)<br>

---
### DeepSinger
**Blog:** [Microsoft’s AI generates voices that sing in Chinese and English](https://venturebeat.com/2020/07/13/microsofts-ai-generates-voices-that-sing-in-chinese-and-english/)<br>
**Paper:** [DeepSinger: Singing Voice Synthesis with Data Mined From the Web](https://arxiv.org/abs/2007.04590)<br>
**Demo:** [DeepSinger: Singing Voice Synthesis with Data Mined From the Web](https://speechresearch.github.io/deepsinger/)<br>

![](https://lfs.aminer.cn/upload/pdf_image/5ecf/e0e/5ecfae0e9e795eb20a615049img-002.png)
<p align="center">The alignment model based on the architecture of automatic speech recognition</p>

![](https://lfs.aminer.cn/upload/pdf_image/5ecf/e0e/5ecfae0e9e795eb20a615049img-004.png)
<p align="center">The architecture of the singing model</p>

![](https://lfs.aminer.cn/upload/pdf_image/5ecf/e0e/5ecfae0e9e795eb20a615049img-005.png)
<p align="center">The inference process of singing voice synthesis</p>

---
## Voice Conversion
**Paper:** [An Overview of Voice Conversion and its Challenges: From Statistical Modeling to Deep Learning](https://arxiv.org/abs/2008.03648)<br>

![](https://d3i71xaburhd42.cloudfront.net/4e1f36855442b761729dad4507513e23ca66206c/7-Figure2-1.png)

**Blog:** [Voice Cloning Using Deep Learning](https://medium.com/the-research-nest/voice-cloning-using-deep-learning-166f1b8d8595)<br>

---
### Deep Voice 3
**Blog:** [Deep Voice 3: Scaling Text to Speech with Convolutional Sequence Learning](https://medium.com/a-paper-a-day-will-have-you-screaming-hurray/day-6-deep-voice-3-scaling-text-to-speech-with-convolutional-sequence-learning-16c3e8be4eda)<br>
**Paper:** [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)<br>
**Code:** [r9y9/deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)<br>
**Code:** [Kyubyong/deepvoice3](https://github.com/Kyubyong/deepvoice3)<br>

![](https://miro.medium.com/max/700/1*06JbKxq2eS9G8yO-fhELWg.png)

---
### Neural Voice Cloning
**Paper:** [Neural Voice Cloning with a Few Samples](https://arxiv.org/abs/1802.06006)<br>
**Code:** [SforAiDl/Neural-Voice-Cloning-With-Few-Samples](https://github.com/SforAiDl/Neural-Voice-Cloning-With-Few-Samples)<br>

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/cb3e87412d2fa52441e40bff3db2135dec9de3b9/4-Figure1-1.png)

---
### SV2TTS
**Blog:** [Voice Cloning: Corentin's Improvisation On SV2TTS](https://www.datasciencecentral.com/profiles/blogs/voice-cloning-corentin-s-improvisation-on-sv2tts)<br>
**Paper:** [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)<br>
**Code:** [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)<br>

![](https://storage.ning.com/topology/rest/1.0/file/get/8569870256)

**Synthesizer** : The synthesizer is Tacotron2 without Wavenet<br>
![](https://storage.ning.com/topology/rest/1.0/file/get/8569881687)

**SV2TTS Toolbox**<br>
<iframe width="1148" height="646" src="https://www.youtube.com/embed/-O_hYhToKoA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### MelGAN-VC
**Paper:** [MelGAN-VC: Voice Conversion and Audio Style Transfer on arbitrarily long samples using Spectrograms](https://arxiv.org/abs/1910.03713)<br>
**Code:** [marcoppasini/MelGAN-VC](https://github.com/marcoppasini/MelGAN-VC)<br>

![](https://d3i71xaburhd42.cloudfront.net/498cdaa589a17bf9a28f85005617088f39685fc2/2-Figure1-1.png)

---
### Vocoder-free End-to-End Voice Conversion
**Paper:** [Vocoder-free End-to-End Voice Conversion with Transformer Network](https://arxiv.org/abs/2002.03808)<br>
**Code:** [kaen2891/kaen2891.github.io](https://github.com/kaen2891/kaen2891.github.io)<br>

![](https://d3i71xaburhd42.cloudfront.net/f0d09082d4ea2b201c992c93eec1d32e7ff166ea/3-Figure1-1.png)

---
### ConVoice
**Paper:** [ConVoice: Real-Time Zero-Shot Voice Style Transfer with Convolutional Network](https://arxiv.org/abs/2005.07815)<br>
**Demo:** [ConVoice: Real-Time Zero-Shot Voice Style Transfer](https://rebryk.github.io/convoice-demo/)<br>

![](https://d3i71xaburhd42.cloudfront.net/c142f05e1577048f712eb3a240baab50ea862301/2-Figure1-1.png)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


