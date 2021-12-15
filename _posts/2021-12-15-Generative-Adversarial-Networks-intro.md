---
layout: post
title: Generative Adversarial Networks Introduction
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Style Transfer, GAN, Image Inpainting, DeepFaceDrawing, Toonify, PoseGAN, Deep Fashion Try-On, DeepFakes, Nerual Talking Head, VAE, Music Seperation, Deep Singer, Voice Conversion.

---
## Style Transfer

### [DeepDream](https://deepdreamgenerator.com/)

![](https://b2h3x3f6.stackpathcdn.com/assets/landing/img/gallery/4.jpg)

---
### Nerual Style Transfer
**Paper:** [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)<br>
**Code:** [ProGamerGov/neural-style-pt](https://github.com/ProGamerGov/neural-style-pt)<br>

![](https://miro.medium.com/max/700/1*sBNwIsv5pPQQqHHEAkBHIw.png)

<table>
<tr>
<td><img src="https://raw.githubusercontent.com/ProGamerGov/neural-style-pt/master/examples/inputs/starry_night_google.jpg"></td>
<td><img src="https://raw.githubusercontent.com/ProGamerGov/neural-style-pt/master/examples/inputs/hoovertowernight.jpg"></td>
<td><img src="https://raw.githubusercontent.com/ProGamerGov/neural-style-pt/master/examples/outputs/starry_stanford_bigger.png"></td>
</tr>
</table>

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

---
## GAN
**Paper:** [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)<br>
G是生成的神經網路，它接收一個隨機的噪訊z，通過這個噪訊生成圖片，為G(z)<br>
D是辨别的神經網路，辨别一張圖片夠不夠真實。它的輸入參數是x，x代表一張圖片，輸出D(x)代表x為真實圖片的機率<br>
![](https://developers.google.com/machine-learning/gan/images/gan_diagram.svg)

**Blog:** [A Beginner's Guide to Generative Adversarial Networks (GANs)](https://wiki.pathmind.com/generative-adversarial-network-gan)<br>

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
### AniGAN
**Paper:** [AniGAN: Style-Guided Generative Adversarial Networks for Unsupervised Anime Face Generation](https://arxiv.org/abs/2102.12593)<br>
**Blog:** [博士後小姐姐把「二次元老婆生成器」升級了：這一次可以指定畫風](https://bangqu.com/5uWn8B.html)<br>

![](https://d2ndd3gtcc6iwc.cloudfront.net/liang/news/20210302/v2-f716db3e91ba8587af8897d55e9e59d1_b.jpg)
![](https://d2ndd3gtcc6iwc.cloudfront.net/liang/news/20210302/v2-0d272ec5646191d4b0a11ffefb41c408_b.jpg)
![](https://d2ndd3gtcc6iwc.cloudfront.net/liang/news/20210302/v2-15685fdb3dbe7c63e12b5da8811c5d51_b.jpg)

---
### CartoonGAN
**Paper:** [CartoonGAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)<br>
**Code:** [mnicnc404/CartoonGan-tensorflow](https://github.com/mnicnc404/CartoonGan-tensorflow)<br>

![](https://gmarti.gitlab.io/assets/cartoongan/architecture_cartoogan.png)
![](https://github.com/mnicnc404/CartoonGan-tensorflow/blob/master/images/cover.gif?raw=true)

---
### Cartoon-GAN
**Paper:** [Generative Adversarial Networks for photo to Hayao Miyazaki style cartoons](https://arxiv.org/abs/2005.07702)<br>
**Code:** [FilipAndersson245/cartoon-gan](https://github.com/FilipAndersson245/cartoon-gan)<br>

![](https://camo.githubusercontent.com/6e53ffe052d3488540d9fe6422bfa4c6579513d7afe6cee3b89d10ceccf17685/68747470733a2f2f7468756d62732e6766796361742e636f6d2f476c6172696e675175657374696f6e61626c654b6f616c612d73697a655f726573747269637465642e676966)

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
### GAN Compression
**Paper:** [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/abs/2003.08936)<br>
**Code:** [mit-han-lab/gan-compression](https://github.com/mit-han-lab/gan-compression)<br>

![](https://github.com/mit-han-lab/gan-compression/raw/master/imgs/teaser.png)
![](https://github.com/mit-han-lab/gan-compression/blob/master/imgs/overview.png?raw=true)

---
### SRGAN
**Paper:** [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)<br>
**Code:** [tensorlayer/srgan](https://github.com/tensorlayer/srgan)<br>
![](https://github.com/tensorlayer/srgan/raw/master/img/model.jpeg?raw=true)
![](https://github.com/tensorlayer/srgan/raw/master/img/SRGAN_Result2.png?raw=true)

---
## Image Inpainting
### High-Resolution Image Inpainting
**Paper:** [High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis](https://arxiv.org/abs/1611.09969)<br>
**Code:** [leehomyc/Faster-High-Res-Neural-Inpainting](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting)<br>

![](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting/blob/master/images/teaser.png?raw=true)

---
### Image Inpainting for Irregular Holes
**Paper:** [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)<br>
**Code:** [NVIDIA/partialconv](https://github.com/NVIDIA/partialconv)<br>
**Blog:** [Image Inpainting for Irregular Holes Using Partial Convolutions](https://medium.com/@neurohive/image-inpainting-for-irregular-holes-using-partial-convolutions-e46ab64f9570)<br>
![](https://miro.medium.com/max/2400/0*NTUlZCsCYHcRI7y4.)
![](https://miro.medium.com/max/2000/1*HUmj7An3CvGrJiTZAgiHBw.png)

---
### DeepFill V2
**Paper:** [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)<br>
**Code:** [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting)<br>
**Blog:** [A Practical Generative Deep Image Inpainting Approach](https://towardsdatascience.com/a-practical-generative-deep-image-inpainting-approach-1c99fef68bd7)<br>
![](https://miro.medium.com/max/1225/1*UaweIaCSL8HmFG9jh-KGlA.png)
![](https://miro.medium.com/max/1400/1*Q38k2RnxBkgWSJxzblzbJA.png)
<table>
<tr>
<td><img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_raw.png"></td>
<td><img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_input.png"></td>
<td><img src="https://raw.githubusercontent.com/JiahuiYu/generative_inpainting/v2.0.0/examples/places2/case4_output.png"></td>
</tr>
</table>

---
### EdgeConnect
**Paper:** [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://arxiv.org/abs/1901.00212)<br>
**Code:** [knazeri/edge-connect](https://github.com/knazeri/edge-connect)<br>

![](https://miro.medium.com/max/2640/1*KKiBBWo20W2BjrzEViWVVA.png)
![](https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png)

---
### Deep Flow-Guided Video Inpainting
**Paper:** [Deep Flow-Guided Video Inpainting](https://arxiv.org/abs/1905.02884)<br>
**Code:** [nbei/Deep-Flow-Guided-Video-Inpainting](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)<br>
![](https://nbei.github.io/video-inpainting/framework.png)
![](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/blob/master/gif/captain.gif?raw=true)
![](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/raw/master/gif/flamingo.gif?raw=true)

---
### Flow-edge Guided Video Completion
**Paper:** [Flow-edge Guided Video Completion](https://arxiv.org/abs/2009.01835)<br>
**Code:** [vt-vl-lab/FGVC](https://github.com/vt-vl-lab/FGVC)<br>

![](https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-58610-2_42/MediaObjects/504453_1_En_42_Fig2_HTML.png)
<iframe width="585" height="329" src="https://www.youtube.com/embed/CHHVPxHT7rc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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
## Variational AutoEncoder
### VAE
**Paper:** [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)<br>
**Code:** [rkuo2000/fashionmnist-vae](https://www.kaggle.com/rkuo2000/fashionmnist-vae)<br>
**Blog:** [VAE(Variational AutoEncoder) 實作](https://ithelp.ithome.com.tw/articles/10226549)<br>

![](https://github.com/timsainb/tensorflow2-generative-models/blob/master/imgs/vae.png?raw=1)
![](https://i.imgur.com/ZN6MyTx.png)
![](https://ithelp.ithome.com.tw/upload/images/20191009/20119971nNxkMbzOB8.png)

---
### VQ-AVE
**Paper:** [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)<br>

![](https://miro.medium.com/max/700/1*9GZoBSZPw4VelO2vV9KfDw.png)

**Paper:** [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)<br>

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-28_at_4.56.19_PM.png)

**Blog:** [帶你認識Vector-Quantized Variational AutoEncoder - 理論篇](https://medium.com/ai-academy-taiwan/%E5%B8%B6%E4%BD%A0%E8%AA%8D%E8%AD%98vector-quantized-variational-autoencoder-%E7%90%86%E8%AB%96%E7%AF%87-49a1829497bb)<br>

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
**Paper:** [Generating Handwritten Chinese Characters using CycleGAN](https://arxiv.org/abs/1801.08624)<br>
**Code:** [kaonashi-tyc/zi2zi](https://github.com/kaonashi-tyc/zi2zi)<br>
**Blog:** [zi2zi: Master Chinese Calligraphy with Conditional Adversarial Networks](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html)<br>

![](https://github.com/kaonashi-tyc/zi2zi/blob/master/assets/intro.gif?raw=true)
![](https://kaonashi-tyc.github.io/assets/network.png)

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
**Paper:** [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341)<br>
**Colab:** [Interacting with Jukebox](https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb)<br>
**Blog:** [Jukebox](https://openai.com/blog/jukebox/)<br>
model modified from **VQ-VAE-2**

---
### DeepSinger
**Paper:** [DeepSinger: Singing Voice Synthesis with Data Mined From the Web](https://arxiv.org/abs/2007.04590)<br>
**Demo:** [DeepSinger: Singing Voice Synthesis with Data Mined From the Web](https://speechresearch.github.io/deepsinger/)<br>
**Blog:** [Microsoft’s AI generates voices that sing in Chinese and English](https://venturebeat.com/2020/07/13/microsofts-ai-generates-voices-that-sing-in-chinese-and-english/)<br>

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
**Paper:** [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)<br>
**Code:** [r9y9/deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch)<br>
**Code:** [Kyubyong/deepvoice3](https://github.com/Kyubyong/deepvoice3)<br>
**Blog:** [Deep Voice 3: Scaling Text to Speech with Convolutional Sequence Learning](https://medium.com/a-paper-a-day-will-have-you-screaming-hurray/day-6-deep-voice-3-scaling-text-to-speech-with-convolutional-sequence-learning-16c3e8be4eda)<br>

![](https://miro.medium.com/max/700/1*06JbKxq2eS9G8yO-fhELWg.png)

---
### Neural Voice Cloning
**Paper:** [Neural Voice Cloning with a Few Samples](https://arxiv.org/abs/1802.06006)<br>
**Code:** [SforAiDl/Neural-Voice-Cloning-With-Few-Samples](https://github.com/SforAiDl/Neural-Voice-Cloning-With-Few-Samples)<br>

![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/cb3e87412d2fa52441e40bff3db2135dec9de3b9/4-Figure1-1.png)

---
### SV2TTS
**Paper:** [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)<br>
**Code:** [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)<br>
**Blog:** [Voice Cloning: Corentin's Improvisation On SV2TTS](https://www.datasciencecentral.com/profiles/blogs/voice-cloning-corentin-s-improvisation-on-sv2tts)<br>

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


