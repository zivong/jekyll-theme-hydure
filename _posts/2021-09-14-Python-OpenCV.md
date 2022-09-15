---
layout: post
title: OpenCV in Python
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

OpenCV Image Processing in Python

---
## Editor & Bash

### Code Editor
**[Notepad++](https://notepad-plus-plus.org/)**
![](https://notepad-plus-plus.org/assets/images/notepad4ever.png)

---
### Linux Bash
Bash(GNU Bourne-Again Shell)是許多Linux平臺的內定Shell，事實上，還有許多傳統UNIX上用的Shell，像tcsh、csh、ash、bsh、ksh等等<br>

* [Git for Windows](https://gitforwindows.org/): provide a BASH emulation
  - [Git安裝在Windows系統](https://gitbook.tw/chapters/environment/install-git-in-windows)
  ![](https://gitbook.tw/images/tw/environment/install-git-in-windows/install3.png)

---
## Python OpenCV

### Python3
* [Python for Windows](https://www.python.org/downloads/): `Download Python3.10.7`<br>
  ![](https://store-images.s-microsoft.com/image/apps.50279.14277708378433140.89e2a756-3562-4beb-873f-e7f7eda6d96a.f574468c-b1e5-4ee3-8906-5f9054dac143)
  ![](https://store-images.s-microsoft.com/image/apps.25395.14277708378433140.89e2a756-3562-4beb-873f-e7f7eda6d96a.660e5f72-67b3-451f-8f81-0e57c3011cc3)
* Ubuntu 22.04 LTS
  ![](https://ubuntuhandbook.org/wp-content/uploads/2021/10/python310-final.png)
  
---
### Install OpenCV
```
cd ~
pip3 install numpy pandas
pip3 install matplotlib pillow imutils
pip3 install opencv-python
pip3 install opencv-contrib-python
git clone https://github.com/rkuo2000/cv2
cd cv2
```

---
### [Python Programming](https://www.programiz.com/python-programming)

---
## OpenCV Image Processing

### read JPEG
`python3 jpg_read.jpg`<br>
[jpg_read.py](https://github.com/rkuo2000/cv2/blob/master/jpg_read.py)<br>
```
import cv2
import sys

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = 'test.jpg'
	
img = cv2.imread(filename)

print(type(img))

print(img.shape)

cv2.imshow('Image',img)

cv2.waitKey(0)
cv2.DestroyAllWindows()
```

---
### open Camera
`python3 cam.py`<br>
[cam.py](https://github.com/rkuo2000/cv2/blob/master/cam.py)<br>

```
import cv2
import sys
if len(sys.argv) >1:
    vid = int(sys.argv[1])
else:
    vid = 0
cap = cv2.VideoCapture(vid)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);

while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape)
    #frame = cv2.flip(frame, 1) # 0: vertical flip, 1: horizontal flip

    cv2.imshow('Camera', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---
### Color system: RGB, CMYK
* [RGB和CMYK的色彩模式差異](https://jessielab.com/rgb%E5%92%8Ccmyk%E7%9A%84%E8%89%B2%E5%BD%A9%E6%A8%A1%E5%BC%8F%E5%B7%AE%E7%95%B0%EF%BC%8C%E9%97%9C%E6%96%BC%E8%A8%AD%E8%A8%88%E9%87%8D%E8%A6%81%E7%9A%84%E5%B0%8F%E4%BA%8B/)<br>
<table>
<tr>
<td><img src="https://jessielab.com/wp-content/uploads/2019/06/RGB-02.png"></td>
<td><img src="https://jessielab.com/wp-content/uploads/2019/06/CMYK-02.png"></td>
</tr>
</table>

* [電腦色彩學 – CMYK與RGB 的差異與設定](https://www.soongrow.com/professional-printing-packaging/1440/)
![](https://www.soongrow.com/wp-content/uploads/2020/09/Image.jpg)

---
* [Color Space](https://handwiki.org/wiki/Color_space)
![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/CIE1931xy_gamut_comparison.svg/300px-CIE1931xy_gamut_comparison.svg.png)

* [Munsell color system](https://en.wikipedia.org/wiki/Munsell_color_system)
hue (basic color), chroma (color intensity), and value (lightness)<br>
<table>
<tr>
<td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Munsell-system.svg/290px-Munsell-system.svg.png"></td>
<td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Munsell_1943_color_solid_cylindrical_coordinates_gray.png/255px-Munsell_1943_color_solid_cylindrical_coordinates_gray.png"></td>
</tr>
</table>

* [HSL and HSV](https://zh.wikipedia.org/zh-tw/HSL%E5%92%8CHSV%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%97%B4)<br>
**H**ue, **S**aturation, **L**ightness and **H**ue色調, **S**aturation 飽和度, **V**alue亮度<br>
![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/HSL_HSV_cylinder_color_solid_comparison.png/400px-HSL_HSV_cylinder_color_solid_comparison.png)
HSV色輪允許用戶快速的選擇眾多顏色<br>
![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Triangulo_HSV.png/220px-Triangulo_HSV.png)
HSV模型的圓錐表示適合於在一個單一物體中展示整個HSV色彩空間<br>
![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/HSV_cone.png/220px-HSV_cone.png)

---
### [OpenCV color conversion](https://docs.opencv.org/4.6.0/de/d25/imgproc_color_conversions.html)
[jpg_csc.py](https://github.com/rkuo2000/cv2/blob/master/jpg_csc.py)
```
import cv2

img = cv2.imread('test.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
org1= cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
org2= cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('ORG' ,img)
cv2.imshow('GRAY',gray)
cv2.imshow('HSV' ,hsv)
cv2.imshow('ORG1',org1)
cv2.imshow('ORG2',org2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
### Object Tracking by Color
[cam_object_tracking.py](https://github.com/rkuo2000/cv2/blob/master/cam_object_tracking.py)<br>

```
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
	
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
	
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
	
    cv2.imshow('FRAME', frame)
    cv2.imshow('MASK', mask)
    cv2.imshow('RESULT', res)
    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break
	
cv2.destroyAllWindows()
```

---
### [Multi Object Tracking based on Color](http://akaifi.github.io/MultiObjectTrackingBasedOnColor/)
<iframe width="728" height="624" src="https://www.youtube.com/embed/hQ-bpfdWQh8" title="Multiple Object Detection with Color Using OpenCV" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [Skin Isolation](https://nalinc.github.io/blog/2018/skin-detection-python-opencv/)
[cam_skin_detection.py](https://github.com/rkuo2000/cv2/blob/master/cam_skin_detection.py)<br>
[jpg_skin_detection.py](https://github.com/rkuo2000/cv2/blob/master/jpg_skin_detection.py)<br>
![](https://nalinc.github.io/assets/img/skin-detection/hsv.png)

---
### [Image Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

* **Simple Thresholding**: [jpg_image_thresholding.py](https://github.com/rkuo2000/cv2/blob/master/jpg_image_thresholding.py)
![](https://docs.opencv.org/4.x/threshold.jpg)
  - cv2.THRESH_BINARY
  - cv2.THRESH_BINARY_INV
  - cv2.THRESH_TRUNC
  - cv2.THRESH_TOZERO
  - cv2.THRESH_TOZERO_INV
  
* **Adaptive Thresholding**: [jpg_adaptive_thresholding.py](https://github.com/rkuo2000/cv2/blob/master/jpg_adaptive_thresholding.py)
![](https://docs.opencv.org/4.x/ada_threshold.jpg)

* **Otsu’s Binarization**: [jpg_otsu_binarization.py](https://github.com/rkuo2000/cv2/blob/master/jpg_otsu_binarization.py)
![](https://docs.opencv.org/4.x/otsu.jpg)

---
## [Image Processing Tutorial](https://docs.opencv.org/4.6.0/d7/da8/tutorial_table_of_content_imgproc.html)

### [Smoothing Images](https://docs.opencv.org/4.6.0/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html)
* blur()
* GaussianBlur()
* medianBlur()
* bilateralFilter()
* filter2D(): [jpg_2dfilter.py](https://github.com/rkuo2000/cv2/blob/master/jpg_2dfilter.py)

![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/OpenCV_Smoothing_Images.png?raw=true)

---
### [Morphological Transformations](https://docs.opencv.org/4.6.0/d4/d76/tutorial_js_morphological_ops.html)
* Erosion
* Dilation
* Opening
* Closing
* Morphological Gradient
* Top Hat
* Black Hat
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/OpenCV_morphological_transformations.png?raw=true)

[jpg_morphological_transformations.py](https://github.com/rkuo2000/cv2/blob/master/jpg_morphological_transformations.py)<br>
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

titles = ['Image','Erosion','Dilation','Opening','Closing','Gradient','Tophat','Blackhat']
images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]

for i in range(8):
    plt.subplot(2,4,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```

---
### [Geometric Transformation](https://docs.opencv.org/4.6.0/dd/d52/tutorial_js_geometric_transformations.html)
* Scaling (Resize): [jpg_resize.py](https://github.com/rkuo2000/cv2/blob/master/jpg_resize.py)
* Rotation : [jpg_rotation.py](https://github.com/rkuo2000/cv2/blob/master/jpg_rotation.py)
* Affine Transform: [jpg_affine_transformation.py](https://github.com/rkuo2000/cv2/blob/master/jpg_affine_transformation.py)
  ![](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdn.net/20180814101638244?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podV9ob25namk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
* Perspective Transform [jpg_perspective_transformation.py](https://github.com/rkuo2000/cv2/blob/master/jpg_perspective_transformation.py)
  ![](https://pic1.xuehuaimg.com/proxy/csdn/https://img-blog.csdn.net/20180814101302480?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podV9ob25namk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Reference: [OpenCV學習筆記】之仿射變換（Affine Transformation）](https://www.twblogs.net/a/5b7b00e42b7177539c24a869)<br>

---
### [Image Gradients](https://docs.opencv.org/4.6.0/d5/d0f/tutorial_py_gradients.html)
![](https://docs.opencv.org/4.6.0/gradients.jpg)
[jpg_sobel.py](https://github.com/rkuo2000/cv2/blob/master/jpg_sobel.py)<br>
```
import cv2

org = cv2.imread('test.jpg')
gray  = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
img   = cv2.GaussianBlur(gray, (3,3), 0) # remove noise

# convolute with proper kernels
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('SobelX', sobel_x)
cv2.imshow('SobelY', sobel_y)

abs_grad_x = cv2.convertScaleAbs(sobel_x)
abs_grad_y = cv2.convertScaleAbs(sobel_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)    
cv2.imshow('Sobel', grad)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
### [Fourier Transform](https://docs.opencv.org/4.6.0/de/dbc/tutorial_py_fourier_transform.html)
[jpg_fft.py](https://github.com/rkuo2000/cv2/blob/master/jpg_fft.py)
![](https://docs.opencv.org/4.6.0/fft1.jpg)

[jpg_dft.py](https://github.com/rkuo2000/cv2/blob/master/jpg_dft.py)
![](https://docs.opencv.org/4.6.0/fft4.jpg)

* Why Laplacian/Sobel is a High Pass Filter?
![](https://docs.opencv.org/4.6.0/fft5.jpg)
From image, you can see what frequency region each kernel blocks, and what region it passes. From that information, we can say why each kernel is a HPF or a LPF

---
### [Spatial Frequency Filtering](https://www.djmannion.net/psych_programming/vision/sf_filt/sf_filt.html)
* UNSW 
![](https://www.djmannion.net/psych_programming/_images/sf_a1.png)
* Converting to frequency space - [jpg_spatial_frequency.py](https://github.com/rkuo2000/cv2/blob/master/jpg_spatial_frequency.py)
![](https://www.djmannion.net/psych_programming/_images/sf_a5.png)
* Creating a spatial frequency filter - [jpg_spatial_frequency_filter.py](https://github.com/rkuo2000/cv2/blob/master/jpg_spatial_frequency_filter.py)
![](https://www.djmannion.net/psych_programming/_images/sf_a6.png)
* Applying a spatial frequency filter - [jpg_spatial_frequency_filtering.py](https://github.com/rkuo2000/cv2/blob/master/jpg_spatial_frequency_filtering.py)
![](https://www.djmannion.net/psych_programming/_images/sf_a7.png)
* Converting back to an image - [jpg_spatial_frequency_filtered.py](https://github.com/rkuo2000/cv2/blob/master/jpg_spatial_frequency_filtered.py)
![](https://www.djmannion.net/psych_programming/_images/sf_a8.png)
* Other spatial frequency filters
![](https://www.djmannion.net/psych_programming/_images/sf_a9.png)

---
### [Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) <br>
1. Noise Reduction : To remove the noise in the image with a 5x5 Gaussian filter. 
2. Finding Intensity Gradient of the Image
   ![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/OpenCV_Canny_finding_intensity_gradient.png?raw=true)
3. Non-maximum Suppression
   ![](https://docs.opencv.org/4.x/nms.jpg)
4. Hysteresis Thresholding
   ![](https://docs.opencv.org/4.x/hysteresis.jpg)
[邊緣偵測懶人包-Canny演算法](https://medium.com/@bob800530/opencv-%E5%AF%A6%E4%BD%9C%E9%82%8A%E7%B7%A3%E5%81%B5%E6%B8%AC-canny%E6%BC%94%E7%AE%97%E6%B3%95-d6e0b92c0aa3)<br>

[jpg_canny.py](https://github.com/rkuo2000/cv2/blob/master/jpg_canny.py)
![](https://docs.opencv.org/4.x/canny1.jpg)

---
### [Hough Line Transform](https://docs.opencv.org/4.6.0/d9/db0/tutorial_hough_lines.html)
[Hough Transform](https://learnopencv.com/hough-transform-with-opencv-c-python/)
* Hough Lines
![](https://learnopencv.com/wp-content/uploads/2019/03/line-detection.jpg)
[jpg_houghlines.py](https://github.com/rkuo2000/cv2/blob/master/jpg_houghlines.py)<br>

```
img = cv2.imread('lanes.jpg', cv2.IMREAD_COLOR) # road.png is the filename
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider, minLineLength=10, maxLineGap=250)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imshow("Result Image", img)
```

* Hough Circles
![](https://learnopencv.com/wp-content/uploads/2019/03/circle-detection.jpg)
![](https://learnopencv.com/wp-content/uploads/2019/03/circle-detection-hough-transform-opencv.jpg)
[jpg_houghcircles.py](https://github.com/rkuo2000/cv2/blob/master/jpg_houghcircles.py)<br>

```
img = cv2.imread('circles.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=5, maxRadius=30)

# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:   
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2) # Draw outer circle       
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3) # Draw inner circle
```

---
### [Image Histogram](https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_image_histogram_calcHist.php)

* **Gray to Histogram**<br>

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

gray_img = cv2.imread('images/GoldenGateSunset.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('GoldenGate',gray_img)
hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
plt.hist(gray_img.ravel(),256,[0,256])
plt.title('Histogram for gray scale picture')
plt.show()
```

<table>
<tr>
<td><img src="https://www.bogotobogo.com/python/OpenCV_Python/images/Histogram/GGsunset.png"></td>
<td><img src="https://www.bogotobogo.com/python/OpenCV_Python/images/Histogram/Histo_gray.png"></td>
</tr>
</table>

* **Color to Histogram**<br>

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/GoldenGateSunset.png', -1)
cv2.imshow('GoldenGate',img)

color = ('b','g','r')
for channel,col in enumerate(color):
    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histogram for color scale picture')
plt.show()
```

<table>
<tr>
<td><img src="https://www.bogotobogo.com/python/OpenCV_Python/images/Histogram/GoldenGateSunsetCV.png"></td>
<td><img src="https://www.bogotobogo.com/python/OpenCV_Python/images/Histogram/GoldenGateSunsetCV.png"></td>
</tr>
</table>

* **Histogram Equalization**<br>
[jpg_histogram_equalization.py](https://github.com/rkuo2000/cv2/blob/master/jpg_histogram_equalization.py)<br>

```
img = cv2.imread('test.jpg')
src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(src)
cv2.imshow('Source', src)
cv2.imshow('Equalized', dst)
```

---
* **Histogram Backprojection**<br>
[jpg_histogram_backprojection.py](https://github.com/rkuo2000/cv2/blob/master/jpg_histogram_backprojection.py)<br>

---
### [Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
[jpg_template_matching.py](https://github.com/rkuo2000/cv2/blob/master/jpg_template_matching.py)<br>
![](https://docs.opencv.org/4.x/template_ccoeff_1.jpg)

[jpg_template_matching_objects.py](https://github.com/rkuo2000/cv2/blob/master/jpg_template_matching_objects.py)<br>
![](https://docs.opencv.org/4.x/res_mario.jpg)
```
img_rgb = cv.imread('mario.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('mario_coin.png',0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img_rgb)
```

---
### Contours
* 求圖像輪廓  cnts, hierarchy = cv2.findContours(thresh, mode, method)
* 畫輪廓 cv2.drawContours(img, cnts, contourIdx, color, lineType)
* 求包覆矩形 (x,y,w,h) = cv2.boundingRect(cnt)
* 求包覆矩形 box = cv2.minAreaRect(cnt)
* 求包覆圓形  ((x,y), radius) = cv2.minEnclosingCircle(cnt)
* 求包覆橢圓形   ellipse = cv2.fitEllipse(cnt)
* 計算輪廓面積    area = cv2.contourArea(cnt)

* `contours, hierarchy = cv2.findContours(thresh, mode, method)`<br>
  - img : output image
  - contours：包含所有輪廓的容器(vector)，每個輪廓都是儲存點的容器(vector)，所以contours的資料結構為vector< vector>。
  - hierarchy：可有可無的輸出向量，以階層的方式記錄所有輪廓
  - thresh：輸入圖，使用八位元單通道圖，所有非零的像素都會列入考慮，通常為二極化後的圖 
  - mode：取得輪廓的模式
  - cv2.RETR_EXTERNAL：只取最外層的輪廓。
  - cv2.RETR_LIST：取得所有輪廓，不建立階層(hierarchy)。
  - cv2.RETR_CCOMP：取得所有輪廓，儲存成兩層的階層，首階層為物件外圍，第二階層為內部空心部分的輪廓，如果更內部有其餘物件，包含於首階層。
  - cv2.RETR_TREE：取得所有輪廓，以全階層的方式儲存。
  - method：儲存輪廓點的方法
  - cv2.CHAIN_APPROX_NONE：儲存所有輪廓點。
  - cv2.CHAIN_APPROX_SIMPLE：對水平、垂直、對角線留下頭尾點，所以假如輪廓為一矩形，只儲存對角的四個頂點。

* `cv2.drawContours(image, contours, contourIdx, color, lineType)`<br>
  - image：輸入輸出圖，會將輪廓畫在此影像上
  - contours：包含所有輪廓的容器(vector)，也就是findContours()所找到的contours
  - contourIdx：指定畫某個輪廓 (-1 = all)
  - color：繪製的顏色 (0,0,255) in G-B-R
  - lineType：繪製的線條型態

* Examples:
  - [jpg_contours.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contours.py)
  - [jpg_contours_boundingRect.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contours_boundingRect.py)
  - [jpg_contours_AreaRect.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contours_AreaRect.py)
  - [jpg_contours_EnclosingCircle.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contours_EnclosingCircle.py)
  - [jpg_contours_fitEllipse.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contours_fitEllipse.py)
  - [jpg_contour_flower.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contour_flower.py)
  - [jpg_contour_golf.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contour_golf.py)
  - [jpg_contour_hand.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contour_hand.py)  
  - [jpg_contour_lawn.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contour_lawn.py)

---
### [Hand Contour](https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/)
![](https://929687.smushcdn.com/2633864/wp-content/uploads/2016/04/extreme_points_header.jpg?lossy=1&strip=1&webp=1)
[jpg_contour_hand.py](https://github.com/rkuo2000/cv2/blob/master/jpg_contour_hand.py)<br>
```
img_path = "hand.jpg"
img = cv.imread(img_path)

# define the upper and lower boundaries of the HSV pixel intensities 
# to be considered 'skin'
hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")
skinMask= cv.inRange(hsvim, lower, upper)

# blur the mask to help remove noise
skinMask= cv.blur(skinMask, (2, 2))

# get threshold image
ret, thresh = cv.threshold(skinMask, 100, 255, cv.THRESH_BINARY)
cv.imshow("thresh", thresh)

# draw the contours on the empty image
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = max(contours, key=lambda x: cv.contourArea(x))
cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
cv.imshow("contours", img)

cv.waitKey()
```

---
### Hand Detection and Finger Counting](https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08)
![](https://miro.medium.com/max/700/1*O5rRGGWEsc7zWNFyIQGunA.jpeg)

---
### [Hand Detection & Gesture Recognition](https://aihubprojects.com/hand-detection-gesture-recognition-opencv-python/)
`pip install cvzone`<br>

```
import cvzone
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = cvzone.HandDetector(detectionCon=0.5, maxHands=1)

while True:
    # Get image frame
    success, img = cap.read()

    # Find the hand and its landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```

---

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

