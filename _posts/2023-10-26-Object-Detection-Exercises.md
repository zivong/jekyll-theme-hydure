---
layout: post
title: Object Detection Exercises
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

The exercises includes Image Annotation Tools, examples of YOLOv4, v5, v6, v7, YOLOR, YOLOX, CSL-YOLO, and YOLOv5 applications, 
Mask RCNN, SSD MobileNet, YOLOv5+DeepSort, Objectron, Steel Defect Detection, PCB Defect Detection, Identify Military Vehicles in Satellite Imagery, Pothole Detection, Car Breaking Detection.

---
## Image Annotation
### [FiftyOne](https://voxel51.com/docs/fiftyone/)
[Annotating Datasets with LabelBox](https://voxel51.com/docs/fiftyone/tutorials/labelbox_annotation.html)<br>
To get started, you need to [install FiftyOne](https://voxel51.com/docs/fiftyone/getting_started/install.html) and [the Labelbox Python client](https://github.com/Labelbox/labelbox-python):<br>
`!pip install fiftyone labelbox`<br>
![](https://voxel51.com/docs/fiftyone/_images/labelbox_detection.png)

---
### [Labelme](https://github.com/wkentaro/labelme)
![](https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/.readme/annotation.jpg?raw=true)
`$pip install labelme`<br>

### [Labelme2YOLO](https://github.com/rooneysh/Labelme2YOLO)
`$pip install labelme2yolo`<br>

* Convert JSON files, split training and validation dataset by --val_size<br>
`python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --val_size 0.2`<br>

---
### [LabelImg](https://github.com/tzutalin/labelImg)
![](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)
`$pip install labelImg`<br>

`$labelImg`<br>
`$labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]`<br>
---
### VOC .xml convert to YOLO .txt
`$cd ~/tf/raccoon/annotations`
`$python ~/tf/xml2yolo.py`

---
### Annotation formats
* YOLO format in .txt
**class_num x, y, w, h**<br>
```
0 0.5222826086956521 0.5518115942028986 0.025 0.010869565217391304
0 0.5271739130434783 0.5057971014492754 0.013043478260869565 0.004347826086956522
```

* COCO format in .xml

```
<annotation>
	<folder>JPEGImages</folder>
	<filename>BloodImage_00000.jpg</filename>
	<path>/home/pi/detection_dataset/JPEGImages/BloodImage_00000.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>640</width>
		<height>480</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>WBC</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>260</xmin>
			<ymin>177</ymin>
			<xmax>491</xmax>
			<ymax>376</ymax>
		</bndbox>
	</object>
```

---
## YOLOs

### YOLOv4
**Kaggle:** [rkuo2000/yolov4](https://kaggle.com/rkuo2000/yolov4)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv4_PyTorch_horses.jpg?raw=true)

### YOLOv5
**Kaggle:** [rkuo2000/yolov5](https://kaggle.com/rkuo2000/yolov5)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5s_horses.jpg?raw=true)

### Scaled YOLOv4
**Kaggle:** [rkuo2000/scaled-yolov4](https://kaggle.com/rkuo2000/scaled-yolov4)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Scaled_YOLOv4_horses.jpg?raw=true)

### YOLOR
**Kaggle:** [rkuo2000/yolor](https://kaggle.com/rkuo2000/yolor)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOR_horses.jpg?raw=true)

### YOLOX
**Kaggle:** [rkuo2000/yolox](https://www.kaggle.com/code/rkuo2000/yolox)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOX_horses.jpg?raw=true)

### CSL-YOLO
**Kaggle:** [rkuo2000/csl-yolo](https://kaggle.com/rkuo2000/csl-yolo)
![](https://github.com/D0352276/CSL-YOLO/blob/main/dataset/coco/pred/000000000001.jpg?raw=true)

### PP-YOLOE
**Kaggle:** [rkuo2000/pp-yoloe](https://www.kaggle.com/code/rkuo2000/pp-yoloe)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/PP-YOLOE_demo.jpg?raw=true)

### YOLOv6
**Kaggle:** [rkuo2000/yolov6](https://www.kaggle.com/code/rkuo2000/yolov6)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6s_image1.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6s_image2.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv6s_horses.png?raw=true)

### YOLOv7
**Kaggle:** [rkuo2000/yolov7](https://www.kaggle.com/code/rkuo2000/yolov7)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv7_horses.jpg?raw=true)

---
## YOLOv5 applications
### [YOLOv5 Detect](https://kaggle.com/rkuo2000/yolov5-detect)
detect image / video
<iframe width="498" height="280" src="https://www.youtube.com/embed/IL9GdRQrI-8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### [YOLOv5 Elephant](https://kaggle.com/rkuo2000/yolov5-elephant)
train YOLOv5 for detecting elephant (dataset from OpenImage V6)
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_elephant.jpg?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_elephants.jpg?raw=true"></td>
</tr>
</table>

---
### [YOLOv5 BCCD](https://kaggle.com/rkuo2000/yolov5-bccd)
BCCD Dataset is a small-scale dataset for blood cells detection.<br>
3 classes: RBC (Red Blood Cell), WBC (White Blood Cell), Platelets (血小板)<br>
![](https://github.com/Shenggan/BCCD_Dataset/raw/master/example.jpg)
Github: [https://github.com/Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset)<br>
Kaggle: [https://www.kaggle.com/datasets/surajiiitm/bccd-dataset](https://www.kaggle.com/datasets/surajiiitm/bccd-dataset)<br>

* Directory Structure:<br>

```
├── BCCD
│   ├── Annotations
│   │       └── BloodImage_00000.xml ~ 00410.xml (364 files)
│   ├── ImageSets/Main/train.txt, val.txt, test.txt, trainval.txt (filename list)
│   └── JPEGImages
│       └── BloodImage_00000.jpg ~ 00410.xml (364 files)
```

* Convert Annotations (from COCO .xml to YOLO format .txt)

```
def cord_converter(size, box):
#   convert xml annotation to darknet format coordinates
#   :param size： [w,h]
#   :param box: anchor box coordinates [upper-left x,uppler-left y,lower-right x, lower-right y]
#   :return: converted [x,y,w,h]
    
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

def save_file(img_jpg_file_name, size, img_box):
    save_file_name = LABELS_ROOT + '/' + img_jpg_file_name + '.txt'
    print(save_file_name)
    file_path = open(save_file_name, "a+")
    for box in img_box:

        cls_num = classes.index(box[0]) # find class_id

        new_box = cord_converter(size, box[1:]) # convert box coord into YOLO x,y,w,h

        file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")

    file_path.flush()
    file_path.close()
    
def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '/' + img_xml_file + '.xml'
    print(img_path)

    dom = parse(img_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    img_size = root.getElementsByTagName("size")[0]
    objects = root.getElementsByTagName("object")
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    # print("img_name:", img_name)
    # print("image_info:(w,h,c)", img_w, img_h, img_c)
    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        # print("box:(c,xmin,ymin,xmax,ymax)", cls_name, x1, y1, x2, y2)
        img_jpg_file_name = img_xml_file + '.jpg'
        img_box.append([cls_name, x1, y1, x2, y2])
    # print(img_box)

    # test_dataset_box_feature(img_jpg_file_name, img_box)
    save_file(img_xml_file, [img_w, img_h], img_box)   
```

```
files = os.listdir(ANNOTATIONS_PATH)
for file in files:
    print("file name: ", file)
    file_xml = file.split(".")
    get_xml_data(ANNOTATIONS_PATH, file_xml[0])
```

* Create yaml for YOLO (train, val path & labels)

```
!echo "train: Dataset/images/train\n" > data/bccd.yaml
!echo "val:   Dataset/images/val\n" >> data/bccd.yaml
!echo "nc : 3\n" >> data/bccd.yaml
!echo "names: ['Platelets', 'RBC', 'WBC']\n" >> data/bccd.yaml

!cat data/bccd.yaml
```

---
### [YOLOv5 Helmet](https://kaggle.com/rkuo2000/yolov5-helmet)
<table>
<tr>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Helmet.jpg?raw=true"></td>
<td><img src="https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Helmet_SafeZone.jpg?raw=true"></td>
</tr>
</table>

**[SafetyHelmetWearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)**<br>

```
|--VOC2028    
    |---Annotations    
    |---ImageSets    
    |---JPEGImages   
```
dataset conversion from COCO to YOLO format<br>

---
**[YOLOv5 Facemask](https://kaggle.com/rkuo2000/yolov5-facemask)**<br>
train YOLOv5 for facemask detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_Facemask.jpg?raw=true)

---
**[YOLOv5 Traffic Analysis](https://kaggle.com/rkuo2000/yolov5-traffic-analysis)**<br>
use YOLOv5 to detect car/truck per frame, then analyze vehicle counts per lane and the estimated speed
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_traffic_analysis.jpg?raw=true)

---
**[YOLOv5 Global Wheat Detection](https://www.kaggle.com/rkuo2000/yolov5-global-wheat-detection)**<br>
train YOLOv5 for wheat detection
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/YOLOv5_GWD.jpg?raw=true)

---
**[EfficientDet Global Wheat Detection](https://www.kaggle.com/rkuo2000/efficientdet-gwd)**<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/EfficientDet_GWD.png?raw=true)

---
## Mask R-CNN
**Kaggle:** [rkuo2000/mask-rcnn](https://www.kaggle.com/rkuo2000/mask-rcnn)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Mask_RCNN_TF2.png?raw=true)

---
### Mask R-CNN transfer learning
**Kaggle:** [Mask RCNN transfer learning](https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155)
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Mask_RCNN_transfer_learning.png?raw=true)

---
### YOLOv5 + DeepSort
**Kaggle:** [YOLOv5 DeepSort](https://kaggle.com/rkuo2000/yolov5-deepsort)<br>
<iframe width="574" height="323" src="https://www.youtube.com/embed/-NHq7yUAY7U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="498" height="280" src="https://www.youtube.com/embed/RKVrtJs1ry8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Objectron
**Kaggle:** [rkuo2000/mediapipe-objectron](https://www.kaggle.com/rkuo2000/mediapipe-objectron)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Objectron_shoes.png?raw=true)

---
### OpenCV-Python play GTA5
**Ref.** [Reading game frames in Python with OpenCV - Python Plays GTA V](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)<br>
**Code:** [Sentdex/pygta5](https://github.com/Sentdex/pygta5)<br>
<iframe width="670" height="377" src="https://www.youtube.com/embed/VRsmPvu0xj0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Steel Defect Detection
**Dataset:** [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)<br>
![](https://diyago.github.io/images/kaggle-severstal/input_data.png)
**Kaggle:** [https://www.kaggle.com/code/jaysmit/u-net (Keras UNet)](https://www.kaggle.com/code/jaysmit/u-net)<br>

---
### PCB Defect Detection
**Dataset:** [HRIPCB dataset (dropbox)](https://www.dropbox.com/s/h0f39nyotddibsb/VOC_PCB.zip?dl=0)<br>
![](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-022-16302-3/MediaObjects/41598_2022_16302_Fig4_HTML.png?as=webp)

---
### Identify Military Vehicles in Satellite Imagery
**Blog:** [Identify Military Vehicles in Satellite Imagery with TensorFlow](https://python.plainenglish.io/identifying-military-vehicles-in-satellite-imagery-with-tensorflow-96015634129d)<br>
**Dataset:** [Moving and Stationary Target Acquisition and Recognition (MSTAR) Dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar)<br>
![](https://github.com/NateDiR/sar_target_recognition_deep_learning/raw/main/images/mstar_example.png)

---
### Pothole Detection
**Blog:** [Pothole Detection using YOLOv4](https://learnopencv.com/pothole-detection-using-yolov4-and-darknet/?ck_subscriber_id=638701084)<br>
**Code:** [yolov4_pothole_detection.ipynb](https://github.com/spmallick/learnopencv/blob/master/Pothole-Detection-using-YOLOv4-and-Darknet/jupyter_notebook/yolov4_pothole_detection.ipynb)<br>
**Kaggle:** [YOLOv7 Pothole Detection](https://www.kaggle.com/code/rkuo2000/yolov7-pothole-detection)
![](https://learnopencv.com/wp-content/uploads/2022/07/Pothole-Detection-using-YOLOv4-and-Darknet.gif)


* create .yaml for YOLO

```
%%writefile data/pothole.yaml
train: ../pothole_dataset/images/train 
val: ../pothole_dataset/images/valid
test: ../pothole_dataset/images/test

# Classes
nc: 1  # number of classes
names: ['pothole']  # class names
```

---
### Car Breaking Detection
**Code**: [YOLOv7 Braking Detection](https://github.com/ArmaanSinghSandhu/YOLOv7-Braking-Detection)<br>
![](https://github.com/ArmaanSinghSandhu/YOLOv7-Braking-Detection/raw/main/results/Detection.gif)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

