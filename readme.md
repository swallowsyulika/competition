# Prerequisites
## Install Dependencies
* python >= 3.8
* [python-opencv](https://pypi.org/project/opencv-python/)
* [numpy](https://numpy.org/)
* [pytorch](https://pytorch.org/get-started/locally/) >= 1.9
* [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
* [albumentations](https://github.com/albumentations-team/albumentations)
* [pyqt5](https://pypi.org/project/PyQt5/)
* [tqdm](https://github.com/tqdm/tqdm)
## Prepare Yolo v4 detector
```console
$ git clone https://github.com/AlexeyAB/darknet
$ cd darknet
```
Edit Makefile to enable CUDA, CUDNN, OPENCV.
```console
$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```
Modify `darknet_executable` and `yolo_pretrained_weight` in `config.py`
# Glossaries
* **Container**: A bonding box that contains one or more Chinese characters.

# Training

## 1. Draw bonding boxes and labels into the images (optional)
```console
$ python -m convert.train.draw_bb
```
## 2. Crop characters from images (to train cleaner model)
```console
$ python -m convert.train.crop_characters
```
## 3. Crop containers from images (to train detection model)
```console
$ python -m convert.train.crop_containers
```
## 4. Train the image cleaner.
```console
$ python -m gan.train
```
## 5. Train the object detection model.
```console
$ python -m detection.train
```
## 6. Train the cleaner model.
```console
$ python -m cleaner.train
```
## 7. Train the character recognition model.
```console
$ python -m recognition.create_available_glyph_cache
$ python -m recognition.train
```
Note: `glyph_cache` only needs to be created once, and it could take a while depending on the size of your font library.

# Inference

## 1. Draw containers to the images (optional)
```console
$ python -m convert.eval.draw_containers
```
## 2. Crop containers from images with four point transform.
```console
$ python -m convert.eval.crop_containers
```
## 3. Use detection model to detect the locations of the characters in images.
```console
$ python -m detection.eval
```
## 4. Draw detection result onto images (optional)
```console
$ python -m detection.draw_eval_result
```
## 5. Crop detection result (characters) from container images
```console
$ python -m detection.crop_eval_result
```
## 6. Clean characters with cleaner model
```console
$ python -m cleaner.eval --set train --checkpoint e_16
$ python -m cleaner.eval --set eval --checkpoint e_16
```
## 6. Recognize characters with recognition model
```console
$ python -m recognition.eval_train
$ python -m recognition.eval
```


# Asset sources
## Fonts
### Cleaner
* [Noto Fonts CJK](https://www.google.com/get/noto/help/cjk/)
* [清松手寫體](https://www.facebook.com/groups/549661292148791/)
* [jf open 粉圓字型](https://justfont.com/huninn/)
* [台北黑體](https://sites.google.com/view/jtfoundry/zh-tw)
* [全字庫正楷體](https://data.gov.tw/dataset/5961)
* [全字庫正宋體](https://data.gov.tw/dataset/5961)
* [源泉圓體](https://github.com/ButTaiwan/gensen-font)
### Recognizer
* [chinesefontdesign.com](https://chinesefontdesign.com/)
## Textures
* [opengameart.org](https://opengameart.org/textures/)
