# Face Recogntion System

## Description

This project showcases a directory based face-recogntion system, which uses K-Nearest Neighbor Classification technique to classify or detect a face through a camera/webcam feed.

## Table of Contents

- Prerequisites
- Installation
- Usage
- Licence

## Prerequisites
 
 Require modules to be imported:
 
 - face_recogntion
 - OpenCV
 - Pickle
 - Sklearn
 - Math

## Installation

### Requirements
- Python 3.3+ or Python 2.7
- macOS or Linux


#### Installing on Mac or Linux

Install dlib if not installed already:

[How to install dlib from source on macOS or Ubuntu](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)

Then, install the face_recogntion module from pypi using pip3 (or pip2 for Python 2):

```
pip install face_recognition
```
Also, install install OpenCV if not installed already

[How to install OpenCV from source on macOS or Ubuntu](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

## Usage

For running face recognition on live video from your web camera "cv2.VideoCapture(0)",(0) indicates web camera is chosen.
Note: If you want to choose a specific camera change the index to 1 or 2.

This project has two parts:
1. Function - train_kNN
2. Face_Recogntion Script

### Process:

1. Call the train_kNN() script.

Note: It is assumed that you have "People_Directory" in the same folder, you can name it something else, considering you also change the train_dir name as well.
It is also important to note that you need to call the train_kNN function just once, after which you may(or may not)comment it out.

2. Face_Recogntion Script :

This script loads up the learned kNN-classifier model and computes distance metric on each frame. Once the closest distance results are obtained it displays the result as a name of the person along with a bounding box.

## License

This project is licensed under the MIT License - see the LICENSE file for details
