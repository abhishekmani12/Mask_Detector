# Realtime Mask Detector

A deep learning based Mask Detector which detects if a person is wearing a mask or not based on a video feed (primary input method).
- This project has undergone three major model iterations. The related notebooks and scripts for all three models     are uploaded to this repository.
- The third iteration (YOLO v5) was chosen to be deployed using Streamlit as it offered faster detections rates       than the other models.

## 1. Fastai Model - Two Stage Detection:

First iteration used the Fastai framework to create a baseline pretrained model - VGG_16 with transfer learning trained on a [publicly available kaggle dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) in addition to a variable learning rate. The images were converted to B/W for experimentation.

Frame handling and bounding Box placements were handled using OpenCV and Object Detection was powered by MTCNN.   
Deployment proved to be a challenge with the framework so a temporary colab notebook inference with javascript(to enable the local machine's webcam) was created.

## 2. Pytorch Model - Two Stage Detection:

Second Iteration used the Pytorch framework to create a pretrained model - ResNet50, trained through transfer learning in two ways:
  -   Resetting the FFC layer
  -   Fixed Feature Extraction - this method outperformed the first


Training was done using the [same dataset as before](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset). Extensive transformations were done to simulate the realtime noisy environment and to reduce the inherent racial and lighting bias present in the model. This was done in a trial error method.

OpenCV was used for framehandling and Bounding Boxes and Object Detection was powered by the DNN Module with its necessary weights file.   
A simple deployment with Streamlit was done which gave realtime frame results of 15-20 FPS.


## 3. YOLOv5 Pytorch - Single Stage Detection:

Third and final iteration used YOLOv5 with custom training(via Pytorch) using a dataset with the respective annotations which was obtained through a [repository fork](https://github.com/techzizou/yolov4-custom_Training). Transformations with Albumentations were performed on the dataset.

An extensive streamlit deployment was done which enables 4 different types of model inferencing to be performed:

###    Realtime inferencing using Webcam:

https://user-images.githubusercontent.com/76105443/213875523-426026ca-f123-4e9f-9d60-63afbe19d30e.mp4

###    Static Image Input with Annotated Image Output:
     
![Screenshot (803)](https://user-images.githubusercontent.com/76105443/213873492-2292aacb-5667-4449-b7d3-173bddedf038.png)

###    Static Image Input with CSV Output:

![Screenshot (789)](https://user-images.githubusercontent.com/76105443/213873701-31fb13fc-2ac8-411d-b76c-aaa14043d1d9.png)


###    Prerecorded Video Input with Annotated Video Output:

https://user-images.githubusercontent.com/76105443/213874990-2685407a-77fe-4022-bf30-5dd54a35d9ee.mp4


Output inference rate is around 30 FPS with 85% detection accuracy.

## Future Plans:

Deployment with ONNX and AWS Lambda. Might include an intermediate class as well to the dataset.
