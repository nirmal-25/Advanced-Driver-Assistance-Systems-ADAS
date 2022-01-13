# Advanced Driver Assistance System (ADAS)


An ADAS pipeline that consists of Traffic Light, Traffic Sign detection and anterior car distance estimation is developed in this project. A deep learning approach is used to develop a robust object detection model. Different algorithms and deep learning frameworks are experimented with, to provide a detailed analysis as to which model performs the best for the particular task at hand.<br/><br/>
The repo consists of scripts that help visualize detections from models that are trained jointly on different datasets - the Bosch Small Traffic Light Dataset and the Tsinghua-Tencent 100K Traffic Sign dataset. The model has been tested on video sequences and achieves a good frame rate with detections from 50 classes which consist of traffic lights, traffic signs and cars. More details about the project can be found in the project_report.pdf file in the materials folder. <br/><br/>
This project is based on [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). However, note that the project was originally implemented using TensorFlow 1.14 and not TensorFlow 2.x
![bosch_dataset (2)](https://user-images.githubusercontent.com/51696913/149266584-f9a6c755-2483-48fc-bf8c-2fc1deeddc84.gif)
