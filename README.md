# Advanced Driver Assistance System (ADAS)


An ADAS pipeline that consists of Traffic Light, Traffic Sign detection and anterior car distance estimation is developed in this project. A deep learning approach is used to develop a robust object detection model. Different algorithms and deep learning frameworks are experimented with, to provide a detailed analysis as to which model performs the best for the particular task at hand.<br/><br/>
The repo consists of scripts that help visualize detections from models that are trained jointly on different datasets - the Bosch Small Traffic Lights Dataset (https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) and the Tsinghua-Tencent 100K Traffic Sign dataset (https://cg.cs.tsinghua.edu.cn/traffic-sign/). The model has been tested on video sequences and achieves a good frame rate with detections from 50 classes which consist of traffic lights, traffic signs and cars. More details about the project can be found in the project_report.pdf file in the materials folder. <br/><br/>
This project is based on [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). However, note that the project was originally implemented using TensorFlow 1.14 and not TensorFlow 2.x <br/><br/>


![Sample video detection output](https://user-images.githubusercontent.com/51696913/149266584-f9a6c755-2483-48fc-bf8c-2fc1deeddc84.gif)![joint_output_4](https://user-images.githubusercontent.com/51696913/149266841-8f476db5-7251-437e-a590-8d70eb7e7529.png)![joint_output_1](https://user-images.githubusercontent.com/51696913/149266845-d2bc8750-6a33-4ae6-8c76-6e094632d1b6.png)
