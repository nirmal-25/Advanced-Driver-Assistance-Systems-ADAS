# Advanced Driver Assistance System (ADAS)


An ADAS pipeline that consists of Traffic Light, Traffic Sign detection and anterior car distance estimation is developed in this project. A deep learning approach is used to develop a robust object detection model. Different algorithms and deep learning frameworks are experimented with, to provide a detailed analysis as to which model performs the best for the particular task at hand.<br/><br/>
The repo consists of scripts that help visualize detections from models that are trained jointly on different datasets - the [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) and the [Tsinghua-Tencent 100K Traffic Sign dataset](https://cg.cs.tsinghua.edu.cn/traffic-sign/). The model has been tested on video sequences and achieves a good frame rate with detections from 50 classes which consist of traffic lights, traffic signs and cars. More details about the project can be found in the project_report.pdf file in the materials folder. <br/><br/>
This project is based on [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). However, note that the project was originally implemented using TensorFlow 1.14 and not TensorFlow 2.x <br/><br/>


![joint_output_1](https://user-images.githubusercontent.com/51696913/149267243-04e9d0ce-2851-4c72-892a-9f7c2336b0e2.png)![joint_output_1_432x428](https://user-images.githubusercontent.com/51696913/149267249-52e5a3a6-7045-46d8-b4e2-87f391c90dcd.png)
