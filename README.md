# Advanced Driver Assistance System (ADAS)


An ADAS pipeline that consists of Traffic Light, Traffic Sign detection and anterior car distance estimation is developed in this project. A deep learning approach is used to develop a robust object detection model. Different algorithms and deep learning frameworks are experimented with, to provide a detailed analysis as to which model performs the best for the particular task at hand.<br/><br/>
The repo consists of scripts that help visualize detections from models that are trained jointly on different datasets - the [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) and the [Tsinghua-Tencent 100K Traffic Sign dataset](https://cg.cs.tsinghua.edu.cn/traffic-sign/). The model has been tested on video sequences and achieves a good frame rate with detections from 50 classes which consist of traffic lights, traffic signs and cars. More details about the project can be found in the project_report.pdf file in the materials folder. <br/><br/>
This project is based on [TensorFlow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). However, note that the project was originally implemented using TensorFlow 1.14 and not TensorFlow 2.x <br/><br/>


<p align="center">
![joint_output_1_432x428](https://user-images.githubusercontent.com/51696913/149267272-55b997f1-ac5c-430f-84a3-975e33e008c2.png)   ![joint_output_4](https://user-images.githubusercontent.com/51696913/149267277-22505d0f-4ce3-49ba-8c34-1fc4e804fcdd.png)

</p>
<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/51696913/149267272-55b997f1-ac5c-430f-84a3-975e33e008c2.png" alt="Material Bread logo">
</p>
