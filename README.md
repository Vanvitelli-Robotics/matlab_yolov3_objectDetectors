# Object Detection and Recognition using Yolov3
The repository contains a MATLAB implementation of a multi-class object detection task via the real-time _Yolov3_ technique.
The goal is to recognize within an image one or multiple object detecting also their coarse orientation among a set of six possibilities.
This task has to be intended as the first necessary step for an intelligent system in order to understand the best way to grasp the object of interest, in the context of shelf replenishment task.
The used network is the _darknet53_ pre-trained on the _COCO_ dataset. The network has been fine-tuned considering a custom dataset made of images containing typical retail items in various orientations.
