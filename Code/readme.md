
## Face Detection Annotations
First, convert the TFW dataset to a yolo format using the `dataset2yolo.ipynb` notebook.

Then, follow these steps to train the [YOLOv5](https://github.com/ultralytics/yolov5) models on the TFW dataset: 
1. Clone the repository from Github and install necessary packages:
```
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```
2. Copy our `yolov5_tfw.yaml` file into `/yolov5/data` and update paths to the training and validation sets.

To inference and get face annotations clone the following repository: [YOLO5Face](https://github.com/deepcam-cn/yolov5-face) models on the ThermVision dataset:
1. Copy the repository from Github and install necessary packages:
```
$ git clone https://github.com/deepcam-cn/yolov5-face.git
$ cd yolov5-face
```

2. Copy our `yolov5_tfw.yaml` file into `/yolov5-face/data` and update paths to the training and validation sets.

## Pre-trained YOLO5Face thermal face detection models
| Model  | Backbone | c-indoor<br>AP<sub>50 | u-outdoor<br>AP<sub>50 | Speed (ms)<br>V100 b1|Params (M)|Flops (G)<br>@512x384|
|  ---:| :---: | :---: | :---: | :---: | :---: | :---: | 

| [YOLOv5s](https://drive.google.com/file/d/1IdsdR1-qUeRo5EKQJzGQmRDi2SrMXJG5/view?usp=sharing) | CSPNet  | 100  | 96.82 | 7.20  | 7.05  | 3.91 |  
| [YOLOv5s6](https://drive.google.com/file/d/1YZX3t7cSPnWWoic7oJo86ljBQgE5PPb2/view?usp=sharing)| CSPNet  | 100  | 96.83 | 9.05  | 12.31 | 3.88 |  
  
To use pre-trained `YOLO5Face` models:
  1. Download pre-trained models from [Google Drive](https://drive.google.com/drive/folders/12ub57wP1hZ4tL2WH7TrUpmbvXXIdi3NU?usp=sharing) and unzip them in the `yolov5-face` repository folder.
  2. Run `detect_face.py` on terminal:
   ```
  python detect_face.py --weights PATH_TO_MODEL --image PATH_TO_IMAGE --img-size 800
   ```
  3. The result is saved as `result.jpg` in `./yolov5-face/`


## This work is based on the following paper. Full credit for the models and inference scripts goes to the original authors:
  ```
@ARTICLE{9781417,
  author={Kuzdeuov, Askat and Aubakirova, Dana and Koishigarina, Darina and Varol, Huseyin Atakan},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={TFW: Annotated Thermal Faces in the Wild Dataset}, 
  year={2022},
  volume={17},
  number={},
  pages={2084-2094},
  doi={10.1109/TIFS.2022.3177949}}
  ```
