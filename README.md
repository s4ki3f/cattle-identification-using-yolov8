# Cattle-Identifiction
YOLOv8 object detection and classification in cattle identification
Cattle Muzzle Identification, may need both object detection and classification both or, may be done alone with classification

<br>
<div>
  <a href="https://colab.research.google.com/github/s4ki3f/yolo/blob/main/notebooks/train-yolov8-object-classification-on-custom-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>
<br>

At First checking the Grapichs information, they are connected or not,
```bash
!nvidia-smi
```
Then It is better to set HOME as current directory by using 
```python
import os
HOME = os.getcwd()
print(HOME)
```
After that, using pip, will install YOLOv8
```bash
!pip install ultralytics
from IPython import display
display.clear_output()
```
Or, can use git
```bash
# Git clone method (for development)
!git clone https://github.com/ultralytics/ultralytics
%pip install -qe ultralytics
```

To check the whereas we are using cpu or gpu, we can run this command
```bash
!yolo mode=checks
```

Now importing YOLO
```python
from ultralytics import YOLO

from IPython.display import display, Image
```
If there is no local data, we can look for public data set like roboflow, MSCOCO etc.

# Downloading DATA

```bash
%cd {HOME}

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="############")
project = rf.workspace("tdc").project("cow-identification")
dataset = project.version(1).download("folder")
```
This dataset is already segmented in three parts, Train, Test, Validation. This, reduce the time for dataset spliting into three.

#Training Model

We are using 3 epochs for the first run for test purpose,
```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='/content/Cow-Identification-1', epochs=3)  # train the model
model('/content/Cow-Identification-1/test/cattle_2700/cattle_2700_DSCF1273_jpg.rf.18e7015665545779572565fd6002ce77.jpg')  # predict on an image
```
If it is a smooth run, then We wil go for 300-600 epoch for a well learnt model.

```bash
!yolo task=classify mode=train model=yolov8n-cls.pt data='/content/Cow-Identification-1' epochs=300
```
Generated trained files will be saved in runs/classify/train under HOME dir
```bash
!ls {HOME}/runs/classify/train
```

#Validation

After training the model, it is time for validating the dataset.
we can run this for validating
```bash
!yolo task=classify mode=val model=/content/runs/classify/train8/weights/best.pt data='/content/Cow-Identification-1'
```
#Test and Prediction
picking up random image from test or validate we will conduct the prediction

```bash
!yolo task=classify mode=predict model='/content/runs/classify/train7/weights/best.pt' conf=0.25 source='/content/Cow-Identification-1/test/cattle_2700/cattle_2700_DSCF1273_jpg.rf.18e7015665545779572565fd6002ce77.jpg'
```

results will be saved under this,
```bash
!ls {HOME}/runs/classify/predict
```

