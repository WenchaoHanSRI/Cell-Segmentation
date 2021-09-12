# Cell-Segmentation
This project provides code for instance single-cell segmentation for the multiplexed images.

## Getting started
Trained weights for the three testing datasets can be downloaded:
https://drive.google.com/drive/folders/1fqgDMQCTEvkSNKR37GJTYXzyXHdl0RAm?usp=sharing
The downloaded folder "Trained_Weights" need to put under the root directory.

Using trained model for segmentation with our example images:

```
python cellseg2finetune.py detect --yaml=val.yaml
```

Validating the trained model to manual annotations:
```
python cellseg2finetune.py val --yaml=val.yaml
```

Train the model using published dataset:
```
python cellseg2finetune.py train --yaml=train.yaml
```

The exteranl dataset has much larger imgae size, thus used multiprocssing for computing validation results, 
the code utilize all the available cores. The code below is recommened when the computing resource has more that 4 CPU core to use. Much more cores (e.g. 48 or more) are recommened for efficiency.
For speed, exteranl dataset validation:
```
python DetectValFL2048_edit.py --yaml=ExternalVal.yaml
```


## Environment setup
The code was tested on a linux machine with python 3.6 and packages that specified in the requirement.txt.

Setup example using a conda environment:
```
conda create -n testenv python=3.6
```
```
pip install -r requirement.txt
```

To use GPU for training and inferencing, cudatookit=10.0.130 and cudnn=7.6.4 needs to be installed:
```
conda install cudatoolkit=10.0.130
```
```
conda install cudnn=7.6.4
```

The setup is not limit to using conda environment as long as the packages and dependencies are properly installed in the environment.

The code was also tested on Windows 10 based machine. 

Training issue using Window machine is adapted based on the report:

https://github.com/matterport/Mask_RCNN/issues/93
