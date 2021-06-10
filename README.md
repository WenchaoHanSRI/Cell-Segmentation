# Cell-Segmentation
This project provides code for instance single-cell segmentation for the multiplexed images.

## Getting started
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

The code was also tested on Windows 10 based machine. Single core was used for training on windows machine. 

Details can refer:

https://github.com/matterport/Mask_RCNN/issues/93
