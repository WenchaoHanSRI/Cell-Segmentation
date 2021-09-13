# Cell-Segmentation
This project provides code for instance single-cell segmentation for the multiplexed images.

## Getting started
Trained weights for the three testing datasets can be downloaded:
https://drive.google.com/drive/folders/1fqgDMQCTEvkSNKR37GJTYXzyXHdl0RAm?usp=sharing

The downloaded folder "Trained_Weights" need to put under the root directory.

Using trained model for segmentation with our example images of ovarian cancer dataset:

```
python cellseg2finetune.py detect --yaml=val.yaml
```

Validating the trained model to manual annotations:
```
python cellseg2finetune.py val --yaml=val.yaml
```
Users may change the val.yaml file to perform inferencing/validation for example images of breast cancer dataset.

Train the model using published dataset:

Training data will be publicly available when the paper is published.
```
python cellseg2finetune.py train --yaml=train.yaml
```

For external dataset:

Testing sample data can be downloaded via: https://drive.google.com/drive/folders/17zJ92-qxMwi15Fpa0lc9At6deYh604A2?usp=sharing. The raw dataset is obtained from https://warwick.ac.uk/fac/sci/dcs/research/tia/data/micronet. We made the following changes from the raw dataset: 1) we created the color images from the raw dataset, 2) we expanded the original contours to allow touching contours. 

The downloaded data folder needs to put under the directory of TestData along with the other two datasets.

Since the exteranl dataset has much larger imgae size, we used multiprocssing for computing validation results, 
the code utilize all the available cores. The code below is recommened when the computing resource has more that 4 CPU core to use. Much more cores (e.g. 48 or more) are recommened for efficiency.
First, perform the segmentation:
```
python cellseg2finetune.py detect --yaml=ExternaDetc.yaml
```
Second, validating the segmentation results against manual annotations:
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

## Citation
```
@misc{DeepCSeg_2021,
  title={DeepCSeg, a CellProfiler plug-in for whole cell segmentation for immunofluorescence multiplexed images},
  author={Wenchao Han},
  year={2021},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/WenchaoHanSRI/DeepCSeg}},
}
```

