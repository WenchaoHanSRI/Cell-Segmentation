# Cell-Segmentation
This project provides code for instance single-cell segmentation for the multiplexed images.


## Environment setup
The code was tested on a linux machine with python 3.6 and packages that specified in the requirement.txt installed.

To use GPU for training and inferencing, cudatookit=10.0.130 and cudnn=7.6.4 needs to be installed.

Setup a conda environment example:
```
conda create -n testenv python=3.6
```
```
conda install cudatoolkit=10.0.130
```
```
conda install cudnn=7.6.4
```

