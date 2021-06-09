# Cell-Segmentation
This project provides code for instance single-cell segmentation for the multiplexed images.


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
