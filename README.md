# DOGNet
Deformable Offset Gating Network for JPEG Artifact Reduction for a Wide Compression Quality Factors


## Prerequisites
* python 3.7
* pytorch 1.13

## Dataset Preparation
Download Training Dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and Flickr2K.

You should have following directory structure:
```
dataset
    |-- DIV2K_train
    |-- Flickr2K
        |-- 000001.png
        |-- ...
    |-- ...
```

## Training
* Run train.py

```
--gpu : GPU Index. If you want to use mutliple GPUs, use --mgpu True
--exp_name : Name of the experiment
--train_dataset : Name of the dataset

python train.py --gpu [GPU INDEX] --exp_name [EXP_NAME] --train_dataset [Datatset Name]

For multiple GPUs

python train.py --gpu [GPU INDEXs] --mgpu True --exp_name [EXP_NAME] --train_dataset [Datatset Name]
```