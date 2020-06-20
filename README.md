# Learning-to-manipulate-individual-objects-in-an-image-Implementation
This repo contains the implementation of the method described in the paper

[Learning to Manipulate Individual Objects in an Image](https://arxiv.org/pdf/2004.05495.pdf)  

Published in the International Conference of Computer Vision and Pattern Recognition (CVPR) 2020.



### Introduction:
We describe a method to train a generative model with latent factors that are (approximately) independent and localized. This means that perturbing the latent variables affects only local regions of the synthesized image, corresponding to objects. Unlike other unsupervised generative models, ours enables object-centric manipulation, without requiring object-level annotations, or any form of annotation for that matter. For more details, please check our paper.

<p align="center">
<img src='doc/objects_room.gif' width = 400>
<img src='doc/multi_texture.gif' width = 240>
<p>
 
<p align="center">
<img src='doc/flying_animals.gif'>
<p>
 


## Running the code
### Prerequisites

This code was tested with the following packages. Note that other version of them might work but are untested.

* Ubuntu 16.04
* python3
* tensorflow-gpu==1.14.0
* python-gflags 3.1.2
* keras 2.3.1
* imageio 2.6.1
* numpy 1.17.2
* gitpython 3.0.5

### Datasets


#### Multi-dSprites and Objects Room

Download two existing datasets with the following commands:
```
mkdir data/multi_dsprites_data data/objects_room_data
wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords -P data/multi_dsprites_data
wget https://storage.googleapis.com/multi-object-datasets/objects_room/objects_room_train.tfrecords -P data/objects_room_data
```
These two datasets are TFRecords files and can be used without pre-processing.


#### Multi-Texture

The components are already included in [data/multi\_texture\_data](data/multi_texture_data) and will be automatically used to generate images online while training and testing.


#### Flying Animals

Please download the zip file from 
[this link](https://drive.google.com/open?id=1xs9CdR8HC_RxfuEbZnD_hmMqQusAuhbO), put it in [data/flying\_animals\_data](data/flying_animals_data) and then run the following commands to decode the raw images into .npz file.
```
cd  data/flying_animals_data
unzip data.zip
python fa_data_decode.py
```
These commands generate img_data.npz and img_data_test.npz in [data/flying\_animals\_data](data/flying_animals_data) for training and testing

### Training

To stabilize and speed up adversarial training, our training consists of three steps. Default hyperparameters settings for four datasets and three steps are included in [script/](script). Please modify some arguments, e.g. the path of output checkpoints, in scripts when necessary. 

#### 1.Pretrain inpainting network

Pretrain inpainting network on the task of predicting pixels on box-shaped occlusions.
```
sh script/dataset_name/pretrain_inpainter.sh
```
Pretrained checkpoints of inpainting network for each dataset can be downloaded [here](https://drive.google.com/drive/folders/1AcFb2kfFpEuD-Wi_Iz_Z9-mkgs3anEOF?usp=sharing). You can directly restore the downloaded checkpoint to skip this step.

#### 2.Spatial disentanglement

Update inpainting network and segmentation network adversarially for spatial disentanglement.

```
sh script/dataset_name/train_CIS.sh
```
Note that while for other datasets we train segmentation network from scratch, for flying animals dataset we suggest initializing ResNetV2-50 with checkpoint pretrained on ImageNet which can be found [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models). Please download the checkpoint by running
```
mkdir resnet && cd resnet
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
tar -xvf resnet_v2_50_2017_04_14.tar.gz
```

#### 3.Train VAE
```
sh script/dataset_name/train_VAE.sh
```
Train encoder and decoder to learn disentangled latent space.

### IoU Evaluation

For a trained model, you can measure its segmentation performance with the function [test\_segmentation.py](./test_segmentation.py). Examples of test script are provided as script/dataset_name/test_segmentation.sh. Edit it with the path to the checkpoint file and run it to compute the mean and standard variance of mean-IoU scores on 10 subsets.
```
sh script/dataset_name/test_segmentation.sh
```

### Disentanglement 

After finishing all training steps, you can visualize the latent space's disentanglement by feeding a target image into the model and varying one latent dimension at a time to see if the perturbation only leads to one type of semantic variation of one particular object in the synthesized image. 

Script examples for disentanglement visualization are provided as script/dataset_name/disentanglement.sh. Edit them with paths to checkpoint and output directories. 
```
sh script/dataset_name/disentanglement.sh
```
Modify some of the arguments when necessary to set which objects and dimensions to perturb and the range of varying latent factors.

### Perceptual cycle-consistency
We demonstrate the effectiveness of perceptual cycle-consistency constrain on Multi-Texture with each image including two objects of different identities, ellipse and square. Training scripts of the experiments are provided in [this folder](./script/multi_texture/perceptual_consistency). The first three training steps are the same as mentioned in [Training](./README.md#Training) without enforcing perceptual cycle-consistency. Then we finetune the model with perceptual cycle-consistency constrain by running
```
sh script/multi_texture/perceptual_consistency/finetune_PC.sh
```
It can be observed that the finetuning decreases identity switching rate and improves identity consistency. As shown in the figure below, finetuned model (middle) consistently captures the ellipse in channel 0 while un-finetuned model (right) can assign the square to channel 0 sometimes.

<p align="center">
<img src='doc/pc.gif'>
<p>
 
To compute identity switching rate of the segmentation network, run 
```
sh script/multi_texture/perceptual_consistency/test_segmentation.sh
```
We provide checkpoints for two models [here](https://drive.google.com/drive/folders/1WCBgnPim9l5aMjbgAg1wBETd3QoLY-Wd?usp=sharing). If you'd like to explore the effectiveness by yourself, we recommend downloading the [model](https://drive.google.com/drive/folders/1X5kDp-1swauBKaFXF32gRy9wnJEvtCKe?usp=sharing) that has been trained for the first three steps and restoring it to finetune with perceptual consistency.

## Downloads

You can download our trained models for all datasets [here](https://drive.google.com/drive/folders/1AcFb2kfFpEuD-Wi_Iz_Z9-mkgs3anEOF?usp=sharing) including pretrained inpainting networks and final checkpoints of all modules. 

## Citation

If you use this code in academic context, please cite the following publication:

```

@InProceedings{Yang_2020_CVPR,
  author = {Yang, Yanchao and Chen, Yutong and Soatto, Stefano},
  title = {Learning to Manipulate Individual Objects in an Image},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

