# Geopose Challenge 1st place solution


## Overview

The solution is mostly based on the [CVPR EarthVision 2021 paper](https://arxiv.org/abs/2105.08229) and the [CVPR 2020 main conference paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Christie_Learning_Geocentric_Object_Pose_in_Oblique_Monocular_Images_CVPR_2020_paper.pdf):


## Dependencies

The system is fully dockerized, hence it requires docker with nvidia runtime support
All requirements are specified in the Dockerfile.
The docker image is based on the latest NGC Pytorch container which is optimized to work with new generation of Nvidia cards.
 
## Improvements over provided baseline

While meta-architecture is mostly the same as provided baseline there are crucial differences that allow to increase score by 10%.

- A very powerful encoder EfficientNet V2 L which has huge capacity,  receptive field and in addition to that it is less prone to overfitting. 
- UNet like encoder with more filters and additional convolution blocks for better handling of fin grained details
- improved train loop that uses mixed precision, random crops, distributed training
- loss functions that directly optimize R2 metric
- validation that computes final metric and used to select best checkpoints

## Building and running docker image
To build an image simply run the following command 
```
 docker build -t geopose .
```
Under the hood it will 
- pull docker image from NGC, 
- install additional libraries like gdal, pygdal, timm, albumentations
- download trained models to 'weights' in container 
- cythonize vflow code

To run docker container and go to  container's  bash terminal
```
 docker run --network=host --gpus all --ipc host -v /<host_dir> :/<container_dir> -it geopose
 # use host dir with the train or test data, e.g.
 docker run --network=host --gpus all --ipc host -v /mnt/datasets/geopose:/data -it geopose
```
## Training

I used a workstation with 4xRTX A6000 during training. With some changes it can be easily trained on 2x RTX 3090 cards.
A bash script that runs all steps required to produce two fully trained models
```shell
    ./train.sh <data_dir> <w_data_dir> <num_gpus>
    # data dir is dir inside container that contains train dataset (mounted during docker run)
    # w_data_dir is used to save converted tif files for training
    # num_gpus - number of gpus used for training
``` 
An example 
```shell
./train.sh /data/train /data/train_tiffs 4 
```

Under the hood it 
- converts AGL to meters, similar to baseline approach but without downsampling. The images will be saved to `/data/train_tiffs`
- trains 100 epochs of first train val split (fold) a smaller crop size (896x896)
- fine tunes 75 epochs with larger crop size
- repeats this procedure for another fold
Text logs and tensorboard logs are stored under `logs/` dir in container's working directory

Parameters used for training can be found in `dist_train.sh` and `dist_train_tune.sh` scripts

## Testing

To run prediction build and run docker image as described above.

`test.sh` script will run prediction  and generate `submission_compressed.tar.gz` in working directory. 

`./test.sh <test_dir> <num_gpus>`


Assuming that challenge's dataset is mounted to `/data` directory and test images are extracted to `/data/test` and there are 2 gpus, the command is
`./test.sh /data/test 2`. Prediction works conventional GPUs like 1080 Ti and requires around 10g of GPU memory. 

Output file is around 5gb, it could be copied to mounted host file system.