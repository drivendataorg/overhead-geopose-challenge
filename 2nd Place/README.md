# Overhead Geopose Challenge - Second Place Solution

This repository contains source code and pre-trained models for the second place solution of the [Overhead Geopose Challenge](https://www.drivendata.org/competitions/78/overhead-geopose-challenge/leaderboard/).

## Installation

Installation should be pretty straightforward by installing python dependencies from requirements file:

`pip install -r requirements.txt`

I've used PyTorch 1.9, but older previous of PyTorch should be compatible as well. The `requirements.txt` file 
contains all dependencies you may need for training or making predictions. 


## Preprocessing

After installing required packages, there is one important step to make - preprocess train & test data. Specifically,
you need to run `python convert_j2k.py` script in order to convert j2k files to PNG format. I found that decoding of 
j2k format in Pillow is much slower than reading data as PNG using OpenCV library. 

To convert the dataset, simply run the following script:

```bash
export DRIVENDATA_OVERHEAD_GEOPOSE="/path/to/dataset"
python convert_j2k.py
# OR
python convert_j2k.py --data_dir="/path/to/dataset"
```

Instead of setting an environment variable, one can use command-line argument `--data_dir /path/to/dataset` instead.

## Reproducing solution

I've included top 2 best submissions based on the private LB. 

Model checkpoints and inference configuration is set by configuration files:

     - `configs/inference/fp32_0.8910_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold89_bo1_d4.yaml` 2 models, D4 TTA (0.8910 private, 0.8896 public)
     - `configs/inference/fp32_0.8914_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold9_bo3_d4.yaml` 9 models, D4 TTA (0.8914 private, 0.8900 public)

All model checkpoints hosted on the GitHub and will be downloaded automatically upon the first run of submission script.

### Single-GPU inference mode 

To generate a submission archive, one can run the following script:

```bash
export DRIVENDATA_OVERHEAD_GEOPOSE="/path/to/dataset"
python submit.py configs/inference/fp32_0.8910_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold89_bo1_d4.yaml
python submit.py configs/inference/fp32_0.8914_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold9_bo3_d4.yaml
# OR
python submit.py configs/inference/fp32_0.8914_b6_rdtsc_unet_fold9_b7_unet_fold0_d4_bo1.yaml --data-dir "/path/to/dataset"
```

Submission archive can be later found in corresponding folders (e.g `submissions/fp32_b6_rdtsc_unet_fold9_b7_unet_fold0_d4_bo1`).

### Multi-GPU inference mode 

Inference time can be greatly reduced in case machine has multiple GPUs. In this case we can distribute the predictions across multiple GPUs and generate them N times faster.
The only change that is required - to call `submit_ddp.py` instead of `submit.py`. 

```bash
export DRIVENDATA_OVERHEAD_GEOPOSE="/path/to/dataset"
python submit_ddp.py configs/inference/fp32_0.8910_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold89_bo1_d4.yaml
python submit_ddp.py configs/inference/fp32_0.8914_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold9_bo3_d4.yaml
# OR
python submit_ddp.py configs/inference/fp32_0.8914_b6_rdtsc_unet_fold9_b7_unet_fold0_d4_bo1.yaml --data-dir "/path/to/dataset"
```

### Inference time

A GPU with 24 GPU is sufficient to generate submission for all ensembles. 

Inference time of the ensembles (Measured on RTX 3090)

| Config                                                     | Inference Time | Sec/Image | Num Models |
| ---------------------------------------------------------- | -------------- | --------- | ---------- |
| fp32_0.8914_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold9_bo3_d4  | 7h             | 26s/img   | 9          |
| fp32_0.8910_b7_fold0_b7_rdtsc_fold1_b6_rdtsc_fold89_bo1_d4 | 3h             | 11s/img   | 4          |

Disabling D4 TTA or switching to D2 would increase inference speed by factor of 8 and 2 accordingly at a price of slightly lower R2 score.

## Training model from scratch

It's also possible to reproduce the training from a scratch. 
For convenience, I've included train scripts to repeat my training protocols. Please note, this is VERY time-consuming and GPU-demanding process.
I've used setup of 2x3090 NVidia GPUs each with 24Gb of RAM and was able to fit very small batch sizes. In addition, I've observed that mixed-precision
training had some issues with this challenge, and I had to switch to fp32 mode, which decreased batch size even more.

Training of each model takes ~3 full days on 2x3090 setup. Training could be started by sequentially running following scripts:

    - scripts/train_b6_rdtsc_unet_fold9.sh
    - scripts/train_b7_unet_fold0.sh

For different setups you may want to tune batch size and number of nodes to match the number of GPUs on the target machine.

### Adjusting train script for your needs

Let's break down the script itself to see how you can hack it. 
This training pipeline uses [hydra](https://github.com/facebookresearch/hydra) framework to configure experiment config.
With help of hydra you can assemble a complete configuration from separate blocks (model config, loss config, etc.).
In the example below, the model is `b6_rdtsc_unet` (See `configs/model/b6_rdtsc_unet.yaml`), 
optimizer - is AdamW with FP16 disabled (see `configs/optimizer/adamw_fp32.yaml`) and loss function 
defined in file `configs/loss/huber_cos.yaml`.

```bash
# scripts/train_b6_rdtsc_unet_fold9.sh
export DRIVENDATA_OVERHEAD_GEOPOSE="CHANGE_THIS_VALUE_TO_THE_LOCATION_OF_TRAIN_DATA"
export OMP_NUM_THREADS=8

python -m torch.distributed.launch --nproc_per_node=2 train.py\
  model=b6_rdtsc_unet\
  optimizer=adamw_fp32\
  scheduler=plateau\
  loss=huber_cos\
  train=light_768\
  train.epochs=300\
  train.early_stopping=50\
  train.show=False\
  train.loaders.train.num_workers=8\
  batch_size=3\
  dataset.fold=9\
  seed=777
```



## References

- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Catalyst](https://github.com/catalyst-team/catalyst)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [TIMM](https://github.com/rwightman/pytorch-image-models)
- [hydra](https://github.com/facebookresearch/hydra)
