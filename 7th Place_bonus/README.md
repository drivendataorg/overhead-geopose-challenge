7th place solution
==================

"Overhead Geopose Challenge"
https://www.drivendata.org/competitions/78/overhead-geopose-challenge/

Author: Igor Ivanov

Email: vecxoz@gmail.com

License: MIT

User: vecxoz


SOLUTION SUMMARY
================

I developed a solution that reached 7th place in the private leaderboard 
with a score of 0.8518 which is 0.05 improvement over the baseline solution. 
First, I implemented training with automatic mixed precision in order 
to speed up training and facilitate experiments with the large architectures. 
Second, I implemented 7 popular decoder architectures and conducted extensive 
preliminary research of different combinations of encoders and decoders. 
For the most promising combinations I ran long training for at least 200 epochs 
to study best possible scores and training dynamics. Third, I implemented 
an ensemble using weighted average for height and scale target and 
circular average for angle target. These approaches helped to substantially 
improve baseline solution.


IMPLEMENTATION NOTES
====================

**Modifications of the baseline**

Baseline repository provides a great framework for the task. 
I implemented several new features: automatic mixed precision, 
gradient accumulation, and 2 new decoders. In order to make it easy 
to identify my modifications made in the codebase
I included full git repository "solution/monocular-geocentric-pose" where
all changes are included in a single commit. 
2 files were modified and 2 new files were added. 
New files contain definitions of DeepLabV3 and FPN decoders.
I did not use them in the final submission but reported their results in write-up,
so I included the code (and weights) for reference. It's convenient to use standard git diff
to see my modifications but I also included comments with "mod:" tag in many places.
It's important to note that in latest release of "segmentation-models-pytorch" (0.2.0) 
some interfaces were slightly changed and my decoder implementations 
require previous version (0.1.3)

modified:
- main.py
- utilities/ml_utils.py

new:
- utilities/deeplabv3_vflow.py
- utilities/fpn_vflow.py

**Command line arguments**

Another note is that for any command line arguments which semantically are bool
I used int type and two explicit choices (0, 1). 
Conversion into actual bool is made in-place (if needed) inside ml_utils.py.
The reason is that command line arguments by nature are strings and argparse package
does not implement appropriate conversion.

**Learning rate**

In general my experiments showed that 0.0001 value for the learning rate was 
a very good choice which worked well for almost all models. 
But for some larger encoders (e.g. Inception v4) it may be better to set 
the learning rate a bit lower. Specifically I encountered the case where 
the Inception model was not learning with value 0.0001 in one of the folds. 
I restarted it with a learning rate of 0.00005 and the training process became stable. 
Four other folds were trained successfully with the original 0.0001 learning rate.

**Automatic mixed precision**

AMP is used in training and validation (after epoch) only. 
Inference (test) is always in full precision. 

**XGBoost**

XGBoost model for scale target is trained using original (not downsampled) data.


HARDWARE
========

**Training**

Models were trained on the 5 separate machines in parallel i.e. single-GPU machine per fold.
Each machine with the following specs:

4 cores, 28 GB RAM, T4 GPU, 1 TB SSD

Training time: 1900 hours (for the whole ensemble)

**Inference**

Machine of the same configuration as for training:

4 cores, 28 GB RAM, T4 GPU, 1 TB SSD

Inference time: 6 hours (for the whole ensemble)


SOFTWARE
========

- Ubuntu 18.04
- Python 3.6.9
- CUDA 11.1
- cuDNN 8.0.4
- GDAL 2.4.2

Detailed package list is supplied in "requirements.txt" file.
Please install GDAL according to your system config. It is not specified in "requirements.txt".


All scripts use absolute paths with the HOME variable.
By default "solution" directory is expected to be extracted as: `$HOME/solution/`

You can set the HOME variable to point to any other place where the solution was extracted.


DIRECTORY STRUCTURE
===================
```
solution/
    data/                               All data will be downloaded here
    models/                             This directory will be downloaded and extracted in the process of inference preparation
    monocular-geocentric-pose/          Full baseline git repository with a single commit includiong all my modifications
    output/                             Predictions are stored here
    training_scripts/                   .sh scripts with training commands including training hyperparameters
    ensemble_and_compress.py            Script to run ensembling and compression (according to submission format)
    LICENSE.txt                         License
    prepare_inference.sh                Script to download and extract test data and download trained models
    prepare_training.sh                 Script to download and extract training data, and 2x downsample images 
    README.md                           This file
    requirements.txt                    Requirements
    run_inference.sh                    Script to run all inference steps and create "submission.tgz"
    run_training_model_0_fold_0.sh      Script to run training of a single model on a single machine / single fold
    run_training_xgb.sh                 Script to train XGBoost model
    split_data.py                       Script to split data into training and validation
    xgb_scale_inference.py              Script to run inference of scale target using XGBoost models
    xgb_scale_training.py               Script to train XGBoost models for scale target
```


RUN INFERENCE
=============

```
cd $HOME/solution

# 20 min
# Links to .csv files should be updated

bash prepare_inference.sh

# 6 hours (T4 GPU)
# "submission.tgz" should appear in the "output" dir

bash run_inference.sh
```


RUN TRAINING
============

My training setup was based on 5 separate machines (machine per fold).
Specifically I downloaded data, downsampled it, and made a disk snapshot.
Than I instantiated 5 virtual machines from this snapshot and on each machine 
I split data in train/val according to the current fold i.e. just moved 
images from train dir to separate val dir. In this way I utilized the data 
pipeline provided in the baseline model without modifications.

Another possible setup is to use a single machine with several GPUs.
But this approach will require to modify data pipeline (to select required images)
or to make 5 copies of data. Also this approach may be suboptimal 
due to massive amounts of hard drive read operations.

```
# This command should be run on each machine or on initial machine with a consequent snapshot as described above

bash prepare_training.sh

# This will train architecture "unet-resnet34" on the machine-0 / fold-0

bash run_training_model_0_fold_0.sh
```

To train another fold or architecture you just need to copy the script and change values 
of the corresponding shell variables inside the script

FOLD_ID: 
- 0, 1, 2, 3, 4

MODEL_DIR:
- run-20210608-2131-unet-resnet34-e500
- run-20210626-0051-unet-inceptionv4-e290
- run-20210704-1650-unet-seresnext50-e120
- run-20210713-0048-unet-regnetx064-e80
- run-20210713-0107-unet-resnext50swsl-e230

```
# The complete training process would be the following:
# (Checkpoints are saved in corresponding directories inside "training_scripts" directory)

bash run_training_model_0_fold_0.sh
# ...
bash run_training_model_4_fold_4.sh
bash run_training_xgb.sh
```


END
===
