## Geopose 5th place solution

Username: chuchu

## Content of the code

- configs: config file

- data: data class

- lib: local package to help the model training

- module: implementation of the final model

- scripts: many bash scripts for reproduction

- utilities: modified from https://github.com/pubgeo/monocular-geocentric-pose/tree/master/utilities

- eval_fn.py: local evaluate functions

- infer.py: python script for model inference

- train.py: python script for model training

## Step.1 install dependencies
```bash
# CUDA 10.1
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
bash ./scripts/install_ever.sh
# install invert_flow
cd utilities
python cythonize_invert_flow.py build_ext --inplace
```

## Step.2 prepare dataset
```bash
ln -s </path/to/dataset> ./dataset
bash ./scripts/prepare_dataset.sh
```

## Step.3 train the model
```bash
bash ./scripts/train.sh
```

## Step.4 infer the results
Please note that you can directly run this script if you use the trained model weights
```bash
bash ./scripts/infer.sh
```
The final prediction can be found at ./log
