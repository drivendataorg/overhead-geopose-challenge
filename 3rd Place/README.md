## Setup

```
$ conda create --name "geopose-model" python=3.8.10

$ pip install -r requirements.txt
$ conda install -c conda-forge gdal

$ pushd utilities
$ python cythonize_invert_flow.py build_ext --inplace
$ popd

$ python -m utilities.augmentation_vflow && echo "OK"
OK
```

If there are issues running `python cythonize_invert_flow.py build_ext --inplace`, then:
1. Clone the NGA code repository: `git clone https://github.com/pubgeo/monocular-geocentric-pose`
2. `conda install cython`
3. On the same machine and in the same environment where you will be running this model, run:
   ```
   cd monocular-geocentric-pose/utilities
   python cythonize_invert_flow.py build_ext --inplace
   ```
4. Copy all of the new files created in utilities to the utilities folder of this repository

## Preprocessing

My code expected that compettion files are located in `./data`.

```
$ pushd data
$ mkdir train && tar zxvf geopose_train.tar.gz -C train
$ mkdir test && tar zxvf geopose_test_rgbs.tar.gz -C test
$ mkdir submission_format && tar xf submission_format.tar -C submission_format
$ popd

$ python preprocessing.py
(snip)
[convert images] done in 2409 s
```

## Train (skip here if you use pretrained weights)

Large GRAM (more than 24GB) are required. I used RTX3090.

```
$ python model.py train model_1 --gpus 0
$ python model.py train model_2 --gpus 0
$ python model.py train model_3 --gpus 1
$ python model.py train model_4 --gpus 1
```

## Inference

If you want to use the pretrained weights, download the archive file from Google Drive.

* [model_weights.tar.gz (995MB)](https://drive.google.com/file/d/15TBEK58-HSjjtumn4TLcCprWoK32mE4u/view?usp=sharing)

```
$ tar zxvf model_weights.tar.gz
```

```
$ python model.py test model_1 --gpus 0
$ python model.py test model_2 --gpus 0
$ python model.py test model_3 --gpus 1
$ python model.py test model_4 --gpus 1
$ python ensemble.py
```

The image similarity search and matching process can be skipped by using the pre-computed file. If you want to use the pre-computed file, download the archive file from Google Drive.

* [matching_results.tar.gz (1.5MB)](https://drive.google.com/file/d/1RFa_SWeJ6VJ23qcBbm03RcJiuFv3QOkF/view?usp=sharing)

```
$ tar zxvf matching_results.tar.gz
$ python fusion.py fusion
```

The generated submission file is uploaded bewlo:

* [fusion_sub.tar.gz (4.7GB)](https://drive.google.com/file/d/1MuPwm87Mhg6N_LA3O2d-3Aq_IRqwI_sw/view?usp=sharing)

## Inference (Image similarity search and image matching)

```
$ python fusion.py search
[extract test features] start.
[extract test features] done in 3055 s
[extract train features] start.
Processing 0 of 5923...
Processing 50 of 5923...
(snip)
Processing 5900 of 5923...
[extract train features] done in 12450 s
[search top100] start.
loading feats
loading feats
search_with_faiss_cpu
dump
[search top100] done in 233 s.

$ python fusion.py match
[quick match] start.
(snip)
```
