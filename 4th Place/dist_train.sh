#!/bin/bash


set -eu


GPU=${GPU:-0,1,2,3}
PORT=${PORT:-29500}
N_GPUS=4

ENCODER_WEIGHTS=imagenet
BATCH_SIZE=8
DATASET_DIR=./data/train_tif
TRAIN_DF=./data/geopose_train.csv
RGB_SUFFIX=tif
UNIT=m
N_FOLDS=40
OPTIM=fusedadam
LR=0.0001
SCHEDULER=cosa
MODEL_TYPE=unet
LMDB=./data/train.lmdb

NUM_EPOCHS=525
FOLD=1
BACKBONE=efficientnet-b6
CHECKPOINT=./chkps_dist/${BACKBONE}/${FOLD}_aug_${MODEL_TYPE}


CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --nproc_per_node=${N_GPUS} --master_port=${PORT} \
    ./overhead-geopose-challenge/main.py \
    --train \
    --use-city \
    --fp16 \
    --agl-weight=2 \
    --angle-weight=50  \
    --scale-weight=50  \
    --lmdb=${LMDB} \
    --model-type ${MODEL_TYPE} \
    --backbone ${BACKBONE} \
    --encoder-weights ${ENCODER_WEIGHTS} \
    --optim ${OPTIM} \
    --learning-rate ${LR} \
    --scheduler ${SCHEDULER} \
    --T-max 25 \
    --num-epochs=${NUM_EPOCHS} \
    --checkpoint-dir=${CHECKPOINT} \
    --dataset-dir=${DATASET_DIR} \
    --train-path-df=${TRAIN_DF} \
    --batch-size=${BATCH_SIZE} \
    --num-workers=${BATCH_SIZE} \
    --rgb-suffix=${RGB_SUFFIX} \
    --unit=${UNIT} \
    --n-folds=${N_FOLDS} \
    --fold=${FOLD} \
    --lmdb=${LMDB} \
    --augmentation \
    --albu \
    --distributed


FOLD=2
for BACKBONE in efficientnet-b7 senet154; do
    CHECKPOINT=./chkps_dist/${BACKBONE}/${FOLD}_aug_${MODEL_TYPE}

    CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --nproc_per_node=${N_GPUS} --master_port=${PORT} \
        ./overhead-geopose-challenge/main.py \
        --train \
        --use-city \
        --fp16 \
        --agl-weight=2 \
        --angle-weight=50  \
        --scale-weight=50  \
        --lmdb=${LMDB} \
        --model-type ${MODEL_TYPE} \
        --backbone ${BACKBONE} \
        --encoder-weights ${ENCODER_WEIGHTS} \
        --optim ${OPTIM} \
        --learning-rate ${LR} \
        --scheduler ${SCHEDULER} \
        --T-max 25 \
        --num-epochs=${NUM_EPOCHS} \
        --checkpoint-dir=${CHECKPOINT} \
        --dataset-dir=${DATASET_DIR} \
        --train-path-df=${TRAIN_DF} \
        --batch-size=${BATCH_SIZE} \
        --num-workers=${BATCH_SIZE} \
        --rgb-suffix=${RGB_SUFFIX} \
        --unit=${UNIT} \
        --n-folds=${N_FOLDS} \
        --fold=${FOLD} \
        --lmdb=${LMDB} \
        --augmentation \
        --albu \
        --distributed
done


NUM_EPOCHS=1025
FOLD=1
BACKBONE=efficientnet-b6
CHECKPOINT=./chkps_dist/${BACKBONE}/${FOLD}_aug_ft_${MODEL_TYPE}
LOAD=./chkps_dist/${BACKBONE}/${FOLD}_aug_${MODEL_TYPE}/model_last.pth

CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --nproc_per_node=${N_GPUS} --master_port=${PORT} \
    ./overhead-geopose-challenge/main.py \
    --train \
    --use-city \
    --fp16 \
    --agl-weight=2 \
    --angle-weight=50  \
    --scale-weight=50  \
    --lmdb=${LMDB} \
    --model-type ${MODEL_TYPE} \
    --backbone ${BACKBONE} \
    --encoder-weights ${ENCODER_WEIGHTS} \
    --optim ${OPTIM} \
    --learning-rate ${LR} \
    --scheduler ${SCHEDULER} \
    --T-max 25 \
    --num-epochs=${NUM_EPOCHS} \
    --checkpoint-dir=${CHECKPOINT} \
    --dataset-dir=${DATASET_DIR} \
    --train-path-df=${TRAIN_DF} \
    --batch-size=${BATCH_SIZE} \
    --num-workers=${BATCH_SIZE} \
    --rgb-suffix=${RGB_SUFFIX} \
    --unit=${UNIT} \
    --n-folds=${N_FOLDS} \
    --fold=${FOLD} \
    --lmdb=${LMDB} \
    --distributed \
    --load=${LOAD}


FOLD=2
for BACKBONE in efficientnet-b7 senet154; do
    CHECKPOINT=./chkps_dist/${BACKBONE}/${FOLD}_aug_ft_${MODEL_TYPE}
    LOAD=./chkps_dist/${BACKBONE}/${FOLD}_aug_${MODEL_TYPE}/model_last.pth

    CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --nproc_per_node=${N_GPUS} --master_port=${PORT} \
        ./overhead-geopose-challenge/main.py \
        --train \
        --use-city \
        --fp16 \
        --agl-weight=2 \
        --angle-weight=50  \
        --scale-weight=50  \
        --lmdb=${LMDB} \
        --model-type ${MODEL_TYPE} \
        --backbone ${BACKBONE} \
        --encoder-weights ${ENCODER_WEIGHTS} \
        --optim ${OPTIM} \
        --learning-rate ${LR} \
        --scheduler ${SCHEDULER} \
        --T-max 25 \
        --num-epochs=${NUM_EPOCHS} \
        --checkpoint-dir=${CHECKPOINT} \
        --dataset-dir=${DATASET_DIR} \
        --train-path-df=${TRAIN_DF} \
        --batch-size=${BATCH_SIZE} \
        --num-workers=${BATCH_SIZE} \
        --rgb-suffix=${RGB_SUFFIX} \
        --unit=${UNIT} \
        --n-folds=${N_FOLDS} \
        --fold=${FOLD} \
        --lmdb=${LMDB} \
        --distributed \
        --load=${LOAD}
done
