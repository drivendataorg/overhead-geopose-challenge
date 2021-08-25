#/bin/bash


DATA_PATH=./data
INDIR=${DATA_PATH}/train
UNIT="cm"
RGB_SUFFIX="j2k"
LMDB=./data/train.lmdb


python3 ./overhead-geopose-challenge/utilities/downsample_images.py \
    --indir ${INDIR} \
    --lmdb ${LMDB} \
    --unit ${UNIT} \
    --rgb-suffix ${RGB_SUFFIX}
