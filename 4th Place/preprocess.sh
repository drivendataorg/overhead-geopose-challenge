#/bin/bash


DATA_PATH=./data
INDIR=${DATA_PATH}/train
OUTDIR=${DATA_PATH}/train_tif
UNIT="cm"
RGB_SUFFIX="j2k"


python3 ./overhead-geopose-challenge/utilities/downsample_images.py \
    --indir ${INDIR} \
    --outdir ${OUTDIR} \
    --unit ${UNIT} \
    --rgb-suffix ${RGB_SUFFIX}
