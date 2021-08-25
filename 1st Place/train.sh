DATA_DIR=$1
WDATA_DIR=$2
GPUS=$3

# prepare tif images
echo 'Convert images'
PYTHONPATH=. python utilities/downsample_images.py --indir $DATA_DIR --outdir $WDATA_DIR

echo 'Start training'
# train fold 5
./dist_train.sh $DATA_DIR $WDATA_DIR  5 $GPUS > logs/l5
./dist_train_tune.sh $DATA_DIR $WDATA_DIR  5 $GPUS  weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 >> logs/l5

rm -r preds

# train fold 6
./dist_train.sh $DATA_DIR $WDATA_DIR  6 $GPUS > logs/l6
./dist_train_tune.sh $DATA_DIR $WDATA_DIR  6 $GPUS  weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 >> logs/l6

echo 'Done training'