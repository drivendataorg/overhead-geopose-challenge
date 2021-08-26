#------------------------------------------------------------------------------
# Inference
#------------------------------------------------------------------------------

# We loop over 5 architectures and 5 folds within each architecture (25 inferences total).
# "eval $X" command allows to get 3 separate shell variables: 
# MODEL_DIR, BACKBONE, and ENCODER_WEIGHTS (this approach imitates python zip function).
# For demo purposes it's convenient to run the whole construction below without 
# actual call to python3 command. It will just print 25 configurations

cd $HOME/solution

for X in "MODEL_DIR=run-20210608-2131-unet-resnet34-e500 BACKBONE=resnet34 ENCODER_WEIGHTS=imagenet" \
         "MODEL_DIR=run-20210626-0051-unet-inceptionv4-e290 BACKBONE=inceptionv4 ENCODER_WEIGHTS=imagenet" \
         "MODEL_DIR=run-20210704-1650-unet-seresnext50-e120 BACKBONE=se_resnext50_32x4d ENCODER_WEIGHTS=imagenet" \
         "MODEL_DIR=run-20210713-0048-unet-regnetx064-e80 BACKBONE=timm-regnetx_064 ENCODER_WEIGHTS=imagenet" \
         "MODEL_DIR=run-20210713-0107-unet-resnext50swsl-e230 BACKBONE=resnext50_32x4d ENCODER_WEIGHTS=swsl"
do 
eval $X

for FOLD_ID in 0 1 2 3 4
do

# We use wildcard in .pth file name because epoch number may vary
CKPT=$(echo $HOME/solution/models/$MODEL_DIR/model_f${FOLD_ID}_*.pth)
PREDS_DIR=$(echo $HOME/solution/output/$MODEL_DIR/preds_f${FOLD_ID})

echo "*****"
echo Current model dir: $MODEL_DIR
echo Current backbone: $BACKBONE
echo Current encoder weights: $ENCODER_WEIGHTS
echo Current fold: $FOLD_ID
echo Current model: $CKPT
echo Current preds dir: $PREDS_DIR

python3 $HOME/solution/monocular-geocentric-pose/main.py \
    --test \
    --model-path=$CKPT \
    --predictions-dir=$PREDS_DIR \
    --dataset-dir=$HOME/solution/data/test_rgbs \
    --batch-size=16 \
    --downsample=2 \
    --test-sub-dir="" \
    --convert-predictions-to-cm-and-compress=0 \
    --backbone=$BACKBONE \
    --encoder_weights=$ENCODER_WEIGHTS \
    --decoder=unet \

done
done

#------------------------------------------------------------------------------
# Ensemble
#------------------------------------------------------------------------------

cd $HOME/solution

OUT_DIR=$HOME/solution/output/ensemble_preds

python3 ensemble_and_compress.py \
    --in_dir=$HOME/solution/output \
    --out_dir=$OUT_DIR \
    --n_proc=16 \

#------------------------------------------------------------------------------
# Predict scale target based on GSD metadata using XGBoost
#------------------------------------------------------------------------------

cd $HOME/solution

COMPRESSED_DIR=${OUT_DIR}_converted_cm_compressed_uint16
XGB_DIR=${OUT_DIR}_converted_cm_compressed_uint16_xgb_scale

# Create full copy of a directory where ensemble predictions are stored
cp -r $COMPRESSED_DIR $XGB_DIR

# The following command will replace scale values in .json files,
# everything else remains the same
python3 xgb_scale_inference.py \
    --data_dir=$HOME/solution/data \
    --model_dir=$HOME/solution/models/run-20210720-0141-xgboost-scale \
    --out_dir=$XGB_DIR

#------------------------------------------------------------------------------
# Create submission
#------------------------------------------------------------------------------

cd $XGB_DIR

tar cfz ../submission.tgz *.*

cd ../..

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


