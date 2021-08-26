#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

python3 $HOME/solution/monocular-geocentric-pose/main.py \
    --train \
    --rgb-suffix=tif \
    --unit=m \
    --save-period=1 \
    --val-period=1 \
    --batch-size=6 \
    --learning-rate=0.0001 \
    --num-epochs=230 \
    --augmentation \
    --num-workers=4 \
    --save-best \
    --dataset-dir=data \
    --train-sub-dir=$HOME/solution/data/train_f${1}_re_2 \
    --valid-sub-dir=$HOME/solution/data/val_f${1}_re_2 \
    --checkpoint-dir=ckpt_f${1} \
    --backbone=resnext50_32x4d \
    --amp_enabled=1 \
    --encoder_weights=imagenet \
    --decoder=unet \

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

