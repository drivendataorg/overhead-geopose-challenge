#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# In fact I trained folds 0,1,2,3 with learning rate specified below (0.0001),
# but in fold 4 model was not learning so I set 2x smaller learning rate (0.00005) 
# and trained the same number of epochs as other folds. 

python3 $HOME/solution/monocular-geocentric-pose/main.py \
    --train \
    --rgb-suffix=tif \
    --unit=m \
    --save-period=1 \
    --val-period=1 \
    --batch-size=6 \
    --learning-rate=0.0001 \
    --num-epochs=290 \
    --augmentation \
    --num-workers=4 \
    --save-best \
    --dataset-dir=data \
    --train-sub-dir=$HOME/solution/data/train_f${1}_re_2 \
    --valid-sub-dir=$HOME/solution/data/val_f${1}_re_2 \
    --checkpoint-dir=ckpt_f${1} \
    --backbone=inceptionv4 \
    --amp_enabled=1 \
    --encoder_weights=imagenet \
    --decoder=unet \

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

