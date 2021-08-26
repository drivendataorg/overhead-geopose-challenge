#------------------------------------------------------------------------------
# machine-0 / fold-0
#------------------------------------------------------------------------------
# We have 5 separate machines
# On each machine we will create single split of data corresponding to the given fold
# e.g. for fold_0 we will have directories "train_f0_re_2" and "val_f0_re_2"
# Data has to be split only once. If we want to train another architecture on the same fold
# there is no need to run split again.
# To train on another fold change "FOLD_ID" variable (from 0 to 4)
# To train another architecture change "MODEL_DIR"
# In total for my final solution I trained 25 models 
# (5 architectures by 5 folds) plus XGBoost model based on meta features
#------------------------------------------------------------------------------

FOLD_ID=0
MODEL_DIR=run-20210608-2131-unet-resnet34-e500

cd $HOME/solution

# Run this once on each new machine
# This command will split data in two separate directories: "train_f0_re_2" and "val_f0_re_2"
python3 split_data.py \
    --data_dir=$HOME/solution/data \
    --fold_id=$FOLD_ID \

cd $HOME/solution/training_scripts/$MODEL_DIR

bash run.sh $FOLD_ID

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
