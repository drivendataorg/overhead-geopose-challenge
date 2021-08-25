DATA=$1
WDATA=$2
FOLD=$3
GPUS=$4
RESUME=$5
PYTHONPATH=.  python -u -m torch.distributed.launch \
 --nproc_per_node=$GPUS \
 --master_port 9901 \
  training/train_segmentor.py \
 --world-size $GPUS  \
 --distributed \
 --config configs/v2l_tune.json \
 --workers 8 \
 --multiplier 1 \
 --data-dir $WDATA \
 --train-dir $DATA \
 --test_every 2 \
 --crop 1792 \
 --freeze-epochs 0 \
 --fold $FOLD \
 --prefix folds_ \
 --resume $RESUME \
 --from-zero