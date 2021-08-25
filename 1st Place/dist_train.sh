DATA=$1
WDATA=$2
FOLD=$3
GPUS=$4
PYTHONPATH=.  python -u -m torch.distributed.launch \
 --nproc_per_node=$GPUS \
 --master_port 9901 \
  training/train_segmentor.py \
 --world-size $GPUS  \
 --distributed \
 --config configs/v2l.json \
 --workers 8 \
 --multiplier 4 \
 --data-dir $WDATA \
 --train-dir $DATA \
 --test_every 2 \
 --crop 896 \
 --freeze-epochs 0 \
 --fold $FOLD \
 --prefix folds_
