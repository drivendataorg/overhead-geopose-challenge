DATA=$1
GPUS=$2

PYTHONPATH=.   python -u -m torch.distributed.launch \
 --nproc_per_node=$GPUS \
 main.py \
--world_size $GPUS \
--test \
--predictions-dir=submission \
--dataset-dir=$DATA \
--model-path weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 \
--batch-size=1 \
--gpus=1 \
--unit="cm" \
--downsample=1 \
--test-sub-dir="" \
--convert-predictions-to-cm-and-compress=True

python gen_tar.py