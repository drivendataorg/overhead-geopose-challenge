export DRIVENDATA_OVERHEAD_GEOPOSE="CHANGE_THIS_VALUE_TO_THE_LOCATION_OF_TRAIN_DATA"
export OMP_NUM_THREADS=8

python -m torch.distributed.launch --nproc_per_node=2 train.py\
  model=b7_unet\
  optimizer=adamw\
  scheduler=cos\
  loss=huber_cos\
  train=light_768\
  train.epochs=250\
  train.show=False\
  train.loaders.train.num_workers=10\
  batch_size=4\
  dataset.fold=0\
  seed=555