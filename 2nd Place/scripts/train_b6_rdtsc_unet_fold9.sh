export DRIVENDATA_OVERHEAD_GEOPOSE="CHANGE_THIS_VALUE_TO_THE_LOCATION_OF_TRAIN_DATA"
export OMP_NUM_THREADS=8

python -m torch.distributed.launch --nproc_per_node=2 train.py\
  model=b6_rdtsc_unet\
  optimizer=adamw_fp32\
  scheduler=plateau\
  loss=huber_cos\
  train=light_768\
  train.epochs=300\
  train.early_stopping=50\
  train.show=False\
  train.loaders.train.num_workers=8\
  batch_size=3\
  dataset.fold=9\
  seed=777