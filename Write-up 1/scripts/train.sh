export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
export PYTHONPATH=$PYTHONPATH:$(pwd)
config_path='g2_hr32_vflow_adamw_mstep'
model_dir='./log/comp/g2_hr32_vflow_adamw_mstep'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 5677 ./train.py \
  --config_path=${config_path} \
  --model_dir=${model_dir}  \
  data.train.params.batch_size 8

export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
config_path='g2_hr32_vflow_adamw_mstep'
model_dir='./log/comp/OMA_hr32_vflow_adamw_mstep_ft/3w'
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 34443 ./train.py \
  --config_path=${config_path} \
  --model_dir=${model_dir}  \
  model.params.init_from_weight True \
  model.params.GLOBAL.weight.path "./log/comp/g2_hr32_vflow_adamw_mstep/model-90000.pth" \
  data.train.params.batch_size 8 \
  data.train.params.city 'OMA' \
  data.train.params.max_building_agl 200.0 \
  learning_rate.params.steps '(20000, 27000)' \
  train.num_iters 30000

