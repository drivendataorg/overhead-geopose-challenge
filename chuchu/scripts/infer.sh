export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

model_dir='./log/comp/g2_hr32_vflow_adamw_mstep'
python ./infer.py competition \
  --model_dir=$model_dir \
  --test_data_dir='./dataset/test' \
  --test_csv_file='./dataset/geopose_test.csv' \
  --output_dir=$model_dir/tta_output \
  --tta=True

model_dir='./log/comp/OMA_hr32_vflow_adamw_mstep_ft/3w'
python ./infer.py competition \
  --model_dir=$model_dir \
  --test_data_dir='./dataset/test' \
  --test_csv_file='./dataset/geopose_test.csv' \
  --output_dir='./log/comp/OMA_hr32_vflow_adamw_mstep_ft/3w/prediction' \
  --city='OMA' \
  --tta=True

mkdir ./log/final_prediction
cp -r ./log/comp/g2_hr32_vflow_adamw_mstep/tta_output_converted_cm_compressed_uint16/* ./log/final_prediction/
cp -r ./log/comp/OMA_hr32_vflow_adamw_mstep_ft/3w/prediction_converted_cm_compressed_uint16/* ./log/final_prediction/

cd ./log/final_prediction ; tar -czf ../final_prediction.tar.gz *