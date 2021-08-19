FROM nvcr.io/nvidia/pytorch:21.06-py3
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
#install specific versions of gdal
RUN apt install libgdal-dev=3.0.4+dfsg-1build3 -y
RUN pip install pygdal==3.0.4.6 albumentations
RUN pip install timm==0.4.12 segmentation_models_pytorch cython tensorboardx madgrad



# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006
WORKDIR /work

RUN mkdir -p logs
RUN mkdir -p weights

RUN wget -O weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2 https://www.dropbox.com/s/0uplf7b5d502ul2/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2?dl=0
RUN wget -O weights/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2 https://www.dropbox.com/s/54dhmlbdip23fif/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2?dl=0

RUN mkdir -p /root/.cache/torch/hub/checkpoints/
RUN wget -O /root/.cache/torch/hub/checkpoints/tf_efficientnetv2_l_21k-91a19ec9.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_l_21k-91a19ec9.pth

COPY . /work/
RUN PYTHONPATH=. python utilities/cythonize_invert_flow.py build_ext --inplace

RUN chmod 777 dist_train.sh
RUN chmod 777 dist_train_tune.sh
RUN chmod 777 train.sh
RUN chmod 777 test.sh

RUN ["/bin/bash"]