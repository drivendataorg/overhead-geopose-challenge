#------------------------------------------------------------------------------
# Download and extract TRAIN data (58 GB)
#------------------------------------------------------------------------------

cd $HOME/solution/data

curl -L -o geopose_train.csv "https://drivendata-prod.s3.amazonaws.com/data/78/public/geopose_train.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210825T210953Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d98c8d3f1ecfe0dd5c0350dbef3a9dff7710b758841dfc1b7689297a19a5082b"
curl -L -o metadata.csv "https://drivendata-prod.s3.amazonaws.com/data/78/public/metadata.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210825T210953Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=809f909b90d46b903d8d4b3e415cb0bf2563913bd16f19f5d2b430447ae83fbd"

curl -L -O https://drivendata-public-assets.s3.amazonaws.com/geopose_train.tar.gz

mkdir -p train

tar xzf geopose_train.tar.gz -C train

#------------------------------------------------------------------------------
# Downsample data 2x
#------------------------------------------------------------------------------

cd $HOME/solution

python3 $HOME/solution/monocular-geocentric-pose/utilities/downsample_images.py \
    --indir=$HOME/solution/data/train \
    --outdir=$HOME/solution/data/train_re_2 \
    --unit=cm \
    --rgb-suffix=j2k \

#------------------------------------------------------------------------------
# Cythonize invert_flow
#------------------------------------------------------------------------------

cd $HOME/solution/monocular-geocentric-pose/utilities

python3 cythonize_invert_flow.py build_ext --inplace

cd $HOME/solution

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


