#------------------------------------------------------------------------------
# Download and extract TEST data and META data (7 GB)
#------------------------------------------------------------------------------

cd $HOME/solution/data

curl -L -o geopose_test.csv "https://drivendata-prod.s3.amazonaws.com/data/78/public/geopose_test.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210825T210953Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a1973c223a6ae8c1464ab0b3da52b308ff2b6b61d4885fe4b099cf82af0449f3"
curl -L -o metadata.csv "https://drivendata-prod.s3.amazonaws.com/data/78/public/metadata.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYVI2LMPSY%2F20210825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210825T210953Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=809f909b90d46b903d8d4b3e415cb0bf2563913bd16f19f5d2b430447ae83fbd"

curl -L -O https://drivendata-public-assets.s3.amazonaws.com/geopose_test_rgbs.tar.gz

mkdir -p test_rgbs

tar xzf geopose_test_rgbs.tar.gz -C test_rgbs

#------------------------------------------------------------------------------
# Download and extract models (4 GB)
#------------------------------------------------------------------------------

cd $HOME/solution

curl -L -o models.tgz https://www.dropbox.com/s/tdyja1f7pn60s0p/models.tgz?dl=0
tar xzf models.tgz

#------------------------------------------------------------------------------
# Cythonize invert_flow
#------------------------------------------------------------------------------

cd $HOME/solution/monocular-geocentric-pose/utilities
python3 cythonize_invert_flow.py build_ext --inplace

cd $HOME/solution

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------