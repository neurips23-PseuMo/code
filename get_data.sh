mkdir dataset && cd dataset

# step1: download Argoverse HD Maps
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz
tar xf hd_maps.tar.gz
# copy map to argoverseapi root folder
PY_SITE_PACKAGE_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "copying map files to python3.7/site-packages: ", $PY_SITE_PACKAGE_PATH
cp -r map_files $PY_SITE_PACKAGE_PATH

# step2: download Argoverse Motion Forecasting **v1.1** 
# train + val + test
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_train_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_val_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_test_v1.1.tar.gz
tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz

# step3: preprocess data to accelerate training

python preprocess_data.py -m lanegcn