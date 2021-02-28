export ASSET_DIR=./tmp/assets
# [[ ! -d $ASSET_DIR ]] && mkdir -p $ASSET_DIR
python3 $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py --assets $ASSET_DIR --download --runtime cpu
