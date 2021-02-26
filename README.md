# snpe-playground

conda create --name snpe python=3.5

conda activate snpe

https://developer.qualcomm.com/docs/snpe/tutorial_setup.html

### Download the model and prepare the assets

The assets directory contains the network model assets. 
If the assets have been previously downloaded set the ASSETS_DIR to this directory 
otherwise select a target directory to store the assets as they are downloaded. 
If the assets are already downloaded to a directory (e.g. ~/tmpdir) then issue the following command.

```
mkdir ./assets
python3 $SNPE_ROOT/models/inception_v3/scripts/setup_inceptionv3.py -a ./assets -d
export ASSET_DIR=./assets

```

The Inception v3 Imagenet classification model is trained to classify images with 1000 labels.
cd inception_v3
snpe-net-run --container dlc/inception_v3.dlc --input_list data/cropped/raw_list.txt
