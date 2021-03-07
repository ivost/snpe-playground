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
```
cd inception_v3
snpe-net-run --container dlc/inception_v3.dlc --input_list data/cropped/raw_list.txt
```


====

install caffee on ubu16 py3.5

https://github.com/adeelz92/Install-Caffe-on-Ubuntu-16.04-Python-3

(venv) 05:29:21 caffe$ export PYTHONPATH=$HOME/caffe/python
(venv) 05:29:53 caffe$ ls $PYTHONPATH 
caffe        CMakeLists.txt  draw_net.py       train.py
classify.py  detect.py       requirements.txt
(venv) 05:29:57 caffe$ python3 -m site
sys.path = [
    '/home/ivo/caffe',
    '/home/ivo/caffe/python',
    '/usr/lib/python35.zip',
    '/usr/lib/python3.5',
    '/usr/lib/python3.5/plat-x86_64-linux-gnu',
    '/usr/lib/python3.5/lib-dynload',
    '/home/ivo/venv/lib/python3.5/site-packages',
    '/home/ivo/.local/lib/python3.5/site-packages',
    '/usr/lib/python3.5/site-packages',
    '/usr/local/lib/python3.5/dist-packages',
    '/usr/lib/python3/dist-packages',
]
USER_BASE: '/home/ivo/.local' (exists)
USER_SITE: '/home/ivo/.local/lib/python3.5/site-packages' (exists)
ENABLE_USER_SITE: True


****

For ssd mobilenet

https://developer.qualcomm.com/docs/snpe/convert_mobilenetssd.html

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

pip install python-dateutil --upgrade


The input and output layers:

Input layer is specified in MobileNetSSD_deploy.prototxt file, via input_shape.
By default, the output layer is the last layer as specified in MobileNetSSD_deploy.prototxt file. In this case that is detection_out (DetectionOutput) layer.


To see info about converted DLC model, use snpe-dlc-info tool

snpe-dlc-info -i mobilenet_ssd.dlc

For ONNX

https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
