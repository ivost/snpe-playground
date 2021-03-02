SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")
export W=$(realpath "$DIR/../models/ssd_mobilenet_v2_coco")
echo "$W"

# snpe-dlc-info -i /home/ivo/github/snpe-playground/models/ssd_mobilenet_v2_coco/dlc/mobilenet_ssd.dlc > info.txt

/tmp/snpe-sample -d $W/dlc/mobilenet_ssd.dlc -i $W/data/cropped/file_list.txt -o /tmp/output

python ./ssd_results.py -i $W/data/cropped/file_list.txt -o /tmp/output -l $W/data/labels.txt

