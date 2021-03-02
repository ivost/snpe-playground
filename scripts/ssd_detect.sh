SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")
export W=$(realpath "$DIR/../models/ssd_mobilenet_v2_coco")
echo "$W"

/tmp/snpe-sample -d $W/dlc/mobilenet_ssd.dlc -i $W/data/cropped/file_list.txt -o /tmp/output

# python ./ssd_results.py -i $W/data/cropped/file_list.txt -o /tmp/output -l $W/data/labels.txt
