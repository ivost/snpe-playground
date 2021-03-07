SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")
export W=$(realpath "$DIR/../models/ssd_mobilenet_v2_coco")
echo "$W"
# snpe-dlc-info -i /home/ivo/github/snpe-playground/models/ssd_mobilenet_v2_coco/dlc/mobilenet_ssd.dlc > info.txt
rm -rf /tmp/output
echo running inference
/tmp/snpe-sample -d $W/dlc/mobilenet_ssd.dlc -i $W/data/processed/file_list.txt -o /tmp/output
echo show results
python $DIR/ssd_results.py -i $W/data/processed/file_list.txt -o /tmp/output -l $W/data/labels.txt
