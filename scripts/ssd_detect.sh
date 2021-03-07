SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")
export W=$(realpath "$DIR/../models/ssd_mobilenet_v2_coco")
echo "DIR: $DIR"
echo "WORKDIRL $W"

rm -rf $W/processed
echo preprocessing images
pushd $DIR
python ./ssd_prepare.py

# snpe-dlc-info -i /home/ivo/github/snpe-playground/models/ssd_mobilenet_v2_coco/dlc/mobilenet_ssd.dlc > info.txt
rm -rf /tmp/output
echo running inference
/tmp/snpe-sample -d $W/dlc/mobilenet_ssd.dlc -i $W/processed/file_list.txt -o /tmp/output
echo show results
python ./ssd_results.py -i $W/processed/file_list.txt -o /tmp/output -l $W/data/labels.txt

popd