SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")
export W=$(realpath "$DIR/../models/inception_v3")
echo "$W"

# snpe-net-run --container $W/dlc/inception_v3.dlc --input_list $W/data/cropped/raw_list.txt

/tmp/snpe-sample -d $W/dlc/inception_v3.dlc -i $W/data/cropped/file_list.txt -o /tmp

python ./scripts/inception_results.py -i $W/data/cropped/file_list.txt -o /tmp -l $W/data/imagenet_slim_labels.txt


