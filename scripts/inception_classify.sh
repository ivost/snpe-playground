SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")
export W=$(realpath "$DIR/../models/inception_v3")
echo "$W"
snpe-net-run --container $W/dlc/inception_v3.dlc --input_list $W/data/cropped/raw_list.txt