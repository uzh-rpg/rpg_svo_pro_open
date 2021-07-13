#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Required arguments: <bag_name> <gt_topic> <hand_eye>"
  exit 1
else
  printf "Going to process %s with gt topic %s and hand-eye %s.\n" "$1" "$2" "$3"
fi

# make sure to set the datatype of the ground truth to correct value in bag_to_pose.py
BAG_NAME=$1
GROUNDTRUTH_TOPIC=$2
HAND_EYE=$3

DATA_DIR_NAME=${BAG_NAME%.bag}
mkdir -p $DATA_DIR_NAME/data/
cd $DATA_DIR_NAME/data

echo "> Extracting groundtruth..."
python ../../python/bag_to_pose.py ../../$BAG_NAME $GROUNDTRUTH_TOPIC

echo "> Associating timestampes..."
python ../../python/associate_timestamps.py images.txt groundtruth.txt
mv matches.txt groundtruth_matches.txt

echo "> handeye transformation..."
if [ "$HAND_EYE" = "none" ]; then
  echo "Not performing hand-eye transformation."
else
  python ../../python/transform_trajectory.py groundtruth.txt ../../handeye_tfs/"handeye_$HAND_EYE.yaml"
  rm groundtruth.txt
  mv groundtruth_transformed.txt groundtruth.txt
fi

echo "> create stamped groundtruth..."
python ../../python/strip_gt_id.py groundtruth.txt

