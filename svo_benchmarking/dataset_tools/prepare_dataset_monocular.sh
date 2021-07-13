#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Required arguments: <bag_name> <img_topic> <imu_topic>"
  exit 1
else
  printf "Going to process %s with image topic %s and imu topic %s.\n" "$1" "$2" "$3"
fi

BAG_NAME=$1
CAM0_TOPIC=$2
IMU_TOPIC=$3
DATA_DIR_NAME=${BAG_NAME%.bag}

mkdir -p $DATA_DIR_NAME/data/img
cd $DATA_DIR_NAME/data/img

# generate images
python2 ../../../python/bag_to_image.py ../../../$BAG_NAME  $CAM0_TOPIC 0
mv images.txt ../images.txt

cd .. #navigate back to /data

# read imu data and ground truth
python2 ../../python/bag_to_imu.py ../../$BAG_NAME $IMU_TOPIC

