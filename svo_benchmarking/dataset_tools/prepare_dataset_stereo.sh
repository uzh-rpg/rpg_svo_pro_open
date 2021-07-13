#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Required arguments: <bag_name> <img0_topic> <img1_topic> <imu_topic>"
  exit 1
else
  printf "Going to process %s with image topics %s and %s and imu topic %s.\n" "$1" "$2" "$3" "$4"
fi

BAG_NAME=$1
CAM0_TOPIC=$2
CAM1_TOPIC=$3
IMU_TOPIC=$4
DATA_DIR_NAME=${BAG_NAME%.bag}
mkdir -p $DATA_DIR_NAME/data/img
cd $DATA_DIR_NAME/data/img

# generate images
python2 ../../../python/bag_to_image.py ../../../$BAG_NAME  $CAM0_TOPIC 0
mv images.txt ../images_0.txt
python2 ../../../python/bag_to_image.py ../../../$BAG_NAME $CAM1_TOPIC 1
mv images.txt ../images_1.txt
cd .. #navigate back to /data
python2 ../../python/combine_images.py
mv images_combined.txt images.txt
rm images_0.txt
rm images_1.txt

# read imu data and ground truth
python2 ../../python/bag_to_imu.py ../../$BAG_NAME $IMU_TOPIC
