# Prepare dataset for benchmarking

In this document, we describe how to convert ROS bags/EuRoC datasets to the format that will be used by the benchmarking tool.

1. [Dataset Format](#dataset-format)
2. [Instructions](#instructions)
3. [Diverse Scripts](#diverse-scripts)

## Dataset Format
Each dataset, including the images, IMU measurements and groundtruth will be organized in a folder, for example:

```
<dataset_name>
├── calib.yaml --> calibration file
├── data
│   ├── groundtruth_matches.txt
│   ├── groundtruth.txt
│   ├── stamped_groundtruth.txt
│   ├── images.txt
│   ├── img --> folder containing actual images
│   └── imu.txt
└── dataset.yaml --> optional: can be used to specify dataset-specific parameters

```

The dataset essentially lives in the `data` folder, and the files are of the following formats

* `groundtruth.txt`: `id time_sec px py pz qx qy qz qw`
* (optional) `stamped_groundtruth.txt`: `time_sec px py pz qx qy qz qw`
* `images.txt`: specifying the image files, with respect to the `data` folder
    * monocular: `id time_sec image`
    * stereo: `id time_sec image_0 image_1`
* `imu.txt`: `id time_sec wx wy wz ax ay az`
* `groundtruth_matches.txt`: `id_img id_gt`

The `id` field is simply an increasing integer. The file `groundtruth_matches.txt` is useful to identify the groundtruth corresponding to a specific image.


## Instructions

First put the ROS bag to process in the `dataset_tools` folder, then the following steps will generate a folder of the aforementioned format.

### Extract images and IMU measurements from bags
For monocular

```bash
./prepare_dataset_monocular <bag_name> <img_topic> <imu_topic>
```
and stereo
```bash
./prepare_dataset_monocular.sh <bag_name> <img0_topic> <img1_topic> <imu_topic>
```

### Extract groundtruth
**EuRoC**

Use the script `python/asl_to_groundtruth_to_pose.py` to convert the groundtruth in EuRoC to our format.

**ROS bags**

Run the script:

```bash
./generate_ground_truth.sh <bag_name> <groundtruth_topic> <hand_eye>
```
The script uses `python/bag_to_pose.py` to extract pose messages. By default, `PoseStamped` is used. You can modify the script for other message types. 

The parameter `<hand_eye>` specify the hand-eye calibration to apply to the groundtruth.
It will look for `handeye_tfs/handeye_<hand_eye>.yaml`.
If it is `none`, then no hand-eye transformation is applied.


### Copy essential files
The last steps are to put what we extracted to the folder from which the benchmark tool will read:

1. Copy the sensor calibration file into the generated  dataset folder (see examples in `svo_benchmarking/data`). Rename it to `calib.yaml` (or change the name in the `experiment.yaml` file when running the benchmarking).

2. move the dataset folder to `svo_benchmarking/data`.

Now the dataset is ready for benchmarking. For the correct structure, one can check some examples under `data/euroc/stereo` (images are omitted).


## Diverse Scripts
It is common that you have to deal with other dataset format. There are some scripts under `python` that may be of use. For example:

* `prepend_id.py`: will prepend an increasing `id` field to a text file per line.
* `swap_stamped_imu_meas`: swap the order of accelerometer and gyroscope measurements.
* ...

Many of the scripts are already used in the previous instructions.

