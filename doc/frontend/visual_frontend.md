# Run the visual front-end
Remember to source the workspace first
```sh
source ~/svo_ws/devel/setup.bash
```
## Example bags and launch files

Download the test bag file from [here](http://rpg.ifi.uzh.ch/datasets/airground_rig_s3_2013-03-18_21-38-48.bag). Launch SVO node:

```sh
roslaunch svo_ros run_from_bag.launch cam_name:=svo_test_pinhole
```

This will also start RVIZ. Then play the bag file:

```sh
rosbag play airground_rig_s3_2013-03-18_21-38-48.bag
```

Now you should be able to observe the camera motion and sparse map in RVIZ.

You can also download [another bag recorded with a fisheye camera](http://rpg.ifi.uzh.ch/datasets/test_fisheye.bag). Then you need to change the following line in `run_from_bag.launch`:

```xml
<rosparam file="$(find svo_ros)/param/pinhole.yaml" />
```

to

```xml
<rosparam file="$(find svo_ros)/param/fisheye.yaml" />
```
And launch SVO via:

```sh
roslaunch svo_ros run_from_bag.launch cam_name:=bluefox_25000826_fisheye
```

## Customize launch files
We provide several example launch files under `svo_ros/launch`, such as:
* `run_from_bag.launch`: run SVO on an topic that publishes images
* `live_nodelet.launch`: run svo as nodelet. These files can not be launched directly, since they depend on some private packages. But they can be used as an example for using svo with other nodes.

In the launch file, you have to specify the following for svo node/nodelet, as in `run_from_bag.launch`:
```xml
    <!-- Camera topic to subscribe to -->
    <param name="cam0_topic" value="camera/image_raw" type="str" />
    
    <!-- Camera calibration file -->
    <param name="calib_file" value="$(arg calib_file)" />
    
    <!--Parameters-->
    <rosparam file="$(find svo_ros)/param/pinhole.yaml" />

  Note that SVO also supports stereo cameras.
```

### Parameter files
We provide two example parameter files under `svo_ros/param`:
* `pinhole.yaml`: for relatively small field of view cameras
* `fisheye.yaml`: for cameras with wide angle lens (e.g., fisheye and catadioptric)

The parameters in these files are typical values. If you wish to change the parameters, please refer to the comments in these two files. 

### Camera calibration files
If you want to use your own camera, make sure a global shutter camera is used. A good choice is [the Bluefox camera](https://www.matrix-vision.com/USB2.0-single-board-camera-mvbluefox-mlc.html) from MatrixVision.
You can put camera calibration files under `svo_ros/calib` and load them as in `run_from_bag.launch`. We use yaml files to specify camera parameters. Please refer to [calibration.md](./calibration.md) for more details.

## Inertial-aided frontend

Without using the relatively heavy ceres-based backend, the frontend can also use IMU to facilitate visual tracking. This is not as accurate as a tightly coupled VIO, but is relatively lightweight.

We provide two launch files for [EuRoC](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) for monocular and stereo setups:

```sh
roslaunch svo_ros euroc_mono_frontend_imu.launch
roslaunch svo_ros euroc_stereo_frontend_imu.launch
```

These launch files read the parameters from `param/euroc_mono_imu.yaml` and `param/euroc_stereo_imu.yaml`. The parameters are not necessarily optimal for every sequence, but should be enough as a good starting point.

The first several images of many EuRoC sequences are not good for initialize `SVO`, especially for the monocular case. Therefore it is better to start the bag at a later time, for example:

```sh
rosbag play MH_01_easy.bag -s 50
rosbag play MH_02_easy.bag -s 45
rosbag play V1_01_easy.bag
rosbag play V1_02_medium.bag -s 13
rosbag play V2_01_easy.bag
rosbag play V2_02_medium.bag -s 13
```

This is to avoid, for example, strong rotation for monocular initialization. For more details on using SVO with this configuration, please read [the step-by-step instruction](./frontend_fla.md).

