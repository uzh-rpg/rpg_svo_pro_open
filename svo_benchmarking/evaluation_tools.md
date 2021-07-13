We provide the following scripts in addition to the accuracy evaluation described in `README.md`:

* [Stationary Precision](#stationary-precision): `scripts/analyze_stationary.py`
* [Calculate Trajectory Length](#calculate-trajectory-length): `scripts/cal_traj_len.py`

## Stationary Precision
To evaluate the stationary precision of the pipeline, one first needs record a dataset where the sensor will be static for a time duration. Then record the output of the pipeline on the dataset, specifically the topic `/svo/backend_pose_imu` (type `geometry_msgs/PoseWithCovarianceStamped`). You can find provided datasets in `dataset_tools/datasets_list.md`.

Given a bag contains the pose estimate topic (`/svo/backend_pose_imu`), you can run the script, for example, as
```
rosrun svo_benchmarking analyze_stationary.py visensor_workshop_output.bag /svo/backend_pose_imu 0.3 0.9

```
The last two numbers gives the start and the end (as percentage) of the poses that will be taken as static. For example, if the pose estimates from `t=0` to `t=100s` are recorded, the pose estimates from `t=30s` to `t=90s` will be used to analyze the stationary precision. This is to exclude non-static poses for, e.g., the situation that the sensor is moving at the beginning for initialization.
You will see the output in the terminal similar to
```
The average position is [-0.1942487  -0.06126397 -0.23687582].
The standard deviation of each axis is [ 0.00038256  0.00019409  0.00032888].
The standard deviation of position is 0.000540697332339
```
and also a plot will be generated in the same folder.

## Calculate Trajectory Length
We calculate the length of the trajectory from the messages that record the groundtruth motion (type `geometry_msgs/PoseStamped`). For convenience, the script will find all the bag files in a folder and calculate the trajectory length from the specified topic.
For example, when the following command is executed:
```
rosrun svo_benchmarking cal_traj_len.py ./flyingroom_visensor /optitrack/visensor
```
the script will find all bag files under `./flyingroom_visensor`, calculate the trajectory length from `/optitrack/visensor` and output the result in the terminal for each bag.
