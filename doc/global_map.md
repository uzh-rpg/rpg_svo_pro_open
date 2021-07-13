# Using SVO with iSAM2 based global map

As described in [SVO + OKVIS backend](./vio.md), fixed-lag smoothing is good for constant time update, but inevitably drifts over time. Using loop closure and pose graph to maintain global consistency is effective to a certain extent. However, lacking the ability to consider raw sensor measurements, loop closure + pose graph is limited in terms of accuracy. 

Therefore, we integrated a global bundle adjusted map using iSAM2. In short, once a keyframes goes out of the sliding window of the Ceres backend, we add it to a global visual-inertial bundle adjustment managed by iSAM2, which is executed in a separate thread. Moreover, in the visual front-end, we try to track points that are already optimized by iSAM2 (assuming iSAM2 converges fast enough and offers better accuracy).

The motivation of the design is to achieve high real-time accuracy by localizing against a globally consistent map at every frame, instead of relying on non-causal information to correct drift. The matching of the current frame against the global map is the same as in SVO - via direct tracking using patches. This is not very robust, but works quite well in mild motion (probably thanks to the gyroscopes).

One important implementation detail is that, to deal with occasionally indeterministic error with iSAM2, a weak prior of the landmark depth is added when the landmark is added to the global map (assuming the Ceres-based back-end provides reasonable initial estimate).

Note that the global map function is only tested with monocular configuration and is relatively unstable compared with the visual-inertial odometry part.



## Examples on EuRoC

Source the workspace first:
```sh
source ~/svo_ws/devel/setup.bash
```
Executing
```
roslaunch svo_ros euroc_global_map_mono.launch
rosbag play MH_03_medium.bag -s 15
```
You should see some visualization as below:

![](./images/mh03_gm.gif)

The green points are the landmarks that are observed by the frames that are currently in the sliding window optimization, which is indicated by the trailing lines. The blue points are the landmarks that are optimized and managed by iSAM2. As mentioned above, once the green points go out of the sliding window, they are taken over by iSAM2. You can observe the points optimized by iSAM2 are slightly more accurate.

Once the camera moves back to the region that is already mapped, the pipeline tries to match features against the blue points, which are indicated by the golden lines. The successfully matched blue points are treated as **fixed anchors** in the sliding window optimization, which anchors the sliding window to the consistent global map managed by iSAM2. The green rectangle surround the image indicates that there are enough fixed landmarks observed in the current sliding window, and we can drop the fixation at the first frame that is usually done in sliding window optimization.