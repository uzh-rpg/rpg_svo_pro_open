# Known issues and possible improvements

1. The initialization of the visual-inertial odometry is currently done in a relatively naive manner: we simple wait for the backend to converge and then scale the map from the front-end accordingly. This mostly affects the stability of the monocular setup. More sophisticated methods, such as the ones used in VINS-Mono and ORB-SLAM3 could be adopted.
2. The global map at this moment takes relatively much memory, since no sparsification of the keyframes and  landmarks are done. Some work could be put in this direction to make it more efficient.
3. The matching of  the features in current frame to the global map is relatively fragile. The features are matched using direct matching individually. Selecting and matching multiple features simultaneously may improve the robustness.
4. In case of loop closing, we shift the sliding window and drop the marginalization term, as the marginalizaiton term depends on previous linearization points, which are not valid any more. This avoids inconsistent linearization points but makes the tracking after loop closing a bit unstable. Recalculating it from the global map, using relative formulation or adopting some approximation could be used to improve this.
5. Occasionally the pipeline with the global map may crash, some investigation needs to be done to improve the stability.

