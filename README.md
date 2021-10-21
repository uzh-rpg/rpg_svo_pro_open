# rpg_svo_pro

This repo includes **SVO Pro** which is the newest version of Semi-direct Visual Odometry (SVO) developed over the past few years at the Robotics and Perception Group (RPG).
SVO was born as a fast and versatile visual front-end as described in the [SVO paper (TRO-17)](http://rpg.ifi.uzh.ch/docs/TRO17_Forster-SVO.pdf). Since then, different extensions have been integrated through various research and industrial projects.
SVO Pro features the support of different [camera models](http://rpg.ifi.uzh.ch/docs/ICRA16_Zhang.pdf), [active exposure control](http://rpg.ifi.uzh.ch/docs/ICRA17_Zhang.pdf), a sliding window based backend, and global bundle adjustment with loop closure.

In summary, this repository offers the following functionalities:

* Visual-odometry: The most recent version of SVO that supports perspective and fisheye/catadioptric cameras in monocular or stereo setup. It also includes active exposure control.
* Visual-inertial odometry: SVO fronted + visual-inertial sliding window optimization backend (modified from [OKVIS](https://github.com/ethz-asl/okvis))
* Visual-inertial SLAM: SVO frontend + visual-inertial sliding window optimization backend +  globally bundle adjusted map (using [iSAM2](https://gtsam.org/)). The global map is updated in real-time, thanks to iSAM2, and used for localization at frame-rate.
* Visual-inertial SLAM with loop closure: Loop closures, via [DBoW2](https://github.com/dorian3d/DBoW2), are integrated in the global bundle adjustment. Pose graph optimization is also included as a lightweight replacement of the global bundle adjustment.

An example of the visual-inertial SLAM pipeline on EuRoC dataset is below (green points - sliding window; blue points - iSAM2 map):

![](./doc/images/v102_gm.gif)

SVO Pro and its extensions have been used to support various projects at RPG, such as our recent work on [multiple camera SLAM](http://rpg.ifi.uzh.ch/docs/ICRA20_Kuo.pdf), [voxel map for visual SLAM](http://rpg.ifi.uzh.ch/docs/ICRA20_Muglikar.pdf) and [the tight-coupling of global positional measurements into VIO](http://rpg.ifi.uzh.ch/docs/IROS20_Cioffi.pdf). We hope that the efforts we made can facilitate the research and applications of SLAM and spatial perception.

## License

The code is licensed under GPLv3. For commercial use, please contact `sdavide [at] ifi [dot] uzh [dot] ch`.

The visual-inertial backend is modified from OKVIS, and the license is retained at the beginning of the related files.

## Credits

If you use the code in the academic context, please cite:

* Christian Forster, Matia Pizzoli, Davide Scaramuzza. SVO: Fast Semi-Direct Monocular Visual Odometry. ICRA, 2014. [bibtex](./doc/bib/Forster14icra.bib)
* Christian Forster, Zichao Zhang, Michael Gassner, Manuel Werlberger, Davide Scaramuzza. SVO: Semi-Direct Visual Odometry for Monocular and Multi-Camera Systems. TRO, 2017. [bibtex](./doc/bib/Forster17tro.bib)

Additionally, please cite the following papers for the specific extensions you make use of:

* *Fisheye/catadioptric camera extension*: Zichao Zhang, Henri Rebecq, Christian Forster, Davide Scaramuzza. Benefit of Large Field-of-View Cameras for Visual Odometry. ICRA, 2016. [bibtex](./doc/bib/Zhang16icra.bib)
* *Brightness/exposure compensation*: Zichao Zhang, Christian Forster, Davide Scaramuzza. Active Exposure Control for Robust Visual Odometry in HDR Environments. ICRA, 2017. [bibtex](./doc/bib/Zhang17icra.bib)
* *Ceres-based optimization backend*: Stefan Leutenegger, Simon Lynen, Michael Bosse, Roland Siegwart, Paul Timothy Furgale. Keyframe-based visual–inertial odometry using nonlinear optimization. IJRR, 2015. [bibtex](./doc/bib/Leutenegger15ijrr.bib)
* *Global map powered by iSAM2*: Michael Kaess, Hordur Johannsson, Richard Roberts, Viorela Ila, John Leonard, Frank Dellaert. iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree. IJRR, 2012. [bibtex](./doc/bib/Kaess12ijrr.bib)
* *Loop closure*: Dorian Gálvez-López and Juan D. Tardós. Bags of Binary Words for Fast Place Recognition in Image Sequences. TRO, 2012. [bibtex](./doc/bib/Galvez12tro.bib)

Our recent publications that use SVO Pro are:

* *Multiple camera SLAM*: Juichung Kuo, Manasi Muglikar, Zichao Zhang, Davide Scaramuzza. Redesigning SLAM for Arbitrary Multi-Camera Systems. ICRA, 2020. [bibtex](./doc/bib/Kuo20icra.bib)
* *Voxel map for visual SLAM*: Manasi Muglikar, Zichao Zhang, Davide Scaramuzza. Voxel Map for Visual SLAM. ICRA, 2020. [bibtex](./doc/bib/Muglikar20icra.bib)
* *Tight-coupling of global positional measurements into VIO*: Giovanni Cioffi, Davide Scaramuzza. Tightly-coupled Fusion of Global Positional Measurements in Optimization-based Visual-Inertial Odometry. IROS, 2020. [bibtex](./doc/bib/Cioffi20iros.bib)

## Install

The code has been tested on

* Ubuntu 18.04 with ROS Melodic
* Ubuntu 20.04 with ROS Noetic

### Install dependences

Install [catkin tools](https://catkin-tools.readthedocs.io/en/latest/installing.html) and [vcstools](https://github.com/dirk-thomas/vcstool) if you haven't done so before. Depending on your operating system, run
```sh
# For Ubuntu 18.04 + Melodic
sudo apt-get install python-catkin-tools python-vcstool
```
or
```sh
# For Ubuntu 20.04 + Noetic
sudo apt-get install python3-catkin-tools python3-vcstool python3-osrf-pycommon
```
Install system dependencies and dependencies for Ceres Solver
```sh
# system dep.
sudo apt-get install libglew-dev libopencv-dev libyaml-cpp-dev 
# Ceres dep.
sudo apt-get install libblas-dev liblapack-dev libsuitesparse-dev
```

### Clone and compile
Create a workspace and clone the code (`ROS-DISTRO`=`melodic`/`noetic`):
```sh
mkdir svo_ws && cd svo_ws
# see below for the reason for specifying the eigen path
catkin config --init --mkdirs --extend /opt/ros/<ROS-DISTRO> --cmake-args -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
cd src
git clone git@github.com:uzh-rpg/rpg_svo_pro_open.git
vcs-import < ./rpg_svo_pro_open/dependencies.yaml
touch minkindr/minkindr_python/CATKIN_IGNORE
# vocabulary for place recognition
cd rpg_svo_pro_open/svo_online_loopclosing/vocabularies && ./download_voc.sh
cd ../../..
```
There are two types of builds that you can proceed from here
1. Build without the global map (**front-end + sliding window back-end + loop closure/pose graph**)

   ```sh
   catkin build
   ```

   

2. Build with the global map using iSAM2  (**all functionalities**)

    First, enable the global map feature

    ```sh
    rm rpg_svo_pro_open/svo_global_map/CATKIN_IGNORE
    ```
    and in `svo_cmake/cmake/Modules/SvoSetup.cmake`

    ```cmake
    SET(USE_GLOBAL_MAP TRUE)
    ```

    Second, clone GTSAM

    ```sh
    git clone --branch 4.0.3 git@github.com:borglab/gtsam.git
    ```

    and modify GTSAM compilation flags a bit:

    ```cmake
    # 1. gtsam/CMakelists.txt: use system Eigen
    -option(GTSAM_USE_SYSTEM_EIGEN "Find and use system-installed Eigen. If 'off', use the one bundled with GTSAM" OFF)
    +option(GTSAM_USE_SYSTEM_EIGEN "Find and use system-installed Eigen. If 'off', use the one bundled with GTSAM" ON)
    # 2. gtsam/cmake/GtsamBuildTypes: disable avx instruction set
    # below the line `list_append_cache(GTSAM_COMPILE_OPTIONS_PUBLIC "-march=native")`
    list_append_cache(GTSAM_COMPILE_OPTIONS_PUBLIC "-mno-avx")
    ```

    > Using the same version of Eigen helps avoid [memory issues](https://github.com/ethz-asl/eigen_catkin/wiki/Eigen-Memory-Issues). Disabling `avx`  instruction set also helps with some segment faults in our experience (this can be however OS and hardware dependent).

    And finally build the whole workspace

    ```sh
    # building GTSAM may take a while
    catkin build
    ```

## Instructions

* Get started: running the pipeline
    * [The visual front-end](./doc/frontend/visual_frontend.md)
    * [Visual-inertial odometry](./doc/vio.md)
    * [VIO + global map](./doc/global_map.md)
* [Benchmarking](./doc/benchmarking.md)
* [Camera and sensor calibration](./doc/calibration.md)
* [Known issues and possible improvements](./doc/known_issues_and_improvements.md)

## Troubleshooting

0. **Weird building issues after some tinkering**. It is recommend to 
   
    * clean your workspace (`catkin clean --all` at the workspace root)  and rebuild your workspace (`catkin build`)
    * or `catkin build --force-cmake`
    
    after your have made changes to CMake files (`CMakeLists.txt` or `*.cmake`) to make sure the changes take effect.
    
    <details>
      <summary>Longer explanation</summary>
      Catkin tools can detect changes in CMake files and re-build affected files only. But since we are working with a multi-package project, some changes may not be detected as desired. For example, changing the building flags in `svo_cmake/cmake/Modules/SvoSetup.cmake` will affect all the packages but the re-compiling may not be done automatically (since the files in each package are not changed). Also, we need to keep the linking (e.g., library version) and compiling flags consistent across different packages. Therefore, unless you are familiar with how the compilation works out, it is the safest to re-build the whole workspace. `catkin build --force-cmake` should also work in most cases.
    </details>
    
1. **Compiling/linking error related to OpenCV**: find `find_package(OpenCV REQUIRED)` in the `CMakeLists.txt` files in each package (in `rpg_common`, `svo_ros`, `svo_direct`, `vikit/vikit_common` and `svo_online_loopclosing`) and replace it with 

   ```cmake
   # Ubuntu 18.04 + Melodic
   find_package(OpenCV 3 REQUIRED)
   # Ubuntu 20.04 + Noetic
   find_package(OpenCV 4 REQUIRED)
   ```

   <details>
     <summary>Longer explanation</summary>
     First, ROS is built against OpenCV 3 on Ubuntu 18.04 and OpenCV 4 on Ubuntu 20.04. It is desired to keep the OpenCV version linked in SVO consistent with the ROS one, since in `svo_ros` we need to link everything with ROS. Second, The original `CMakeLists.txt` files will work fine if you only have the default OpenCV installed. But if you have some customized version of OpenCV installed (e.g., from source), it is recommended to explicitly specify the version of OpenCV that should be used (=the version ROS uses) as mentione above.
   </details>

2. **Visualization issues with the PointCloud2**: Using `Points` to visualize `PointCloud2` in RVIZ seems to be [problematic](https://github.com/ros-visualization/rviz/issues/1508) in Ubuntu 20.04. We use other visualization types instead of `Points` per default. However, it is good to be aware of this if you want to customize the visualization.

3. **Pipeline crashes with loop closure enabled**: If the pipeline crashes calling `svo::loadVoc()`, did you forgot to download the vocabulary files as mentioned above?

    ```sh
    cd rpg_svo_pro_open/svo_online_loopclosing/vocabularies && ./download_voc.sh
    ```

4. **Inconsistent Eigen versions during compilation**: The same Eigen should be used across the whole project (which should be system Eigen, since we are also using ROS). Check whether `eigen_catkin` and `gtsam` find the same version of Eigen:

    ```sh
    # for eigen_catkin
    catkin build eigen_catkin --force-cmake --verbose
    # for gtsam
    catkin build gtsam --force-cmake --verbose
    ```
    
    <details>
         <summary>Longer explanation</summary>
    One common pitfall of using Eigen in your projects is have different libraries compiled against different Eigen versions. For SVO, eigen_catkin (https://github.com/ethz-asl/eigen_catkin) is used to keep the Eigen version same, which should be the system one (under /usr/include) on 18.04 and 20.04. For GTSAM, system Eigen is found via a cumstomized cmake file (https://github.com/borglab/gtsam/blob/develop/cmake/FindEigen3.cmake#L66). It searches for `/usr/local/include` first, which may contain Eigen versions that are manually installed. Therefore, we explicitly specifies `EIGEN_INCLUDE_PATH` when configuring the workspace to force GTSAM to find system Eigen. If you still encounter inconsistent Eigen versions, the first thing to check is whether different versions of Eigen are still used.
    </details>

## Acknowledgement

Thanks to Simon Klenk, Manasi Muglikar, Giovanni Cioffi and Javier Hidalgo-Carrió for their valuable help and comments for the open source code.

The work is made possible thanks to the efforts of many contributors from RPG. Apart from the authors listed in the above papers, Titus Cieslewski and Henri Rebecq made significant contributions to the visual front-end. Jeffrey Delmerico made great efforts to apply SVO on different real robots, which in turn helped improve the pipeline. Many PhD and master students and lab engineers have also contributed to the code.

The Ceres-based optimization back-end is based on code developed at [Zurich-eye](https://www.wysszurich.uzh.ch/projects/completed-projects/zurich-eye), a spin-off from RPG. Jonathan Huber is the main contributor that integrated the back-end with SVO. Kunal Shrivastava (now CEO of [SUIND](https://suind.com/)) developed the loop closure module during his semester project and internship at RPG. The integration of the iSAM2-based global map was developed by Zichao Zhang.

We would like to thank our collaborators at [Prophesee](https://www.prophesee.ai/) for pointing out several bugs in the visual front-end. Part of the code was developed during a funded project with [Huawei](https://www.huawei.com/en/).
