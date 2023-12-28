# ubuntu 20.04
FROM osrf/ros:noetic-desktop-full-focal

RUN apt update -y &&\
    apt install -y python3-catkin-tools python3-vcstool python3-osrf-pycommon 

# System dep
RUN apt install -y libglew-dev libopencv-dev libyaml-cpp-dev 

# Ceres dep.
RUN apt install -y libblas-dev liblapack-dev libsuitesparse-dev

RUN apt install -y git wget libeigen3-dev libtool

# Get rpg_svo_pro_open
RUN mkdir svo_ws &&\
    cd svo_ws &&\
    catkin config --init --mkdirs --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 &&\
    cd src &&\
    git clone --branch docker-updates https://github.com/escondido-maps/rpg_svo_pro_open.git

# Get dependencies
RUN cd svo_ws/src &&\
    vcs-import < ./rpg_svo_pro_open/dependencies.yaml &&\
    touch minkindr/minkindr_python/CATKIN_IGNORE &&\
    # vocabulary for place recognition
    cd ./rpg_svo_pro_open/svo_online_loopclosing/vocabularies &&\
    ./download_voc.sh &&\
    cd ../../..

# Build rpg_svo_pro_open with global map functionality
RUN cd svo_ws/src &&\
    rm -f rpg_svo_pro_open/svo_global_map/CATKIN_IGNORE &&\
    git clone https://github.com/escondido-maps/gtsam.git &&\
    cd gtsam &&\
    git checkout rpg-svo-pro-version

RUN mkdir ~/.ssh &&\
    chmod 700 ~/.ssh

RUN cd svo_ws/src &&\
    catkin build --force-cmake -j8

# OPTIONAL: Install test x11 utils (xclock, xeyes, etc)
RUN apt install -y x11-apps mesa-utils