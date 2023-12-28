# Docker deployment instructions

**NOTE**: (For users running this Docker on a local machine) As of 2023/12/1, Docker Desktop interferes with x11 forwarding since it uses a VM, and thus all x11 forwarding goes to the VM instead of the local machine. This means you'll need to stop / uninstall Docker Desktop (and potentially [uninstall Docker Engine](https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine) also) and [install Docker Engine](https://docs.docker.com/engine/install/ubuntu/) instead.

## Instructions

1a. Build the docker image (might need to use `sudo` for the following docker commands depending on your docker install)
```sh
cd rpg_svo_pro_open
docker build docker/. -t vio:latest
```

1b. 

2. Download a test dataset to run
```sh
mkdir test_data
wget http://rpg.ifi.uzh.ch/datasets/airground_rig_s3_2013-03-18_21-38-48.bag -P test_data/
```

3. On your local machine, make sure you have x11 and openGL properly installed and functioning:
```sh
xeyes       # tests x11
glxgears    # tests openGL
```
You should see an animated window from `glxgears` [like this](http://www.subdude-site.com/WebPages_Local/RefInfo/Computer/Linux/LinuxPerformance/3Dperformance/decopics/glxgears_screenshot_306x326.jpg), and a mouse responsive window from `xeyes` [like this](https://blog.dhampir.no/wp-content/uploads/2012/02/xeyes.png). 

4. Make sure that forwarding is activated for docker:
```
sh
xhost + local:docker
```

5. Launch the docker image. The `--volume=$HOME/.Xauthority:/root/.Xauthority:rw` part is necessary for X11, and the `--volume /dev:/dev` part is necessary for openGL. 
```sh
docker run --net=host --env="DISPLAY" --volume=$HOME/.Xauthority:/root/.Xauthority:rw --volume /dev:/dev --volume $PWD/test_data:/test_data -it vio:latest
```

6. Now that you are inside the docker container, test out the GUI forwarding. Should get the same results as you did when you tested earlier on your local machine.
```sh
xeyes       # tests x11
glxgears    # tests openGL
```

7. If the tests above are successful, then we can try and run the test Visual Odometry example:
```sh
source svo_ws/devel/setup.bash
roslaunch svo_ros run_from_bag.launch cam_name:=svo_test_pinhole &
rosbag play test_data/airground_rig_s3_2013-03-18_21-38-48.bag
```
