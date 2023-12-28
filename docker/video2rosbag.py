import time, sys
import rosbag
import roslib, rospy
import argparse

roslib.load_manifest("sensor_msgs")
# from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
import cv2


def AddVideo2Bag(videopath, bag, compressed):
    """Add video to ROSBAG file"""

    if compressed:
        IMG_TOPIC = "camera/image_raw/compressed"
    else:
        IMG_TOPIC = "camera/image_raw"

    cap = cv2.VideoCapture(videopath)
    cb = CvBridge()

    # Find FPS value
    prop_fps = cap.get(cv2.CAP_PROP_FPS)
    if prop_fps != prop_fps or prop_fps <= 1e-2:
        print("Warning: can't get FPS. Assuming 24.")
        prop_fps = 24
    ret = True
    frame_id = 0

    # Needed to use an actual t0 start time for things to make sense. Not from original code.
    t0 = time.time()

    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        stamp = rospy.rostime.Time.from_sec(t0 + (float(frame_id) / prop_fps))
        frame_id += 1

        if compressed:
            image = cb.cv2_to_compressed_imgmsg(frame)
        else:
            image = cb.cv2_to_imgmsg(frame)

        image.header.stamp = stamp
        image.header.frame_id = "camera"
        bag.write(IMG_TOPIC, image, stamp)

    cap.release()


def AddIMU2Bag(imu_path, bag):
    """Add IMU data from .csv/.txt file to ROSBAG file"""

    IMU_TOPIC = "imu/data"
    imu_file = open(imu_path, "r")

    for x in imu_file.readlines():
        a = x.split(",")

        imu_msg = Imu()

        sec = int(((a[0]).split("."))[0])
        msec = int(((a[0]).split("."))[1])
        nsec = int(msec * 1000)

        imu_msg.header.stamp.secs = sec
        imu_msg.header.stamp.nsecs = nsec
        imu_msg.angular_velocity.x = float(a[1])
        imu_msg.angular_velocity.y = float(a[2])
        imu_msg.angular_velocity.z = float(a[3])
        imu_msg.linear_acceleration.x = float(a[4])
        imu_msg.linear_acceleration.y = float(a[5])
        imu_msg.linear_acceleration.z = float(a[6])

        bag.write(IMU_TOPIC, imu_msg, imu_msg.header.stamp)

    imu_file.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bag", help="Output .bag file path")
    parser.add_argument(
        "-v",
        "--video",
        help="Input video file path (.mov, .mp4, mv4 are tested as valid inputs)",
    )
    parser.add_argument("-i", "--imu", help="Input .txt IMU data file path")
    args = parser.parse_args()

    # Initialize ROS bag
    bag = rosbag.Bag(args.bag, "w")

    # Add video file to bag
    compressed = False
    AddVideo2Bag(args.video, bag, compressed)
    # Optionally, add IMU data to bag
    if args.imu:
        AddIMU2Bag(args.imu, bag)

    bag.close()
