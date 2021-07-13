#!/usr/bin/python

import rosbag
import argparse

def extract(bagfile, pose_topic, out_filename):
    n = 0
    f = open(out_filename, 'w')
    f.write('# id timestamp tx ty tz qx qy qz qz\n')    
    with rosbag.Bag(bagfile, 'r') as bag:
        for (topic, msg, ts) in bag.read_messages(topics=str(pose_topic)):
            if False:
                # Geometry Message PoseWithCovarianceStamped
                f.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (n, 
                         msg.header.stamp.to_sec(),
                         msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                         msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
            elif True:
                # Geometry Message PoseStamped
                f.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (n,
                         msg.header.stamp.to_sec(),
                         msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                         msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w))
            elif False:
                # Geometry Message PointStamped
                f.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (n,
                         msg.header.stamp.to_sec(),
                         msg.point.x, msg.point.y, msg.point.z,
                         0, 0, 0, 1))
            else:
                # Transform Stamped
                f.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' % 
                        (n,
                         msg.header.stamp.to_sec(),
                         msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z,
                         msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w))
            n += 1
    print('wrote ' + str(n) + ' pose messages to the file: ' + out_filename)
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Extracts IMU messages from bagfile.
    ''')
    parser.add_argument('bag', help='Bagfile')
    parser.add_argument('topic', help='Topic')
    args = parser.parse_args()    
    out_filename = 'groundtruth.txt'
    print('Extract pose from bag '+args.bag+' in topic ' + args.topic)
    print('Saving to file '+out_filename)
    extract(args.bag, args.topic, out_filename)
