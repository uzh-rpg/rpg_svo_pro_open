#!/usr/bin/python2

import rosbag
import argparse
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

def extract(bagfile, pose_topic, out_filename, cam_id):
    n = 0
    f = open(out_filename, 'w')
    f.write('# id timestamp image_name\n')
    cv_bridge = CvBridge()
    extract_every_nth_image = 1
    #max_imgs = 400

    with rosbag.Bag(bagfile, 'r') as bag:
        for (topic, msg, ts) in bag.read_messages(topics=str(pose_topic)):
            if np.mod(n, extract_every_nth_image) == 0:
                try:
                    img = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                except CvBridgeError, e:
                    print e
                    
                ts = msg.header.stamp.to_sec()
                image_name = 'image_'+str(cam_id)+'_'+str(n)+'.png'
                f.write('%d %.12f img/%s \n' % (n, ts, image_name))
                cv2.imwrite(image_name, img)
            n += 1
            #if n > max_imgs:
            #    break
            
    print('wrote ' + str(n) + ' images messages to the file: ' + out_filename)
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Extracts Images messages from bagfile.
    ''')
    parser.add_argument('bag', help='Bagfile')
    parser.add_argument('topic', help='Topic')
    parser.add_argument('cam_id', help='Camera-Id')
    args = parser.parse_args()
    print('Extract images from bag '+args.bag+' in topic ' + args.topic)
    extract(args.bag, args.topic, 'images.txt', args.cam_id)
