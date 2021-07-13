#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import argparse
import os
import numpy as np

def calGTLengthFromBag(bag_fn, gt_topic):
    bag = rosbag.Bag(bag_fn)
    gt_msgs = []
    for topic, msg, t in bag.read_messages(topics=[gt_topic]):
        gt_msgs.append(msg)

    prev_seq = -1
    gt_positions = []
    for msg_i in gt_msgs:
        seq_i = msg_i.header.seq
        assert seq_i > prev_seq
        prev_seq = seq_i
        pos_i = msg_i.pose.position
        gt_positions.append([pos_i.x, pos_i.y, pos_i.z])
    assert len(gt_positions) > 0, "Found no groundtruth messages."

    gt_positions = np.array(gt_positions)
    print("Collecting {0} groundtruth positions.".format(gt_positions.shape[0]))

    gt_dist = np.diff(gt_positions, axis=0)

    gt_len = np.sum([np.linalg.norm(v) for v in gt_dist])

    return gt_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate the trajectory length for bag files under a directory.")
    parser.add_argument('dir', help='directory that contains the bags to calculate the groundtruth')
    parser.add_argument('gt_topic', help='groundtruth topic')

    args = parser.parse_args()

    print("Going to analyze bags under {0} for groundtruth topic {1}.".format(args.dir, args.gt_topic))

    bag_fns = [x for x in os.listdir(args.dir) if x.endswith('.bag')]
    bag_fns.sort()
    bag_fullfns = [os.path.join(args.dir, v) for v in bag_fns]
    print("Found bag files {0}.".format(bag_fullfns))

    gt_len = []
    for bag_i in bag_fullfns:
        print("Analyzing {0}".format(bag_i))
        gt_len_i = calGTLengthFromBag(bag_i, args.gt_topic)
        gt_len.append(gt_len_i)
        print(">>> GT {0} meters.".format(gt_len_i))
