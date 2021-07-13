#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rosbag
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="analyze stationary standard deviation")
    parser.add_argument('bag_name', help='bag contains the output pose topic')
    parser.add_argument('topic_name', help='pose topic in the bag')
    parser.add_argument('start', type=float, help='percentage of starting point')
    parser.add_argument('end', type=float, help='percentage of end point')

    args = parser.parse_args()

    assert args.start > 0.0 and args.start < 1.0
    assert args.end > 0.0 and args.end < 1.0
    assert args.end > args.start

    assert os.path.exists(args.bag_name), "File does not exist."
    print("Going to analyze {0} in {1}...".format(args.topic_name, args.bag_name))
    save_prefix = os.path.basename(args.bag_name).split('.')[0]\
                  + '_'.join(args.topic_name.split('/')) + '_'
    print("Save preifx is {0}".format(save_prefix))

    bag = rosbag.Bag(args.bag_name)
    pose_msgs = []
    for topic, msg, t in bag.read_messages(topics=[args.topic_name]):
        pose_msgs.append(msg)

    n_total = len(pose_msgs)
    print("Collected {0} pose messages.".format(len(pose_msgs)))

    n_start = int(n_total * args.start)
    n_end = int(n_total * args.end)
    print("Going to analyze from {0} to {1}...".format(n_start, n_end))

    positions = []
    times = []
    prev_seq = -1
    for i in range(n_total):
        msg_i = pose_msgs[i]
        seq_i = msg_i.header.seq
        assert seq_i > prev_seq, "{0} vs {1}".format(seq_i, prev_seq)
        prev_seq = seq_i
        pos_i = pose_msgs[i].pose.pose.position
        positions.append([pos_i.x, pos_i.y, pos_i.z])
        times.append(msg_i.header.stamp.to_sec())

    static_pos = np.array(positions[n_start:n_end])
    static_times = np.array(times[n_start:n_end])
    n_static = static_pos.shape[0]
    print("Extracted {0} static positions.".format(n_static))

    # calculate statistics
    mean_pos = np.mean(static_pos, axis=0)
    std_pos = np.zeros((3, ))
    for i in range(3):
        std_pos[i] = np.std(static_pos[:, i])
    names = ['x', 'y', 'z']

    print("The average position is {0}.".format(mean_pos))
    print("The standard deviation of each axis is {0}.".format(std_pos))
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    for i, ax in enumerate(axes[0:3]):
        mean_i = np.ones((static_pos.shape[0])) * mean_pos[i]
        ax.plot(static_times, mean_i, ls='--', color='k',
                label='average '+names[i])
        ax.plot(static_times, static_pos[:, i], ls='-', color='b',
                label='estimated '+names[i])
        ax.set_ylabel('Position-'+names[i]+' (meter)')
        ax.set_title('std: '+ str(std_pos[i]) + ' meters')
        ax.legend()


    offset_pos = static_pos - mean_pos
    dist_pos = np.array([np.linalg.norm(v) for v in offset_pos])
    std_pos_vec = np.sqrt((np.sum(dist_pos**2)) / (n_static - 1))
    print("The standard deviation of position is {0}".format(std_pos_vec))
    zero_dist = np.zeros((static_pos.shape[0]))
    ax = axes[3]
    ax.plot(static_times, zero_dist, ls= '--', color='k',
            label='zero')
    ax.plot(static_times, dist_pos, ls='-',
            color='b', label='Offset from average position')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Offset (meter)')
    ax.set_title('Position std: ' + str(std_pos_vec) + ' meters')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_prefix+'xyz' + '.pdf')

    # plt.show()


