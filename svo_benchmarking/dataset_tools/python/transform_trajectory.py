#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Wed Apr 18 11:16:38 2018

@author: Jonathan Huber

Transform textfile containing a trajectory using a constant transformation
"""

import os
import numpy as np
import argparse
import math

from svo_benchmarking.tfs_utils import quat2dcm, dcm2quat

if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(description='''
    Analyze trajectories
    ''')
    parser.add_argument(
        'trajectory',
        help='File containing [id timestamp tx ty tz qx qy qz qw]')
    parser.add_argument(
        'transformation',
        help='File with 4x4 transformation matrix, Comma separated rows')
    args = parser.parse_args()
    assert os.path.exists(
        args.trajectory), "{0} not exist.".format(args.trajectory)
    assert os.path.exists(args.transformation), "{0} not exist.".format(
        args.transformation)

    # load the file
    f = open(args.trajectory, "r")
    lines = f.readlines()
    id_and_timestamp = []
    positions = []
    quats = []
    i = 0
    for x in lines:
        if(i == 0):
            header = x
            i = i+1
            continue
        id_and_timestamp.append(x.split(' ')[0:2])
        positions.append(np.array(x.split(' ')[2:5], dtype=float))
        quats.append(np.array(x.split(' ')[5:9], dtype=float))
    f.close()

    T_HE = np.loadtxt(args.transformation)
    N = len(positions)
    assert(N == len(quats))
    transformed_positions = np.zeros([N, 3])
    transformed_quats = np.zeros([N, 4])
    print('Transforming ', N, ' trajectory poses')
    for i in range(N):
        T_old = np.eye(4)
        T_old[0:3, 0:3] = quat2dcm(quats[i])
        T_old[0:3, 3] = positions[i]
        T_new = T_old.dot(T_HE)
        transformed_positions[i, :] = T_new[0:3, 3]
        transformed_quats[i, :] = dcm2quat(T_new[0:3, 0:3])

    file_lines = []
    file_lines.append(header)
    for i in range(N):
        file_lines.append(''.join([str(id_and_timestamp[i][0]), ' ',
                                   str(id_and_timestamp[i][1]), ' ',
                                   str(transformed_positions[i, 0]), ' ',
                                   str(transformed_positions[i, 1]), ' ',
                                   str(transformed_positions[i, 2]), ' ',
                                   str(transformed_quats[i, 0]), ' ',
                                   str(transformed_quats[i, 1]), ' ',
                                   str(transformed_quats[i, 2]), ' ',
                                   str(transformed_quats[i, 3]), '\n']))

    with open('transformed_poses.txt', 'w') as f:
        f.writelines(file_lines)
    print('Wrote transformed poses to file transformed_poses.txt')
