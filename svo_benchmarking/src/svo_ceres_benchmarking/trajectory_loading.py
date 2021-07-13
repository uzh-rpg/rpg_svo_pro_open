#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import numpy as np
import trajectory_utils


def read_nanosecond_timestamps_from_csv_file(filename, col=0, delimiter=','):
    return np.genfromtxt(filename, usecols=(col),
                         delimiter=delimiter, dtype=np.longlong, skip_header=1)


def load_hand_eye_calib(params):
    print('loading hand-eye-calib')
    if 'T_sensor_trackable' in params:
        T_cm_quat = np.array([params['T_sensor_trackable']['qx'],
                              params['T_sensor_trackable']['qy'],
                              params['T_sensor_trackable']['qz'],
                              params['T_sensor_trackable']['qw']])
        T_cm_tran = np.array([params['T_sensor_trackable']['tx'],
                              params['T_sensor_trackable']['ty'],
                              params['T_sensor_trackable']['tz']])
        T_cm = trajectory_utils.get_rigid_body_trafo(T_cm_quat, T_cm_tran)
    else:
        T_cm = np.eye(4)
    return T_cm


def load_dataset_csv(results_dir):
    print('loading csv dataset in '+results_dir)

    filename_es = os.path.join(results_dir, 'traj_estimate.txt')
    filename_gt = os.path.join(results_dir, 'groundtruth.csv')

    matches = np.genfromtxt(os.path.join(
        results_dir, 'groundtruth_matches.csv'), dtype=np.longlong,
        delimiter=',', skip_header=1)
    matches = dict([(row[0], row[1]) for row in matches])  # { es -> gt }

    keys_es = read_nanosecond_timestamps_from_csv_file(
        filename_es, col=0, delimiter=' ')
    data_es = np.genfromtxt(filename_es, delimiter=' ',
                            dtype=np.float64, skip_header=1)[:, 1:]

    keys_gt = np.genfromtxt(filename_gt, usecols=(
        0), delimiter=',', dtype=np.longlong, skip_header=1)
    data_gt = np.genfromtxt(filename_gt, delimiter=',',
                            dtype=np.float64, skip_header=1)[:, 1:]

    groundtruth = dict([(keys_gt[i], data_gt[i, :])
                        for i in range(len(keys_gt))])

    p_es = []
    p_gt = []
    q_es = []
    q_gt = []
    t_gt = []
    t_es = []
    for i in range(len(keys_es)):
        if keys_es[i] in matches:
            t_es_val = keys_es[i]
            t_gt_val = matches[t_es_val]
            gt_data = groundtruth[t_gt_val]
            p_gt.append(gt_data[:3])
            # quaternion order we want is x y z w, file has x y z w
            tmp = gt_data[3:7]
            q_gt.append([tmp[1], tmp[2], tmp[3], tmp[0]])
            p_es.append(data_es[i, 0:3])
            q_es.append(data_es[i, 3:7])
            t_es.append(t_es_val)
            t_gt.append(t_gt_val)
    p_es = np.array(p_es)
    p_gt = np.array(p_gt)
    q_es = np.array(q_es)
    q_gt = np.array(q_gt)
    t_gt = np.array(t_gt)
    t_es = np.array(t_es)

    return t_es, p_es, q_es, t_gt, p_gt, q_gt


def load_synthetic_dataset(results_dir):
    gt_trajectory = np.loadtxt(os.path.join(results_dir, 'groundtruth.txt'))
    es_trajectory = np.loadtxt(os.path.join(results_dir, 'traj_estimate.txt'))
    N = es_trajectory.shape[0]
    p_es = es_trajectory[:, 1:4]
    q_es = es_trajectory[:, 4:8]

    map_index_slot = dict()
    for i, index in enumerate(gt_trajectory[:, 0]):
        map_index_slot[int(index)] = i
    gt_indices = np.zeros(N)
    for i in range(N):
        gt_indices[i] = map_index_slot[int(es_trajectory[i, 0])]

    p_gt = gt_trajectory[np.int32(gt_indices), 1:4]
    q_gt = gt_trajectory[np.int32(gt_indices), 4:8]
    t_gt = gt_indices
    return t_gt, p_es, q_es, t_gt, p_gt, q_gt
