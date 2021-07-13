#!/usr/bin/env python2

import os
import yaml

import numpy as np


def get_num_monte_carlo(trace_dir):
    exp_param_fn = os.path.join(trace_dir, "experiment.yaml")
    assert os.path.exists(exp_param_fn),\
        "{0} does not exist".format(exp_param_fn)
    exp_params = yaml.load(open(exp_param_fn, 'r'))
    assert 'num_monte_carlo_runs' in exp_params
    return exp_params['num_monte_carlo_runs']


def get_desired_indices(ids, times_sec, start_id, max_end_id, max_duration):
    valid_entries = []
    t_start = None
    for i, v in enumerate(ids):
        if v < start_id:
            continue
        elif t_start is None:
            t_start = times_sec[i]
        if max_end_id > 0 and v >= max_end_id:
            break
        if max_duration > 0 and t_start is not None:
            t_cur = times_sec[i]
            if t_cur - t_start > max_duration:
                break

        valid_entries.append(i)
    return valid_entries


def load_traj_est(results_dir, mc_idx=None):
    '''
    format: id time_sec x y z qx qy qz qw
    '''
    print('loading trajectory estimate in '+results_dir)

    if mc_idx is not None:
        data_es = np.loadtxt(
            os.path.join(results_dir, 'traj_estimate' + str(mc_idx) + '.txt'))
    else:
        data_es = np.loadtxt(os.path.join(results_dir, 'traj_estimate.txt'))
    assert data_es.shape[1] == 9

    es_id = []
    es_t_sec = []
    p_es = []
    q_es = []

    for row_es in data_es:
        es_id.append(int(row_es[0]))
        es_t_sec.append(row_es[1])
        p_es.append(row_es[2:5])
        q_es.append(row_es[5:9])

    return np.array(es_id), np.array(es_t_sec),\
        np.array(p_es), np.array(q_es)


def load_speed_bias_estimate(results_dir, mc_idx=None):
    '''
    format: id time_sec vx vy vz gyr_bx gyr_by gyr_bz acc_bx acc_by acc_bz
    '''
    print("loading speed and bias estimate in " + results_dir)
    if mc_idx is not None:
        data_es = np.loadtxt(
            os.path.join(results_dir, 'speed_bias_estimate' +
                         str(mc_idx) + '.txt'))
    else:
        data_es = np.loadtxt(os.path.join(results_dir,
                                          'speed_bias_estimate.txt'))
    assert data_es.shape[1] == 11

    es_id = []
    es_t_sec = []
    v_b = []
    gyr_bias = []
    acc_bias = []

    for row_es in data_es:
        es_id.append(int(row_es[0]))
        es_t_sec.append(row_es[1])
        v_b.append(row_es[2:5])
        gyr_bias.append(row_es[5:8])
        acc_bias.append(row_es[8:11])

    return np.array(es_id), np.array(es_t_sec), np.array(v_b),\
        np.array(gyr_bias), np.array(acc_bias)


def load_status(results_dir, mc_idx=None):
    '''
    format: id time_sec stage tracking_quality update_result
    '''
    if mc_idx is not None:
        abs_fn = os.path.join(results_dir, 'status' + str(mc_idx) + '.txt')
    else:
        abs_fn = os.path.join(results_dir, 'status.txt')

    print("Loading status from {0}.".format(abs_fn))

    with open(abs_fn) as f:
        contents = f.readlines()

    contents = [x.strip().split(' ') 
                for x in contents if not x.startswith('#')]
    es_id = []
    es_t_sec = []
    stage = []
    tracking_quality = []
    update_result = []
    for l in contents:
        es_id.append(l[0])
        es_t_sec.append(l[1])
        stage.append(l[2])
        tracking_quality.append(l[3])
        update_result.append(l[4])

    return es_id, es_t_sec, stage, tracking_quality, update_result


def load_traj_est_gt_matched(results_dir, mc_idx=None):
    '''
    format: id time_sec x y z qx qy qz qw
    '''
    print('loading matched trajectory and groundtruth in '+results_dir)

    if mc_idx is not None:
        data_es = np.loadtxt(
            os.path.join(results_dir, 'traj_estimate' + str(mc_idx) + '.txt'))
    else:
        data_es = np.loadtxt(os.path.join(results_dir, 'traj_estimate.txt'))
    data_gt = np.loadtxt(os.path.join(results_dir, 'groundtruth.txt'))
    data_gt = dict([(int(row[0]), row[1:]) for row in data_gt])
    matches = np.loadtxt(os.path.join(results_dir, 'groundtruth_matches.txt'))
    matches = dict([(int(row[0]), int(row[1])) for row in matches])
    p_es = []
    p_gt = []
    q_es = []
    q_gt = []
    t_gt = []
    t_es = []
    id_es = []
    for row_es in data_es:
        image_id = int(row_es[0])
        if image_id in matches:
            groundtruth_id = matches[image_id]
            gt_single = data_gt[groundtruth_id]
            p_es.append(row_es[2:5])
            p_gt.append(gt_single[1:4])
            q_es.append(row_es[5:9])
            q_gt.append(gt_single[4:8])
            t_gt.append(gt_single[1])
            t_es.append(row_es[1])
            id_es.append(image_id)
    p_es = np.array(p_es)
    p_gt = np.array(p_gt)
    q_es = np.array(q_es)
    q_gt = np.array(q_gt)
    t_gt = np.array(t_gt)
    t_es = np.array(t_es)
    id_es = np.array(id_es)

    return id_es, t_es, p_es, q_es, t_gt, p_gt, q_gt
