#!/usr/bin/env python2

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import init, Fore

import svo_benchmarking.transformations as tfs
import svo_benchmarking.tfs_utils as tu
import svo_benchmarking.results_loading as res_load
import svo_benchmarking.dataset_loading as data_load

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)
FORMAT = '.png'

init(autoreset=True)


def load_est_gt_mc(trace_dir, mc_indices):
    es_p_mc = []
    es_q_mc = []
    gt_p_mc = []
    gt_q_mc = []

    for i in mc_indices:
        _, _, es_p, es_q, _, gt_p, gt_q =\
            res_load.load_traj_est_gt_matched(trace_dir, i)
        es_p_mc.append(es_p)
        es_q_mc.append(es_q)
        gt_p_mc.append(gt_p)
        gt_q_mc.append(gt_q)

    return es_p_mc, es_q_mc, gt_p_mc, gt_q_mc


def rotation_matrices_to_eulers(rot_mats, eu='rzxy'):
    rot_ypr = []
    for v in rot_mats:
        homo_v = np.identity(4)
        homo_v[0:3, 0:3] = v
        y, p, r = tfs.euler_from_matrix(homo_v, eu)
        rot_ypr.append([np.rad2deg(y), np.rad2deg(p), np.rad2deg(r)])
    rot_ypr = np.array(rot_ypr)
    return rot_ypr


def align_est_and_calculate_ATE(es_p, es_q, gt_p, gt_q):
    es_T = [tu.get_rigid_body_trafo(q, t) for q, t in zip(es_q, es_p)]
    gt_T = [tu.get_rigid_body_trafo(q, t) for q, t in zip(gt_q, gt_p)]
    T_est0 = es_T[0]
    T_gt0 = gt_T[0]
    align_T = np.dot(T_gt0, np.linalg.inv(T_est0))
    print("The alignment transformation is:\n{0}.".format(align_T))
    aligned_es_T = [np.dot(align_T, v) for v in es_T]
    aligned_es_p = np.array([v[0:3, 3] for v in aligned_es_T])
    aligned_es_q = np.array([tu.dcm2quat(v[0:3, 0:3]) for v in aligned_es_T])
    ate_dist = [np.linalg.norm(v) for v in aligned_es_p - gt_p]

    return aligned_es_p, aligned_es_q, ate_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    analyze the pure IMU integration when the visual tracking fails
    ''')
    parser.add_argument(
        'trace_dir', help='directory containing the trace files')
    parser.add_argument(
        '--start_id', type=int, default=0, help='start of analysis')
    parser.add_argument(
        '--end_id', type=int, default=-1, help='end of analysis (-1: end)')
    parser.add_argument(
        '--duration', type=float, default=-1, help='end of analysis (-1: end)')
    args = parser.parse_args()

    assert os.path.exists(args.trace_dir)
    print("Going to analyze trace in {0} from {1} to {2}".format(
        args.trace_dir, args.start_id, args.end_id))
    print("The time duration is set to {0} seconds".format(args.duration))
    unique_folder_name = '_'.join(
        [str(args.start_id), str(args.end_id), str(args.duration),
         'analyze_imu_integration_dyn'])
    save_dir = os.path.join(args.trace_dir, "eval_res", unique_folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("The results/plots will be written to {0}.".format(save_dir))

    n_trials = res_load.get_num_monte_carlo(args.trace_dir)
    mc_indices = np.arange(n_trials)
    print("Number of monte-carlo runs is {0}.".format(n_trials))

    print(Fore.GREEN + "> Loading estimate...")
    print("monte-carlo results...")
    es_p_mc, es_q_mc, gt_p_mc, gt_q_mc = load_est_gt_mc(args.trace_dir,
                                                        mc_indices)
    print("Single trial...")
    es_id, es_t_sec, es_p, es_q, _, gt_p, gt_q =\
        res_load.load_traj_est_gt_matched(args.trace_dir, 0)
    es_t_sec = es_t_sec - es_t_sec[0]
    _, _, es_v, es_gyr_bias, es_acc_bias =\
        res_load.load_speed_bias_estimate(args.trace_dir, 0)
    n_est = es_id.shape[0]
    assert es_v.shape[0] == n_est

    print("Get estimates within desired range...")
    valid_es_entries = res_load.get_desired_indices(es_id, es_t_sec,
                                                    args.start_id, args.end_id,
                                                    args.duration)
    es_p_mc = [v[valid_es_entries] for v in es_p_mc]
    es_q_mc = [v[valid_es_entries] for v in es_q_mc]
    gt_p_mc = [v[valid_es_entries] for v in gt_p_mc]
    gt_q_mc = [v[valid_es_entries] for v in gt_q_mc]
    es_id = es_id[valid_es_entries]
    es_t_sec = es_t_sec[valid_es_entries]
    es_p = es_p[valid_es_entries]
    es_q = es_q[valid_es_entries]
    es_R = [tfs.quaternion_matrix(v)[0:3, 0:3] for v in es_q]

    es_v = es_v[valid_es_entries]
    es_gyr_bias = es_gyr_bias[valid_es_entries]
    es_acc_bias = es_acc_bias[valid_es_entries]
    gt_p = gt_p[valid_es_entries]
    gt_q = gt_q[valid_es_entries]
    gt_R = [tfs.quaternion_matrix(v)[0:3, 0:3] for v in gt_q]

    print("Extact estimate range from {0} to {1}".format(es_id[0],
                                                         es_id[-1]))

    print(Fore.GREEN + "> Loading dataset...")
    # load and get corresponding imu measurements
    imu_id, imu_time, gyr, acc = data_load.load_imu_meas(args.trace_dir)
    imu_time = imu_time - imu_time[0]
    print("Loaded {0} IMU measurements.".format(imu_id.shape[0]))
    valid_imu_entries = []
    for i, v in enumerate(imu_time):
        if v < es_t_sec[0]:
            continue
        if v > es_t_sec[-1]:
            break
        valid_imu_entries.append(i)
    imu_id = imu_id[valid_imu_entries]
    imu_time = imu_time[valid_imu_entries]
    gyr = gyr[valid_imu_entries]
    acc = acc[valid_imu_entries]
    print("Extract {0} IMU measurements in the estimate range.".format(
        imu_id.shape[0]))

    print(Fore.GREEN + '> Analyzing and plotting...')

    print("Monte-Carlo results...")
    aligned_es_p_mc = []
    ate_mc = []
    for ep, eq, gp, gq in zip(es_p_mc, es_q_mc, gt_p_mc, gt_q_mc):
        aligned_ep, _, ate = align_est_and_calculate_ATE(ep, eq, gp, gq)
        aligned_es_p_mc.append(aligned_ep)
        ate_mc.append(ate)
    fig = plt.figure()
    axx = fig.add_subplot(411, xlabel='t (sec)', ylabel='x (m)}')
    axy = fig.add_subplot(412, xlabel='t (sec)', ylabel='y (m)')
    axz = fig.add_subplot(413, xlabel='t (sec)', ylabel='z (m)')
    axate = fig.add_subplot(414, xlabel='t (sec)', ylabel='ATE (m)')
    for aligned_ep, ate in zip(aligned_es_p_mc, ate_mc):
        axx.plot(es_t_sec, aligned_ep[:, 0], alpha=0.7)
        axy.plot(es_t_sec, aligned_ep[:, 1], alpha=0.7)
        axz.plot(es_t_sec, aligned_ep[:, 2], alpha=0.7)
        axate.plot(es_t_sec, ate, 'b')
        axate.set_ylim([0, 0.2])
    mean_end_err = np.mean(np.array([v[-1] for v in ate_mc]))
    med_end_err = np.median(np.array([v[-1] for v in ate_mc]))
    min_end_err = np.amin(np.array([v[-1] for v in ate_mc]))
    max_end_err = np.amax(np.array([v[-1] for v in ate_mc]))
    np.savetxt(save_dir+'/statistics_end_drift.txt', np.array([mean_end_err,
                                                               med_end_err,
                                                               min_end_err,
                                                               max_end_err]))
    axx.plot(es_t_sec, gt_p[:, 0], 'k', label='gt')
    plt.legend()
    axy.plot(es_t_sec, gt_p[:, 1], 'k', label='gt')
    plt.legend()
    axz.plot(es_t_sec, gt_p[:, 2], 'k', label='gt')
    plt.legend()
    fig.savefig(save_dir+'/mc_aligned_position_est_gt'+FORMAT)

    print("Position estimate and groundtruth for one trial...")
    fig = plt.figure()
    ax = fig.add_subplot(321, xlabel='t (sec)', ylabel='est x (m)')
    ax.plot(es_t_sec, es_p[:, 0])
    ax = fig.add_subplot(323, xlabel='t (sec)', ylabel='est y (m)')
    ax.plot(es_t_sec, es_p[:, 1])
    ax = fig.add_subplot(325, xlabel='t (sec)', ylabel='est z (m)')
    ax.plot(es_t_sec, es_p[:, 2])
    ax = fig.add_subplot(322, xlabel='t (sec)', ylabel='gt x (m)')
    ax.plot(es_t_sec, gt_p[:, 0])
    ax = fig.add_subplot(324, xlabel='t (sec)', ylabel='gt y (m)')
    ax.plot(es_t_sec, gt_p[:, 1])
    ax = fig.add_subplot(326, xlabel='t (sec)', ylabel='gt z (m)')
    ax.plot(es_t_sec, gt_p[:, 2])
    fig.savefig(save_dir+'/position_estimate_gt'+FORMAT)

    print("Orientation estimation and groundtruth...")
    es_ypr = rotation_matrices_to_eulers(es_R)
    gt_ypr = rotation_matrices_to_eulers(gt_R)
    fig = plt.figure()
    ax = fig.add_subplot(321, xlabel='t (sec)', ylabel='est yaw (deg)')
    ax.plot(es_t_sec, es_ypr[:, 0])
    ax = fig.add_subplot(323, xlabel='t (sec)', ylabel='est pitch (deg)')
    ax.plot(es_t_sec, es_ypr[:, 1])
    ax = fig.add_subplot(325, xlabel='t (sec)', ylabel='est roll (deg)')
    ax.plot(es_t_sec, es_ypr[:, 2])
    ax = fig.add_subplot(322, xlabel='t (sec)', ylabel='gt yaw (deg)')
    ax.plot(es_t_sec, gt_ypr[:, 0])
    ax = fig.add_subplot(324, xlabel='t (sec)', ylabel='gt pitch (deg)')
    ax.plot(es_t_sec, gt_ypr[:, 1])
    ax = fig.add_subplot(326, xlabel='t (sec)', ylabel='gt roll (deg)')
    ax.plot(es_t_sec, gt_ypr[:, 2])
    fig.savefig(save_dir+'/orientation_estimate_gt'+FORMAT)

    print("ATE for one trial...")
    aligned_est_p, _, ate_dist = align_est_and_calculate_ATE(es_p, es_q,
                                                             gt_p, gt_q)
    fig = plt.figure()
    ax = fig.add_subplot(411, xlabel='t (sec)', ylabel='x (m)}')
    ax.plot(es_t_sec, aligned_est_p[:, 0], label='est')
    ax.plot(es_t_sec, gt_p[:, 0], label='gt')
    ax = fig.add_subplot(412, xlabel='t (sec)', ylabel='y (m)')
    ax.plot(es_t_sec, aligned_est_p[:, 1], label='est')
    ax.plot(es_t_sec, gt_p[:, 1], label='gt')
    ax = fig.add_subplot(413, xlabel='t (sec)', ylabel='z (m)')
    ax.plot(es_t_sec, aligned_est_p[:, 2], label='est')
    ax.plot(es_t_sec, gt_p[:, 2], label='gt')
    plt.legend()
    ax = fig.add_subplot(414, xlabel='t (sec)', ylabel='ATE (m)')
    ax.plot(es_t_sec, ate_dist)
    fig.savefig(save_dir+'/aligned_position_est_gt'+FORMAT)

    print("Velocity estimate for one trial...")
    es_v_norm = np.array([np.linalg.norm(v) for v in es_v])
    fig = plt.figure()
    ax = fig.add_subplot(411, xlabel='t (sec)', ylabel='x (m/s)')
    ax.plot(es_t_sec, es_v[:, 0])
    ax = fig.add_subplot(412, xlabel='t (sec)', ylabel='y (m/s)')
    ax.plot(es_t_sec, es_v[:, 1])
    ax = fig.add_subplot(413, xlabel='t (sec)', ylabel='z (m/s)')
    ax.plot(es_t_sec, es_v[:, 2])
    ax = fig.add_subplot(414, xlabel='t (sec)', ylabel='vel. norm')
    ax.plot(es_t_sec, es_v_norm)
    fig.savefig(save_dir+'/vel_estimate'+FORMAT)

    print("Bias estimate for one trial...")
    sig_gyr_bias = np.zeros((3, ))
    sig_acc_bias = np.zeros((3, ))
    for i in range(3):
        sig_gyr_bias[i] = np.std(es_gyr_bias[:, i])
        sig_acc_bias[i] = np.std(es_acc_bias[:, i])

    fig = plt.figure()
    ax = fig.add_subplot(211, xlabel='t (sec)', ylabel='gyr bias')
    ax.plot(es_t_sec, es_gyr_bias[:, 0], label=r'$b_g$ x')
    ax.plot(es_t_sec, es_gyr_bias[:, 1], label=r'$b_g$ y')
    ax.plot(es_t_sec, es_gyr_bias[:, 2], label=r'$b_g$ z')
    plt.legend()
    ax = fig.add_subplot(212, xlabel='t (sec)', ylabel='acc bias')
    ax.plot(es_t_sec, es_acc_bias[:, 0], label=r'$b_a$ x')
    ax.plot(es_t_sec, es_acc_bias[:, 1], label=r'$b_a$ y')
    ax.plot(es_t_sec, es_acc_bias[:, 2], label=r'$b_a$ z')
    plt.legend()
    fig.savefig(save_dir+'/bias_estimate'+FORMAT)

    print("Raw IMU measurements...")
    fig = plt.figure()
    ax = fig.add_subplot(311, xlabel='t (sec)', ylabel='gyr x')
    ax.plot(imu_time, gyr[:, 0])
    ax = fig.add_subplot(312, xlabel='t (sec)', ylabel='gyr y')
    ax.plot(imu_time, gyr[:, 1])
    ax = fig.add_subplot(313, xlabel='t (sec)', ylabel='gyr z')
    ax.plot(imu_time, gyr[:, 2])
    fig.savefig(save_dir+'/raw_gyr'+FORMAT)

    fig = plt.figure()
    ax = fig.add_subplot(311, xlabel='t (sec)', ylabel='acc x')
    ax.plot(imu_time, acc[:, 0])
    ax = fig.add_subplot(312, xlabel='t (sec)', ylabel='acc y')
    ax.plot(imu_time, acc[:, 1])
    ax = fig.add_subplot(313, xlabel='t (sec)', ylabel='acc z')
    ax.plot(imu_time, acc[:, 2])
    fig.savefig(save_dir+'/raw_acc'+FORMAT)
