#!/usr/bin/env python2

import argparse
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import init, Fore

import svo_benchmarking.transformations as tfs
import svo_benchmarking.dataset_loading as data_load

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)
FORMAT = '.png'

init(autoreset=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    analyze IMU properties while the sensor is static
    ''')
    parser.add_argument(
        'trace_dir', help='directory containing the trace files')
    parser.add_argument(
        '--start_time_offset', type=int, default=None,
        help='start of analysis')
    parser.add_argument(
        '--end_time_offset', type=int, default=None, help='end of analysis')
    args = parser.parse_args()

    assert os.path.exists(args.trace_dir)
    print("Going to analyze trace in {0} from {1} to {2}".format(
        args.trace_dir, args.start_time_offset, args.end_time_offset))
    unique_folder_name = '_'.join(
        [str(args.start_time_offset), str(args.end_time_offset),
         'analyze_imu_properties'])
    save_dir = os.path.join(args.trace_dir, "eval_res", unique_folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("The results/plots will be written to {0}.".format(save_dir))

    print(Fore.GREEN + "> Loading IMU measurements...")
    # load and get corresponding imu measurements
    imu_id, imu_time, gyr, acc = data_load.load_imu_meas(args.trace_dir)
    imu_time = imu_time - imu_time[0]
    dt = np.mean(np.diff(imu_time))
    sqrt_dt = np.sqrt(dt)
    print("The average dt is {0}.".format(dt))
    print("Loaded {0} IMU measurements of dt {1}.".format(imu_id.shape[0],
                                                          dt))
    valid_imu_entries = []
    for i, v in enumerate(imu_time):
        if args.start_time_offset is not None and\
                v < args.start_time_offset:
            continue
        if args.end_time_offset is not None and\
                v > args.end_time_offset:
            break
        valid_imu_entries.append(i)
    imu_id = imu_id[valid_imu_entries]
    imu_time = imu_time[valid_imu_entries]
    gyr = gyr[valid_imu_entries]
    acc = acc[valid_imu_entries]
    print("Extracted {0} IMU measurements in the specified range.".format(
        imu_id.shape[0]))

    print("Plotting raw IMU measurements...")
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

    print('Calculating gyro bias...')
    bias_gyr_x, bias_gyr_y, bias_gyr_z =\
        np.mean(gyr[:, 0]), np.mean(gyr[:, 1]), np.mean(gyr[:, 2])
    print("Gyro bias are:, {0} {1} {2}".format(bias_gyr_x,
                                               bias_gyr_y, bias_gyr_z))

    print('Calculate noise properties...')
    sig_gyr_x, sig_gyr_y, sig_gyr_z =\
        np.std(gyr[:, 0]), np.std(gyr[:, 1]), np.std(gyr[:, 2])
    sig_acc_x, sig_acc_y, sig_acc_z =\
        np.std(acc[:, 0]), np.std(acc[:, 1]), np.std(acc[:, 2])
    imu_noise = {'sig_gyr_x_c': float(sig_gyr_x*sqrt_dt),
                 'sig_gyr_y_c': float(sig_gyr_y*sqrt_dt),
                 'sig_gyr_z_c': float(sig_gyr_z*sqrt_dt),
                 'sig_acc_x_c': float(sig_acc_x*sqrt_dt),
                 'sig_acc_y_c': float(sig_acc_y*sqrt_dt),
                 'sig_acc_z_c': float(sig_acc_z*sqrt_dt)
                 }
    with open(os.path.join(save_dir, 'noise_properties.yaml'), 'w') as f:
        yaml.dump(imu_noise, f, default_flow_style=False)
    print("Calculated IMU noise properties:\n{0}".format(imu_noise))
    
    sig_gyr = (sig_gyr_x + sig_gyr_y + sig_gyr_z) * sqrt_dt / 3
    sig_acc = (sig_acc_x + sig_acc_y + sig_acc_z) * sqrt_dt / 3
    print("Continuous gyro noise density: {0}".format(sig_gyr))
    print("Continuous acceleromter noise density: {0}".format(sig_acc))

    print("Calculating variance at dt {0}...".format(dt))
    sim_t = np.linspace(0, 5, int(5/0.01))
    sig_angle = sig_gyr * sqrt_dt * sim_t
    sig_pos = sig_acc * sqrt_dt * (1 / np.sqrt(3.0)) * np.power(sim_t, 1.5)
    fig = plt.figure()
    ax = fig.add_subplot(211, xlabel='t (sec)', ylabel='angle sigma')
    ax.plot(sim_t, sig_angle)
    ax = fig.add_subplot(212, xlabel='t (sec)', ylabel='position sigma')
    ax.plot(sim_t, sig_pos)
    fig.savefig(save_dir+'/angel_position_random_walk'+FORMAT)
