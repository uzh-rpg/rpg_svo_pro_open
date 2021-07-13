#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Analyze the timing for VIO''')
    parser.add_argument(
        'result_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
    args = parser.parse_args()

    abs_res_dir = os.path.abspath(args.result_dir)
    t_fn = os.path.join(abs_res_dir, 'trace_backend.csv')
    assert os.path.exists(t_fn)

    trace = {}
    keys = []
    with open(t_fn, 'r') as f:
        reader = csv.reader(f)
        n_row = 0
        for row in reader:
            if n_row == 0:
                for k in row:
                    trace[k] = []
                    keys.append(k)
            else:
                for k, v in zip(keys, row):
                    trace[k].append(float(v))
            n_row += 1

    stats = {}
    print("Statistics:")
    for k, v in trace.iteritems():
        d = trace[k] = np.array(trace[k])
        stats[k] = {}
        print("- {0}: {1}, {2}, {3} - {4}".format(k, np.mean(d), np.median(d),
                                                  np.min(d), np.max(d)))
        stats[k]['mean'] = np.mean(d)
        stats[k]['median'] = np.median(d)
        stats[k]['min'] = np.min(d)
        stats[k]['max'] = np.max(d)

    fig = plt.figure(figsize=(12, 2.5))
    ax = fig.add_subplot(111, ylabel='Time (s)')
    ax.plot(trace['tot_time'], label='Frame processing time')
    ax.set_title('Mean: {0:.4f}; Min: {1:.4f}; Max: {2:.4f}'.format(
        stats['tot_time']['mean'], stats['tot_time']['min'],
        stats['tot_time']['max']))
    plt.legend()
    fig.savefig(abs_res_dir+'/'+'total_timing'+FORMAT, bbox_inches="tight")
    plt.close(fig)
