#!/usr/bin/env python2

import os
import numpy as np


def load_imu_meas(trace_dir, fn='imu.txt'):
    abs_fn = os.path.join(trace_dir, fn)
    assert os.path.exists(abs_fn), "{0} does not exist.".format(abs_fn)

    imu_id = []
    imu_time_sec = []
    gyr_meas = []
    acc_meas = []

    imu_data = np.loadtxt(abs_fn)
    assert imu_data.shape[1] == 8, "Wrong data format."
    for row in imu_data:
        imu_id.append(int(row[0]))
        imu_time_sec.append(row[1])
        gyr_meas.append(row[2:5])
        acc_meas.append(row[5:8])

    return np.array(imu_id), np.array(imu_time_sec),\
        np.array(gyr_meas), np.array(acc_meas)
