#!/usr/bin/env python2

import os
import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='swap stamped imu values (Acc and Gyr)')
    parser.add_argument('org_imu',
                        help="orignal imu values as N x 7 matrix")
    args = parser.parse_args()

    assert os.path.exists(args.org_imu)
    outdir = os.path.dirname(args.org_imu)
    outfn = os.path.join(outdir,
                         'swapped_' + os.path.basename(args.org_imu))

    print("Going to swap Acc and Gyr from {0} to {1}".format(
        args.org_imu, outfn))

    org_values = np.loadtxt(args.org_imu)
    assert org_values.shape[1] == 7
    print("Read {0} IMU values.".format(org_values.shape[0]))

    swapped_values = np.zeros(org_values.shape)

    swapped_values[:, 0] = org_values[:, 0]
    swapped_values[:, 1:4] = org_values[:, 4:7]
    swapped_values[:, 4:7] = org_values[:, 1:4]

    np.savetxt(outfn, swapped_values)
