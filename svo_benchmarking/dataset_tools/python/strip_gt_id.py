#!/usr/bin/env python2

import os
import argparse

import numpy as np


def strip_gt_id_and_save(id_stamp_gt_fn):
    outdir = os.path.dirname(id_stamp_gt_fn)
    outfn = os.path.join(outdir,
                         'stamped_' + os.path.basename(id_stamp_gt_fn))

    print("Going to strip id from {0} and write to {1}".format(
        id_stamp_gt_fn, outfn))

    id_stamp_gt = np.loadtxt(id_stamp_gt_fn)
    print("Found {0} groundtruth data".format(id_stamp_gt.shape[0]))

    stamp_gt = []
    for v in id_stamp_gt.tolist():
        stamped_v = v[1:]
        stamp_gt.append(stamped_v)

    np.savetxt(outfn, stamp_gt, header='time x y z qx qy qz qw')
    print("Written {0} stamped groundtruth.".format(len(stamp_gt)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='strip the id from the groundtruth')
    parser.add_argument('id_stamp_gt',
                        help="groundtruth file constains id and stamps as "
                        "the first two columns")
    args = parser.parse_args()

    assert os.path.exists(args.id_stamp_gt)
    strip_gt_id_and_save(args.id_stamp_gt)
