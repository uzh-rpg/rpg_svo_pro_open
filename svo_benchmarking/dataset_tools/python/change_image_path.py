#!/usr/bin/env python2

import os
import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='change the image path in images.txt')
    parser.add_argument('org_img_list',
                        help="file contains original image file values")
    parser.add_argument('new_path',
                        help="new path that is to be prepended "
                        "to the image name")
    args = parser.parse_args()

    assert os.path.exists(args.org_img_list)
    outdir = os.path.dirname(args.org_img_list)
    outfn = os.path.join(outdir,
                         'changed_' + os.path.basename(args.org_img_list))

    print("Going to prepend {0} to {1} and write to {2}".format(
        args.new_path, args.org_img_list, outfn))

    org_content = []
    with open(args.org_img_list) as f:
        org_content = f.readlines()
    org_content = [x.strip().split(' ') for x in org_content]
    print("Found {0} data".format(len(org_content)))

    changed_content = []
    cnt = 0
    for l in org_content:
        if l[0].startswith('#'):
            continue
        old_fn = os.path.basename(l[-1])
        l[-1] = os.path.join(args.new_path, old_fn)
        changed_content.append(l)

    changed_content = [' '.join(x) + '\n' for x in changed_content]
    with open(outfn, 'w') as f:
        f.writelines(changed_content)

