#!/usr/bin/env python2

import os
import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prepend id to values in a text file')
    parser.add_argument('org_file',
                        help="file contains original values"
                        "the first two columns")
    args = parser.parse_args()

    assert os.path.exists(args.org_file)
    outdir = os.path.dirname(args.org_file)
    outfn = os.path.join(outdir,
                         'id_' + os.path.basename(args.org_file))

    print("Going to prepend id to {0} and write to {1}".format(
        args.org_file, outfn))

    org_content = []
    with open(args.org_file) as f:
        org_content = f.readlines()
    print("Found {0} data".format(len(org_content)))

    id_content = []
    cnt = 0
    for l in org_content:
        if not l.startswith('#'):
            l = str(cnt) + ' ' + l
            cnt += 1
        else:
            if cnt == 0:
                l = l.replace('#', '# id')
        id_content.append(l)

    with open(outfn, 'w') as f:
        f.writelines(id_content)

