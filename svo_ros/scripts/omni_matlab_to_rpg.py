#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import argparse
import os
import time
import numpy as np
from collections import OrderedDict


class UnsortableList(list):
    def sort(self, *args, **kwargs):
        pass


class UnsortableOrderedDict(OrderedDict):
    def items(self, *args, **kwargs):
        return UnsortableList(OrderedDict.items(self, *args, **kwargs))


def strToFloatList(raw_str, delimiter=' '):
    str_list = raw_str.split(' ')
    return [float(s) for s in str_list if s != '']


def parseMatlabFormat(matlab_filename):
    f = open(matlab_filename, 'r')
    with open(matlab_filename) as f:
        lines = f.readlines()
    filtered = [l[:-1] for l in lines if l[0] != '\n' and l[0] != '#']

    pol = strToFloatList(filtered[0])[1:]
    invpol = strToFloatList(filtered[1])[1:]
    invpol += [0.0] * (12 - len(invpol))
    center = strToFloatList(filtered[2])[::-1]
    affine = strToFloatList(filtered[3])
    img_size = strToFloatList(filtered[4])[::-1]

    print('parsed parameters:')
    print('polynomial: {0}\ninverse polynomial: {1}'.format(pol, invpol))
    print('image size: {0}\nimage center: {1}'.format(img_size, center))
    print('affine correction: {0}'.format(affine))

    return {'pol': pol, 'invpol': invpol,
            'center': center, 'affine': affine, 'img_size': img_size}


def dictToYaml(cam_params, serial_num, matlab_filename):
    cams = []
    cam = UnsortableOrderedDict()
    cam['camera'] = UnsortableOrderedDict()
    cam['camera']['distortion'] = UnsortableOrderedDict()
    cam['camera']['distortion']['parameters'] = UnsortableOrderedDict()
    cam['camera']['distortion']['parameters']['cols'] = 1
    cam['camera']['distortion']['parameters']['rows'] = 0
    cam['camera']['distortion']['parameters']['data'] = []
    cam['camera']['distortion']['type'] = 'none'
    cam['camera']['image_height'] = int(cam_params['img_size'][1])
    cam['camera']['image_width'] = int(cam_params['img_size'][0])
    cam['camera']['intrinsics'] = UnsortableOrderedDict()
    cam['camera']['intrinsics']['cols'] = 1
    cam['camera']['intrinsics']['rows'] = 24
    cam['camera']['intrinsics']['data'] = \
        cam_params['pol'] + cam_params['center'] + \
        cam_params['affine'] + cam_params['invpol'] + [0.0, 0.0]
    cam['camera']['label'] = 'cam0'
    cam['camera']['line-delay-nanoseconds'] = 0
    cam['camera']['type'] = 'omni'
    cam['camera']['mask'] = str(serial_num)+'_fisheye_mask.png'

    cam['T_B_C'] = UnsortableOrderedDict()
    T_B_C = np.eye(4, dtype=float)
    cam['T_B_C']['cols'] = T_B_C.shape[1]
    cam['T_B_C']['rows'] = T_B_C.shape[0]
    cam['T_B_C']['data'] = T_B_C.flatten().tolist()
    cam['serial_no'] = int(serial_num)
    cam['calib_date'] = time.strftime("%d/%m/%Y")
    cams.append(cam)

    S = UnsortableOrderedDict()
    S['cameras'] = cams
    S['label'] = matlab_filename

    return S


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert the format of matlabtoolbox to rpg format')
    parser.add_argument('matlab_filename', help='matlab output file')
    parser.add_argument('serial_num', help='camera serial number')

    args = parser.parse_args()

    if not os.path.isfile(args.matlab_filename):
        print('please input the correct file name for the matlab format')
        exit(-1)

    rpg_filename = 'bluefox_' + args.serial_num + '_fisheye.yaml'
    print('converting {0} to {1}'.format(args.matlab_filename, rpg_filename))

    cam_params = parseMatlabFormat(args.matlab_filename)
    cam_yaml = dictToYaml(cam_params, args.serial_num, args.matlab_filename)

    yaml.add_representer(UnsortableOrderedDict,
                         yaml.representer.SafeRepresenter.represent_dict)

    f = open(rpg_filename, 'w')
    f.write(yaml.dump(cam_yaml, default_flow_style=None))
    f.close()

    print('Finished. Do not forget to create the mask before usage.')
