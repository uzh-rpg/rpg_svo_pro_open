#!/usr/bin/python

import yaml
import sys
import argparse
import numpy as np

from collections import OrderedDict

class UnsortableList(list):
    def sort(self, *args, **kwargs):
        pass

class UnsortableOrderedDict(OrderedDict):
    def items(self, *args, **kwargs):
        return UnsortableList(OrderedDict.items(self, *args, **kwargs))

yaml.add_representer(UnsortableOrderedDict, yaml.representer.SafeRepresenter.represent_dict)
parser = argparse.ArgumentParser(description='Convert calibration from Kalibr to SVO format.')
parser.add_argument('--kalibr', help='kalibr camchain-imu yaml file.')
parser.add_argument('--imu', help='imu parameters yaml file.')
args = parser.parse_args(sys.argv[1:])

if args.kalibr is None:
    sys.exit('You must provide yaml files for the Kalibr format.')
has_imu = True
if args.imu is None:
    print('No IMU specification provided, will process camera parameters only.')
    has_imu = False

# cameras
with open(args.kalibr, 'r') as f:
    K = yaml.load(f)
cams = []
for c in K:
    cam = UnsortableOrderedDict()
    cam['camera'] = UnsortableOrderedDict()
    cam['camera']['distortion'] = UnsortableOrderedDict()
    cam['camera']['distortion']['parameters'] = UnsortableOrderedDict()
    cam['camera']['distortion']['parameters']['cols'] = 1
    cam['camera']['distortion']['parameters']['rows'] = len(K[c]['distortion_coeffs'])
    cam['camera']['distortion']['parameters']['data'] = K[c]['distortion_coeffs']
    cam['camera']['distortion']['type'] = K[c]['distortion_model'] 
    cam['camera']['image_height'] = K[c]['resolution'][1]
    cam['camera']['image_width'] = K[c]['resolution'][0]
    cam['camera']['intrinsics'] = UnsortableOrderedDict()
    cam['camera']['intrinsics']['cols'] = 1
    cam['camera']['intrinsics']['rows'] = len(K[c]['intrinsics'])
    cam['camera']['intrinsics']['data'] = K[c]['intrinsics']
    cam['camera']['label'] = c
    cam['camera']['line-delay-nanoseconds'] = 0
    cam['camera']['type'] = K[c]['camera_model']

    cam['T_B_C'] = UnsortableOrderedDict()
    T_B_C = np.eye(4)
    if has_imu:
        T_C_B = np.reshape(np.array(K[c]['T_cam_imu']), (4, 4))
        T_B_C = np.linalg.inv(T_C_B)
    else:
        print('No IMU is specified, so T_B_C is set to identity.')
    cam['T_B_C']['cols'] = T_B_C.shape[1]
    cam['T_B_C']['rows'] = T_B_C.shape[0]
    cam['T_B_C']['data'] = T_B_C.flatten().tolist()

    cam['serial_no'] = 0
    cam['calib_date'] = 0
    cam['description'] = K[c]['rostopic']
    cams.append(cam)
S = UnsortableOrderedDict()
S['cameras'] = cams
S['label'] = args.kalibr

# IMU
if has_imu:
    with open(args.imu, 'r') as f:
        I = yaml.load(f)
    Sip = UnsortableOrderedDict()
    Sip['imu_params'] = UnsortableOrderedDict()
    Sip['imu_params']['delay_imu_cam'] = 0.0
    Sip['imu_params']['max_imu_delta_t'] = 0.01
    Sip['imu_params']['acc_max'] = 176.0
    Sip['imu_params']['omega_max'] = 7.8
    Sip['imu_params']['sigma_omega_c'] = I['gyroscope_noise_density']
    Sip['imu_params']['sigma_acc_c'] = I['accelerometer_noise_density']
    Sip['imu_params']['sigma_omega_bias_c'] = I['gyroscope_random_walk']
    Sip['imu_params']['sigma_acc_bias_c'] = I['accelerometer_random_walk']
    Sip['imu_params']['sigma_integration'] = 0.0
    Sip['imu_params']['g'] = 9.8082
    Sip['imu_params']['imu_rate'] = I['update_rate']

    Sii = UnsortableOrderedDict()
    Sii['imu_initialization'] = UnsortableOrderedDict()
    Sii['imu_initialization']['velocity'] = [0.0, 0.0, 0.0]
    Sii['imu_initialization']['omega_bias'] = [0.0, 0.0, 0.0]
    Sii['imu_initialization']['acc_bias'] = [0.0, 0.0, 0.0]
    Sii['imu_initialization']['velocity_sigma'] = 1.0
    Sii['imu_initialization']['omega_bias_sigma'] = I['gyroscope_noise_density']
    Sii['imu_initialization']['acc_bias_sigma'] = I['accelerometer_noise_density']

output_file = 'svo_' + args.kalibr
print('Writing to {0}.'.format(output_file))
f = open(output_file, 'w')
f.write(yaml.dump(S, default_flow_style=None) )
if has_imu:
    f.write('\n')
    f.write( yaml.dump(Sip, default_flow_style=False) )
    f.write('\n')
    f.write( yaml.dump(Sii, default_flow_style=None) )
f.close()
print('Done.')
