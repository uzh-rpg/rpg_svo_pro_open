#!/usr/bin/python

import math
import numpy as np

_EPS = np.finfo(float).eps * 4.0


   
    
        
    
def quat2dcm(quaternion):
    """Returns direct cosine matrix from quaternion (Hamiltonian, [x y z w])
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])),
        dtype=np.float64)


def dcm2quat(matrix_3x3):
    """Return quaternion (Hamiltonian, [x y z w]) from rotation matrix.
    This algorithm comes from  "Quaternion Calculus and Fast Animation",
    Ken Shoemake, 1987 SIGGRAPH course notes
    (from Eigen)
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix_3x3, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > 0.0:
        t = math.sqrt(t+1.0)
        q[3] = 0.5*t
        t = 0.5/t
        q[0] = (M[2, 1] - M[1, 2])*t
        q[1] = (M[0, 2] - M[2, 0])*t
        q[2] = (M[1, 0] - M[0, 1])*t
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = math.sqrt(M[i, i] - M[j, j] - M[k, k] + 1.0)
        q[i] = 0.5*t
        t = 0.5/t
        q[3] = (M[k, j] - M[j, k])*t
        q[j] = (M[i, j] + M[j, i])*t
        q[k] = (M[k, i] + M[i, k])*t
    return q
    
def dcm2rpy(matrix):
    """Return roll pitch yaw from matrix
    """
    R = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    rpy = np.empty((3), dtype=np.float64)
    rpy[1] = math.atan2(-R[2,0], math.sqrt( R[0,0]**2 + R[1,0]**2 ))
        
    if np.abs(rpy[1] - math.pi/2.0 ) < 0.00001:
        rpy[2] = 0;
        rpy[0] = -math.atan2( R[0,1], R[1,1] )
    else:
        if np.abs(rpy[1] + math.pi/2.0 ) < 0.00001:
            rpy[2] = 0;
            rpy[0] = -math.atan2( R[0,1], R[1,1] );
        else:
            rpy[2] = math.atan2( R[1,0]/math.cos(rpy[1]), R[0,0]/math.cos(rpy[1]) )
            rpy[0] = math.atan2( R[2,1]/math.cos(rpy[1]), R[2,2]/math.cos(rpy[1]) )
            
    return rpy
    
def quat2rpy(quaternion):
    """Return roll pitch yaw from quaternion
    """
    return dcm2rpy(quat2dcm(quaternion))
    
def quat_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)


def quat_conjugate(quaternion):
    """Return conjugate of quaternion.
    """
    return np.array((-quaternion[0], -quaternion[1],
                        -quaternion[2], quaternion[3]), dtype=np.float64)

def quat_inverse(quaternion):
    """Return inverse of quaternion.
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


    
    
