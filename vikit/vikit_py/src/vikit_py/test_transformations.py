#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
import transformations as tf

class TestTransformations(unittest.TestCase):
    
    def test_skew(self):
        for i in range(100):
            v = tf.random_vector(3)
            R = tf.skew(v)
            npt.assert_array_almost_equal(v, tf.unskew(R))
            
    def test_first_order_rotation(self):
        for i in range(100):
            rotvec = tf.random_vector(3)*np.pi*2.0
            R = tf.first_order_rotation(rotvec)
            npt.assert_array_almost_equal(R, np.identity(3)+tf.skew(rotvec))
            
    def test_axis_angle(self):
        for i in range(100):
            axis = tf.random_vector(3)
            axis = axis/np.linalg.norm(axis)
            theta = np.random.rand()*np.pi
            R = tf.axis_angle(axis, theta)
            npt.assert_almost_equal(np.linalg.det(R), 1.0)
    
    def test_compare_axis_angle(self):
        for i in range(100):
            axis = tf.random_vector(3)
            axis = axis/np.linalg.norm(axis)
            theta = np.random.rand()*np.pi
            R_1 = tf.axis_angle(axis, theta)
            R_2 = tf.rotation_matrix(theta, axis)[:3,:3]
            npt.assert_array_almost_equal(R_1, R_2)
    
    def test_right_jacobian_so3(self):
        # test first-order approximation
        for i in range(100):
            rotvec = tf.random_vector(3)*np.pi
            perturbation = tf.random_vector(3) / 1000.0
            R = tf.expmap_so3(rotvec+perturbation)
            Jr = tf.right_jacobian_so3(rotvec)
            R_approx = np.dot(tf.expmap_so3(rotvec), tf.expmap_so3(np.dot(Jr, perturbation)))
            npt.assert_array_almost_equal(R, R_approx)
            
    def test_logmap_so3(self):
        for i in range(100):
            axis = tf.random_vector(3)
            axis = axis/np.linalg.norm(axis)
            # can't have angles larger than np.pi otherwise the logmap is a bijection
            theta = np.random.rand()*np.pi*0.999 
            R = tf.expmap_so3(axis*theta)
            rotvec = tf.logmap_so3(R)
            npt.assert_array_almost_equal(axis*theta, rotvec)
            
    def test_S_inv_eulerZYX_body_deriv(self):
        
        for i in range(100):
            euler_coordinates = tf.random_vector(3)
            omega = tf.random_vector(3) / 10.0
            perturbation = tf.random_vector(3) / 1000.0
            e = np.dot(tf.S_inv_eulerZYX_body(euler_coordinates), omega)
            e_perturbed = np.dot(tf.S_inv_eulerZYX_body(euler_coordinates + perturbation), omega)
            J = tf.S_inv_eulerZYX_body_deriv(euler_coordinates, omega)                              
            e_perturbed_predicted = e + np.dot(J, perturbation)
            npt.assert_array_almost_equal(e_perturbed, e_perturbed_predicted)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTransformations)
    unittest.TextTestRunner(verbosity=2).run(suite)