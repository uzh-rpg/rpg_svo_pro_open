/*
 * camera_pinhole_test.cpp
 *
 *  Created on: Oct 26, 2012
 *      Author: cforster
 */

#include <string>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <Eigen/Core>
#include <vikit/math_utils.h>
#include <vikit/timer.h>
#include <vikit/sample.h>
#include <vikit/pinhole_camera.h>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
  vk::PinholeCamera cam(752, 480, 414.5, 414.2, 348.8, 240.0);
  Matrix4d T_ref_cur(Matrix4d::Identity());
  T_ref_cur(0,3) = 0.5;
  Vector2d u_cur(200, 300);
  Vector3d f_cur(cam.cam2world(u_cur));
  double depth_cur = 2.0;
  Vector3d f_ref(vk::project3d(T_ref_cur*vk::unproject3d(f_cur*depth_cur)));
  double depth_ref = f_ref.norm();
  Vector2d u_ref(cam.world2cam(f_ref));
  double z_ref, z_cur;
  vk::depthFromTriangulationExact(T_ref_cur.topLeftCorner<3,3>(), T_ref_cur.topRightCorner<3,1>(), f_ref, f_cur, z_ref, z_cur);
  printf("depth = %f, triangulated depth = %f\n", depth_cur, z_cur);
  printf("depth = %f, triangulated depth = %f\n", depth_ref, z_ref);

  return 0;
}
