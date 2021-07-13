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
#include <vikit/pinhole_camera.h>
#include <vikit/pinhole_equidistant_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>
#include <vikit/sample.h>

using namespace std;
using namespace Eigen;


void testTiming(vk::AbstractCamera::Ptr cam)
{
  Vector3d xyz;
  Vector2d px(320.64, 253.54);
  vk::Timer t;
  t.start();
  for(size_t i=0; i<1000; ++i)
  {
    xyz = cam->cam2world(px);
  }
  t.stop();
  cout << "Time unproject = " << t.getTime()*1000 << "ms" << endl;

  t.start();
  for(size_t i=0; i<1000; ++i)
  {
    px = cam->world2cam(xyz);
  }
  t.stop();
  cout << "Time project = " << t.getTime()*1000 << "ms" << endl;
}

void testAccuracy(vk::AbstractCamera::Ptr cam)
{
  double error = 0.0;
  vk::Timer t;
  for(size_t i=0; i<1000; ++i)
  {
    Vector2d px(1.0/100.0 * vk::Sample::uniform(0, cam->width()*100), 
                1.0/100.0 * vk::Sample::uniform(0, cam->height()*100));
    Vector3d xyz = cam->cam2world(px);
    Vector2d px2 = cam->world2cam(xyz);
    error += (px-px2).norm();
  }
  cout << "Reprojection error = " << error << " (took " << t.stop()*1000 << "ms)" << endl;
}

int main(int argc, char **argv)
{
  Eigen::Quaterniond q(vk::rpy2dcm(Eigen::Vector3d(-3.1415, 0, 0)));
  std::cout << "qx = " << q.x() << std::endl;
  std::cout << "qy = " << q.y() << std::endl;
  std::cout << "qz = " << q.z() << std::endl;
  std::cout << "qw = " << q.w() << std::endl;

  vk::AbstractCamera::Ptr cam_pinhole(
        new vk::PinholeCamera(
          640, 480,
          323.725240539365, 323.53310403533,
          336.407165453746, 235.018271952295,
          -0.258617082313663, 0.0623042373522829, 0.000445967802619555, -0.000269839440982019));

  vk::AbstractCamera::Ptr cam_atan(
        new vk::ATANCamera(752, 480, 0.511496, 0.802603, 0.530199, 0.496011, 0.934092));

  vk::AbstractCamera::Ptr cam_equidistant(
        new vk::PinholeEquidistantCamera(
          752, 480,
          463.34128261574796, 461.90887721986877,
          361.5557340721321, 231.13558880965206,
          -0.0027973061697674074, 0.024145501123661265, -0.04304211254137983, 0.031185314072573474));

  printf("\nPINHOLE CAMERA:\n");
  testTiming(cam_pinhole);
  testAccuracy(cam_pinhole);

  printf("\nATAN CAMERA:\n");
  testTiming(cam_atan);
  testAccuracy(cam_atan);

  printf("\nPINHOLE EQUIDISTANT CAMERA:\n");
  testTiming(cam_equidistant);
  testAccuracy(cam_equidistant);

  return 0;
}
