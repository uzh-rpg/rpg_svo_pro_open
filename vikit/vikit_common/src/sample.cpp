/*
 * sample.cpp
 *
 *  Created on: May 14, 2013
 *      Author: cforster
 */

#include <vikit/sample.h>

namespace vk {

std::ranlux24 Sample::gen_real;
std::mt19937 Sample::gen_int;

void Sample::setTimeBasedSeed()
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  gen_real = std::ranlux24(seed);
  gen_int = std::mt19937(seed);
}

int Sample::uniform(int from, int to)
{
  std::uniform_int_distribution<int> distribution(from, to);
  return distribution(gen_int);
}

double Sample::uniform()
{
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  return distribution(gen_real);
}

double Sample::gaussian(double stddev)
{
  std::normal_distribution<double> distribution(0.0, stddev);
  return distribution(gen_real);
}

Eigen::Vector3d Sample::randomDirection3D()
{
  // equal-area projection according to:
  // https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
  const double z = Sample::uniform()*2.0-1.0;
  const double t = Sample::uniform()*2.0*M_PI;
  const double r = std::sqrt(1.0 - z*z);
  const double x = r*std::cos(t);
  const double y = r*std::sin(t);
  return Eigen::Vector3d(x,y,z);
}

Eigen::Vector2d Sample::randomDirection2D()
{
  const double theta = Sample::uniform()*2.0*M_PI;
  return Eigen::Vector2d(std::cos(theta), std::sin(theta));
}

} // namespace vk
