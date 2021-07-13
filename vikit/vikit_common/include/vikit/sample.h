#ifndef VIKIT_SAMPLE_H_
#define VIKIT_SAMPLE_H_

#include <random>
#include <chrono>
#include <Eigen/Core>

namespace vk {

class Sample
{
public:
  static void setTimeBasedSeed();
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
  static std::ranlux24 gen_real;
  static std::mt19937 gen_int;
  static Eigen::Vector3d randomDirection3D();
  static Eigen::Vector2d randomDirection2D();
};

} // namespace vk

#endif // VIKIT_SAMPLE_H_
