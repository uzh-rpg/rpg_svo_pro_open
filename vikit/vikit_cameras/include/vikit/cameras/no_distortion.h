#pragma once

#include <iostream>

#include <Eigen/Dense>

namespace vk {
namespace cameras {

class NoDistortion 
{
public:
  NoDistortion() = default;
  ~NoDistortion() = default;

  inline void distort(
      double& /*u*/, double& /*v*/) const
  {}

  inline Eigen::Vector2d distort(const Eigen::Vector2d& vector) const {
    return vector;
  }

  inline Eigen::Matrix2d jacobian(const Eigen::Vector2d& /*vector*/) const {
    return Eigen::Matrix2d::Identity();
  }

  inline void undistort(
      double& /*u*/, double& /*v*/) const
  {}

  inline void print(std::ostream& out) const
  {
    out << "  Distortion: No" << std::endl;
  }

  // returns distortion parameters as vector (here empty vector)
  inline Eigen::VectorXd getDistortionParameters() const
  {
    Eigen::VectorXd distortion;
    return distortion;
  }

  static inline NoDistortion createTestDistortion() {
    return NoDistortion();
  }


};


} // namespace cameras
} // namespace vk
