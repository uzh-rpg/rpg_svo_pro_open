#pragma once

#include <iostream>
#include <Eigen/Core>
#include <glog/logging.h>

namespace vk {
namespace cameras {

// This class implements the radial and tangential distortion model used by
// OpenCV and ROS. Reference:
// docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
class RadialTangentialDistortion 
{
public:

  RadialTangentialDistortion(
      const double& k1, const double& k2, const double& p1, const double& p2)
    : k1_(k1), k2_(k2), p1_(p1), p2_(p2)
  {}

  RadialTangentialDistortion(
      const Eigen::VectorXd& parameters)
  {
    CHECK(parameters.size() == 4);
    k1_ = parameters(0);
    k2_ = parameters(1);
    p1_ = parameters(2);
    p2_ = parameters(3);
  }

  ~RadialTangentialDistortion() = default;

  inline void distort(double& x, double& y) const
  {
    const double xx = x * x;
    const double yy = y * y;
    const double xy = x * y;
    const double xy2 = 2.0*xy;
    const double r2 = xx + yy;
    const double cdist = (k1_ + k2_ * r2) * r2;
    x += x * cdist + p1_ * xy2 + p2_ * (r2 + 2.0 * xx);
    y += y * cdist + p2_ * xy2 + p1_ * (r2 + 2.0 * yy);
  }

  inline Eigen::Vector2d distort(const Eigen::Vector2d& vector) const {
    const double xx = vector(0) * vector(0);
    const double yy = vector(1) * vector(1);
    const double xy = vector(0) * vector(1);
    const double xy2 = 2.0*xy;
    const double r2 = xx + yy;
    const double cdist = (k1_ + k2_ * r2) * r2;
    return Eigen::Vector2d(
        vector(0) + vector(0) * cdist + p1_ * xy2 + p2_ * (r2 + 2.0 * xx),
        vector(1) + vector(1) * cdist + p2_ * xy2 + p1_ * (r2 + 2.0 * yy));
  }

  //! Input must be already distorted px vector
  inline Eigen::Matrix2d jacobian(const Eigen::Vector2d& px_distorted) const {
    Eigen::Matrix2d J_dist;
    const double xx = px_distorted[0] * px_distorted[0];
    const double yy = px_distorted[1] * px_distorted[1];
    const double xy = px_distorted[0] * px_distorted[1];
    const double r2 = xx + yy;
    const double cdist = (k1_ + k2_ * r2) * r2;

    const double k2_r2_x4 = k2_ * r2 * 4.0;
    const double cdist_p1 = cdist + 1.0;
    J_dist(0,0) = cdist_p1 + k1_ * 2.0 * xx + k2_r2_x4 * xx
        + 2.0 * p1_ * px_distorted[1]  + 6.0 * p2_ * px_distorted[0];
    J_dist(1,1) = cdist_p1 + k1_ * 2.0 * yy + k2_r2_x4 * yy
        + 2.0 * p2_ * px_distorted[0] + 6.0 * p1_ * px_distorted[1] ;
    J_dist(1,0) = 2.0 * k1_ * xy + k2_r2_x4 * xy
        + 2.0 * p1_ * px_distorted[0] + 2.0 * p2_ * px_distorted[1] ;
    J_dist(0,1) = J_dist(1,0);
    return J_dist;
  }

  inline void undistort(double& x, double& y) const
  {
    double x0=x, y0=y;
    for(int i = 0; i < 5; ++i)
    {
      const double xx = x*x;
      const double yy = y*y;
      const double xy = x*y;
      const double xy2 = 2*xy;
      const double r2 = xx + yy;
      const double icdist = 1.0/(1.0 + (k1_ + k2_ * r2) * r2);
      const double dx = p1_ * xy2 + p2_ * (r2 + 2.0 * xx);
      const double dy = p2_ * xy2 + p1_ * (r2 + 2.0 * yy);
      x = (x0 - dx)*icdist;
      y = (y0 - dy)*icdist;
    }
  }

  inline void print(std::ostream& out) const
  {
    out << "  Distortion: RadTan("
        << k1_ << ", " << k2_ << ", " << p1_ << ", " << p2_ << ")"
        << std::endl;
  }

  enum DistortionParameters
  {
    kRadialDistortionFactor1,
    kRadialDistortionFactor2,
    kTangentialDistortionFactor1,
    kTangentialDistortionFactor2
  };
  // returns distortion parameters as vector
  // [k1 k2 p1 p2]
  inline Eigen::VectorXd getDistortionParameters() const
  {
    Eigen::VectorXd distortion(4);
    distortion(0) = k1_;
    distortion(1) = k2_;
    distortion(2) = p1_;
    distortion(3) = p2_;
    return distortion;
  }

  double k1_ = 0; // Radial distortion factor 1
  double k2_ = 0; // Radial distortion factor 2
  double p1_ = 0; // Tangential distortion factor 1
  double p2_ = 0; // Tangential distortion factor 2
};

} // namespace cameras
} // namespace vk
