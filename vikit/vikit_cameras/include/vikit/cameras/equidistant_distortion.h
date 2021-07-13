#pragma once

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <glog/logging.h>

namespace vk
{
namespace cameras
{
// This class implements the distortion model described in the paper:
// "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle,
// and Fish-Eye Lenses" by Juho Kannala and Sami S. Brandt, PAMI.
class EquidistantDistortion
{
public:
  EquidistantDistortion(const double k1, const double k2, const double k3,
                        const double k4)
    : k1_(k1), k2_(k2), k3_(k3), k4_(k4)
  {
  }

  EquidistantDistortion(const Eigen::VectorXd& parameters)
  {
    CHECK(parameters.size() == 4);
    k1_ = parameters(0);
    k2_ = parameters(1);
    k3_ = parameters(2);
    k4_ = parameters(3);
  }

  ~EquidistantDistortion() = default;

  inline void distort(double& x, double& y) const
  {
    const double r = std::sqrt(x * x + y * y);
    if (r < kRThresh)
    {
      return;
    }

    const double theta = std::atan(r);
    const double thetad = thetad_from_theta(theta);
    const double scaling = thetad / r;
    x *= scaling;
    y *= scaling;
  }

  inline Eigen::Vector2d distort(const Eigen::Vector2d& vector) const
  {
    const double r = vector.norm();
    if (r < kRThresh)
    {
      return vector;
    }

    const double theta = std::atan(r);
    const double thetad = thetad_from_theta(theta);
    const double scaling = thetad / r;
    return vector * scaling;
  }

  inline Eigen::Matrix2d jacobian(const Eigen::Vector2d& uv) const
  {
    const double r = uv.norm();
    if (r < kRThresh)
    {
      return Eigen::Matrix2d::Identity();
    }

    const double inv_r = 1.0 / r;
    const double r2 = r * r;
    const double dr_du = uv(0) * inv_r;
    const double dr_dv = uv(1) * inv_r;

    const double theta = std::atan(r);
    const double dtheta_dr = 1.0 / (1 + r * r);

    const double thetad = thetad_from_theta(theta);
    const double dthetad_dtheta = deriv_thetad_from_theta(theta);
    const double dthetad_dr = dthetad_dtheta * dtheta_dr;

    const double scaling = thetad / r;
    const double dscaling_du = (dthetad_dr * dr_du * r - dr_du * thetad) / r2;
    const double dscaling_dv = (dthetad_dr * dr_dv * r - dr_dv * thetad) / r2;

    const double dx_du = dscaling_du * uv(0) + scaling;
    const double dx_dv = dscaling_dv * uv(0);
    const double dy_du = dscaling_du * uv(1);
    const double dy_dv = dscaling_dv * uv(1) + scaling;
    Eigen::Matrix2d jac;
    jac << dx_du, dx_dv, dy_du, dy_dv;
    return jac;
  }

  inline void undistort(double& x, double& y) const
  {
    const double thetad = std::sqrt(x * x + y * y);
    double theta = thetad;
    for (int i = 0; i < 5; ++i)
    {
      const double theta2 = theta * theta;
      const double theta4 = theta2 * theta2;
      const double theta6 = theta4 * theta2;
      const double theta8 = theta4 * theta4;
      theta = thetad /
              (1.0 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);
    }
    const double scaling = std::tan(theta) / thetad;
    x *= scaling;
    y *= scaling;
  }

  inline void print(std::ostream& out) const
  {
    out << "  Distortion: Equidistant(" << k1_ << ", " << k2_ << ", " << k3_
        << ", " << k4_ << ")" << std::endl;
  }

  enum DistortionParameters
  {
    kRadialDistortionFactor1,
    kRadialDistortionFactor2,
    kRadialDistortionFactor3,
    kRadialDistortionFactor4
  };
  // returns distortion parameters as vector
  // [k1 k2 k3 k4]
  inline Eigen::VectorXd getDistortionParameters() const
  {
    Eigen::VectorXd distortion(4);
    distortion(0) = k1_;
    distortion(1) = k2_;
    distortion(2) = k3_;
    distortion(3) = k4_;
    return distortion;
  }

  double k1_ = 0;  // Radial distortion factor 1
  double k2_ = 0;  // Radial distortion factor 2
  double k3_ = 0;  // Radial distortion factor 3
  double k4_ = 0;  // Radial distortion factor 4

private:
  inline double thetad_from_theta(const double theta) const
  {
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta4 * theta2;
    const double theta8 = theta4 * theta4;
    const double thetad = theta * (1.0 + k1_ * theta2 + k2_ * theta4 +
                                   k3_ * theta6 + k4_ * theta8);
    return thetad;
  }

  inline double deriv_thetad_from_theta(const double theta) const
  {
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta4 * theta2;
    const double theta8 = theta4 * theta4;
    return 1 + 3 * k1_ * theta2 + 5 * k2_ * theta4 + 7 * k3_ * theta6 +
           9 * k4_ * theta8;
  }

  const double kRThresh = 1e-8;
};

}  // namespace cameras
}  // namespace vk
