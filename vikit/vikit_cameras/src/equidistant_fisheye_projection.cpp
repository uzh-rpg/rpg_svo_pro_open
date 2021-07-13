#include "vikit/cameras/equidistant_fisheye_projection.h"

#include <glog/logging.h>

namespace vk {
namespace cameras {

// May later be generalized to any distortion once a use case exists.
EquidistantFisheyeProjection::EquidistantFisheyeProjection(
    double focal_length, Eigen::Vector2d principal_point)
: focal_length_(focal_length), principal_point_(principal_point)
{
  CHECK_NE(focal_length_, 0.);
}

bool EquidistantFisheyeProjection::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    Eigen::Vector3d* out_bearing_vector) const
{
  CHECK_NOTNULL(out_bearing_vector);

  constexpr double kEpsilon = 1e-10;
  Eigen::Vector2d p_c = keypoint - principal_point_;
  const double r = p_c.norm();
  if(fabs(r) < kEpsilon)
  {
    *out_bearing_vector << 0.0, 0.0, 1.0;
    return true;
  }

  const double theta = r / focal_length_;
  const double x = p_c[0];
  const double y = p_c[1];
  const double sin_theta_by_r = sin(theta) / r;
  *out_bearing_vector << x * sin_theta_by_r, y * sin_theta_by_r, cos(theta);
  return true;
}

void EquidistantFisheyeProjection::project3(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point) const
{
  CHECK_NOTNULL(out_keypoint);

  const double kEpsilon = 1e-10;

  const double x = point_3d[0];
  const double y = point_3d[1];
  const double z = point_3d[2];

  const double x2 = x * x;
  const double y2 = y * y;
  const double xy_norm2 = x2 + y2;
  const double xy_norm = std::sqrt(xy_norm2);
  const double xyz_norm2 = xy_norm2 + std::pow(z,2);
  const double xyz_norm = std::sqrt(xyz_norm2);

  CHECK_GE(xyz_norm, kEpsilon);

  double theta;
  if(fabs(xy_norm) < kEpsilon)
  {
    theta = 0.;
    (*out_keypoint) = principal_point_;
  }
  else
  {
    theta = std::acos(z / xyz_norm);
    const double r = focal_length_ * theta;

    (*out_keypoint) = principal_point_ + r / xy_norm * point_3d.head<2>();
  }

  if(out_jacobian_point)
  {
    const double R = xyz_norm;
    const double rho = xy_norm;
    const double f = focal_length_;
    const double rho2 = xy_norm2;
    const double R2 = xyz_norm2;

    const double duf_dx = f / rho * (theta - x2*theta/rho2 + z*x2/(rho*R2));
    const double duf_dy = f*x*y/rho2 * (z/R2 - theta/rho);
    const double duf_dz = -x*f/rho2 * (1-z*z/R2);

    const double dvf_dx = duf_dy;
    const double dvf_dy = f / rho * (theta - y2*theta/rho2 + z*y2/(rho*R2));
    const double dvf_dz = -y*f/rho2 * (1-z*z/R2);

    (*out_jacobian_point) << duf_dx, duf_dy, duf_dz,
                             dvf_dx, dvf_dy, dvf_dz;
  }
}

double EquidistantFisheyeProjection::errorMultiplier() const
{
  // TODO(tcies) What is this?
  return 1.;
}

double EquidistantFisheyeProjection::getAngleError(double img_err) const
{
  return img_err / focal_length_;
}

void EquidistantFisheyeProjection::print(std::ostream& out) const
{
  out << "  Projection = Equidistant Fisheye" << std::endl;
  out << "  Focal length = (" << focal_length_ << std::endl;
  out << "  Principal point = " << principal_point_.transpose() << std::endl;
}

Eigen::VectorXd EquidistantFisheyeProjection::getIntrinsicParameters() const
{
  Eigen::VectorXd intrinsics(3);
  intrinsics(0) = focal_length_;
  intrinsics(1) = principal_point_(0);
  intrinsics(2) = principal_point_(1);
  return intrinsics;
}

Eigen::VectorXd EquidistantFisheyeProjection::getDistortionParameters() const
{
  Eigen::VectorXd distortion;
  return distortion;
}

}  // namespace cameras
}  // namespace vk
