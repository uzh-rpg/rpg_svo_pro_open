#pragma once

#include "vikit/cameras/camera_geometry.h"
#include "vikit/cameras/omni_projection.h"

namespace vk {
namespace cameras {

// Omni geometry is strongly typed to enforce mask.
class OmniGeometry : public CameraGeometry<OmniProjection>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static size_t kParamNum = 24;
  OmniGeometry(const int width, const int height,
               const Eigen::Matrix<double, 5, 1>& polynomial,
               const Eigen::Vector2d& principal_point,
               const Eigen::Vector3d& distortion,
               const Eigen::Matrix<double, 12, 1>& inverse_polynomial,
               const Eigen::Vector2d& mask_relative_radii);
  // Version which does not apply a mask.
  OmniGeometry(const int width, const int height,
               const Eigen::Matrix<double, 5, 1>& polynomial,
               const Eigen::Vector2d& principal_point,
               const Eigen::Vector3d& distortion,
               const Eigen::Matrix<double, 12, 1>& inverse_polynomial);
  OmniGeometry(const int width, const int height,
               const Eigen::VectorXd& intrinsics);

  virtual ~OmniGeometry() = default;

 private:
  static const Eigen::Vector2d kDontMask;
};

}  // namespace cameras
}  // namespace vk
