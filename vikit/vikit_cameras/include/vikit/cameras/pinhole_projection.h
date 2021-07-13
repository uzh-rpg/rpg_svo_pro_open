#pragma once

#include <Eigen/Core>

#include "vikit/cameras/no_distortion.h"
#include "vikit/cameras/camera_geometry_base.h"

namespace vk {
namespace cameras {

template<typename Distortion>
class PinholeProjection
{
public:
  typedef Distortion distortion_t;
  const CameraGeometryBase::Type cam_type_ = CameraGeometryBase::Type::kPinhole;

  PinholeProjection() = default;

  PinholeProjection(
      double fx, double fy, double cx, double cy, distortion_t distortion);

  // Intrinsic parameters ordering: fu, fv, cu, cv
  PinholeProjection(const Eigen::VectorXd& intrinsics, distortion_t distortion);

  ~PinholeProjection() = default;

  // Computes bearing vector from pixel coordinates. Z-component of the returned
  // bearing vector is 1. IMPORTANT: returned vector is NOT of unit length!
  bool backProject3(
      const Eigen::Ref<const Eigen::Vector2d>& keypoint,
      Eigen::Vector3d* out_point_3d) const;

  // Computes pixel coordinates from bearing vector.
  void project3(
        const Eigen::Ref<const Eigen::Vector3d>& point_3d,
        Eigen::Vector2d* out_keypoint,
        Eigen::Matrix<double, 2, 3>* out_jacobian_point) const ;

  // Returns focal length (transforms unit plane error to pixel error).
  double errorMultiplier() const;

  double getAngleError(double img_err) const;

  // Focal length will be width / 2, thus corresponding to a 90deg field of
  // view.
  static PinholeProjection<Distortion> createTestProjection(
      const size_t image_width, const size_t image_height);

  template<typename T>
  const T* distortion() const;

  void print(std::ostream& out) const;

  enum IntrinsicParameters
  {
    kFocalLengthX,
    kFocalLengthY,
    kPricipalPointX,
    kPrincipalPointY
  };
  // Returns the intrinsic Parameters in vectof of form
  // [fx, fy, cx, cy]
  Eigen::VectorXd getIntrinsicParameters() const;

  // returns the distortion Parameters of the underlying distortion model
  Eigen::VectorXd getDistortionParameters() const;

  double fx_ = 1; // Focal length x.
  double fy_ = 1; // Focal length y.
  double fx_inv_ = 1; // Inverse focal length x
  double fy_inv_ = 1; // Inverse focal length y
  double cx_ = 0; // Principle point x.
  double cy_ = 0; // Principle point y.
  distortion_t distortion_;
};

} // namespace cameras
} // namespace vk

#include <vikit/cameras/implementation/pinhole_projection.hpp>
