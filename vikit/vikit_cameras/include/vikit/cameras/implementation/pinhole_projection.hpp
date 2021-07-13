#include <iostream>
#include <glog/logging.h>
#include <vikit/cameras/pinhole_projection.h>

namespace vk {
namespace cameras {

template<typename Distortion>
PinholeProjection<Distortion>::PinholeProjection(
    double fx, double fy, double cx, double cy, distortion_t distortion)
  : fx_(fx), fy_(fy), fx_inv_(1.0/fx_), fy_inv_(1.0/fy_)
  , cx_(cx), cy_(cy), distortion_(distortion)
{}

template<typename Distortion>
PinholeProjection<Distortion>::PinholeProjection(
    const Eigen::VectorXd& intrinsics, distortion_t distortion)
  : distortion_(distortion)
{
  CHECK(intrinsics.size() == 4);
  fx_ = intrinsics(0);
  fy_ = intrinsics(1);
  cx_ = intrinsics(2);
  cy_ = intrinsics(3);
  fx_inv_ = 1.0/fx_;
  fy_inv_ = 1.0/fy_;
}

template<typename Distortion>
bool PinholeProjection<Distortion>::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    Eigen::Vector3d* out_point_3d) const
{
  double x = (keypoint[0]-cx_)*fx_inv_;
  double y = (keypoint[1]-cy_)*fy_inv_;
  distortion_.undistort(x, y);
  (*out_point_3d)[0] = x;
  (*out_point_3d)[1] = y;
  (*out_point_3d)[2] = 1.0;
  return true;
}

template<typename Distortion>
void PinholeProjection<Distortion>::project3(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point) const
{
  const double z_inv = 1 / point_3d(2);
  // TODO(tcies) precalced member?
  const Eigen::DiagonalMatrix<double, 2> focal_matrix(fx_, fy_);
  const Eigen::Vector2d uv = point_3d.head<2>() * z_inv;
  (*out_keypoint) =
      focal_matrix * distortion_.distort(uv) + Eigen::Vector2d(cx_, cy_);

  if(out_jacobian_point)
  {
    Eigen::Matrix<double, 2, 3> duv_dxy;
    duv_dxy.leftCols<2>() = Eigen::Matrix2d::Identity() * z_inv;
    duv_dxy.rightCols<1>() = - point_3d.head<2>() * z_inv * z_inv;

    (*out_jacobian_point) = focal_matrix * distortion_.jacobian(uv) * duv_dxy;
  }
}

template<typename Distortion>
double PinholeProjection<Distortion>::errorMultiplier() const
{
  return std::abs(fx_);
}

template<typename Distortion>
double PinholeProjection<Distortion>::getAngleError(double img_err) const
{
  return std::atan(img_err/(2.0*fx_)) + std::atan(img_err/(2.0*fy_));
}

template<typename Distortion>
PinholeProjection<Distortion>
PinholeProjection<Distortion>::createTestProjection(
    const size_t image_width, const size_t image_height) {
  return PinholeProjection(
      image_width / 2, image_width / 2, image_width / 2, image_height / 2,
      Distortion::createTestDistortion());
}

template<typename Distortion>
template<typename T>
const T* PinholeProjection<Distortion>::distortion() const
{
  return dynamic_cast<const T*>(&distortion_);
}

template<typename Distortion>
void PinholeProjection<Distortion>::print(std::ostream& out) const
{
  out << "  Projection = Pinhole" << std::endl;
  out << "  Focal length = (" << fx_ << ", " << fy_ << ")" << std::endl;
  out << "  Principal point = (" << cx_ << ", " << cy_ << ")" << std::endl;
  distortion_.print(out);
}

template<typename Distortion>
Eigen::VectorXd PinholeProjection<Distortion>::getIntrinsicParameters() const
{
  Eigen::VectorXd intrinsics(4);
  intrinsics(0) = fx_;
  intrinsics(1) = fy_;
  intrinsics(2) = cx_;
  intrinsics(3) = cy_;
  return intrinsics;
}

template<typename Distortion>
Eigen::VectorXd PinholeProjection<Distortion>::getDistortionParameters() const
{
  return distortion_.getDistortionParameters();
}


} // namespace cameras
} // namespace vk
