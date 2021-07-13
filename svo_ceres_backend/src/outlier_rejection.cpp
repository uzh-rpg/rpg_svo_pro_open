#include "svo/outlier_rejection.hpp"

#include <svo/common/frame.h>
#include <svo/common/point.h>

namespace svo
{
void OutlierRejection::removeOutliers(
    Frame& frame,
    size_t& n_deleted_edges,
    size_t& n_deleted_corners,
    std::vector<int> &deleted_points,
    const bool ignore_fixed) const
{
  // calculate threshold
  const double outlier_threshold = reproj_err_threshold_
      / frame.getErrorMultiplier(); // focal length;

  for(size_t i = 0; i < frame.num_features_; ++i)
  {

    if (ignore_fixed && isFixedLandmark(frame.type_vec_[i]))
    {
      continue;
    }

    Position xyz_world;
    if(frame.landmark_vec_[i] != nullptr)
    {
      xyz_world = frame.landmark_vec_[i]->pos_;
    }
    else
      continue;

    // calculate residual according to different feature type and residual
    double unwhitened_error;

    if(isEdgelet(frame.type_vec_[i]))
    {
      calculateEdgeletResidualUnitPlane(
            frame.f_vec_.col(static_cast<int>(i)), xyz_world,
            frame.grad_vec_.col(static_cast<int>(i)), frame.T_cam_world(),
            unwhitened_error);
    }
    else
    {
      calculateFeatureResidualUnitPlane(frame.f_vec_.col(static_cast<int>(i)),
                                        xyz_world, frame.T_cam_world(),
                                        unwhitened_error);
    }
    unwhitened_error *= 1.0 / (1 << frame.level_vec_(static_cast<int>(i)));

    if(std::fabs(unwhitened_error) > outlier_threshold)
    {
      deleted_points.push_back(frame.track_id_vec_[static_cast<int>(i)]);
      if(isEdgelet(frame.type_vec_[i]))
        ++n_deleted_edges;
      else
        ++n_deleted_corners;

      frame.type_vec_[i] = FeatureType::kOutlier;
      frame.seed_ref_vec_[i].keyframe.reset();
      frame.landmark_vec_[i]->removeObservation(frame.id());
      frame.landmark_vec_[i] = nullptr; // delete landmark observation
      continue;
    }
  }
}

void OutlierRejection::calculateEdgeletResidualUnitPlane(
    const Eigen::Ref<const BearingVector>& f,
    const Position& xyz_in_world,
    const Eigen::Ref<const GradientVector>& grad,
    const Transformation& T_cam_world,
    double& unwhitened_error) const
{
  const Eigen::Vector3d xyz_in_cam(T_cam_world*xyz_in_world);

  // Compute error.
  double e = grad.dot(vk::project2(f) - vk::project2(xyz_in_cam));
  unwhitened_error = std::abs(e);
}

void OutlierRejection::calculateFeatureResidualUnitPlane(
    const Eigen::Ref<const BearingVector>& f,
    const Position& xyz_in_world,
    const Transformation& T_cam_world,
    double& unwhitened_error) const
{
  const Eigen::Vector3d xyz_in_cam(T_cam_world*xyz_in_world);

  // Prediction error.
  Eigen::Vector2d e = vk::project2(f) - vk::project2(xyz_in_cam);
  unwhitened_error = e.norm();
}

} //namespace svo

