#pragma once

#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <memory>

namespace svo {
namespace trajectory {

class Trajectory
{
public:
  typedef std::pair<double, Transformation> StampedPose;

  static Trajectory loadfromFile(const std::string& filename);
  static Trajectory createCircularTrajectory(size_t length, double fps, const Eigen::Vector3d& center, double radius);
  static Trajectory createStraightTrajectory(size_t length, double fps, const Eigen::Vector3d& start, const Eigen::Vector3d& end);

  Trajectory() {}
  Trajectory(const std::vector<StampedPose>& poses)
    : current_idx_(0)
    , trajectory_(poses) {}

  StampedPose getStampedPoseAtIdx(size_t requested_idx) const
  {
    return trajectory_[requested_idx];
  }

  StampedPose getNextStampedPose() const
  {
    return trajectory_[current_idx_++];
  }

  size_t length() const { return trajectory_.size(); }
  void seekPose(size_t idx) const { current_idx_ = idx; }
  void reset() const { current_idx_ = 0; }

private:
  mutable size_t current_idx_;
  std::vector<StampedPose> trajectory_;
};

}
}
