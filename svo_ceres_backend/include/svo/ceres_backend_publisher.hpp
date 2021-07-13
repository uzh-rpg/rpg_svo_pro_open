// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2017 Jonathan Huber <jonathan.huber at uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <svo/vio_common/backend_types.hpp>
#include <mutex>

#include "svo/ceres_backend/map.hpp"

namespace svo
{
class CeresBackendPublisher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<CeresBackendPublisher> Ptr;
  using PointCloud = pcl::PointCloud<pcl::PointXYZI>;
  using PointType = pcl::PointXYZI;
  const std::string kWorldFrame = "world";

  CeresBackendPublisher(const ros::NodeHandle& nh_private,
                        const std::shared_ptr<ceres_backend::Map>& map_ptr);
  ~CeresBackendPublisher()
  {
  }

  Transformation getLastT_W_B() const
  {
    return state_.get_T_W_B();
  }

  void addFrame(const BundleId& bundle_id)
  {
    std::lock_guard<std::mutex> lock(mutex_frame_id_);
    last_added_frame_ = bundle_id;
  }

  void publish(const ViNodeState& state, const int64_t timestamp,
               const int32_t seq);

private:
  ros::NodeHandle pnh_;

  mutable std::mutex mutex_frame_id_;

  std::shared_ptr<ceres_backend::Map> map_ptr_;  ///< The underlying svo::Map.

  // Transform used for tracing
  ViNodeState state_;
  BundleId state_frame_id_ = -1;
  BundleId last_added_frame_ = -1;

  // publisher helpers
  ros::Publisher pub_imu_pose_;
  ros::Publisher pub_imu_pose_viz_;
  ros::Publisher pub_points_;

  // publisher functions
  void publishImuPose(const ViNodeState& state, const int64_t timestamp,
                      const int32_t seq);
  void publishBackendLandmarks(const int64_t timestamp) const;
};

}  // namespace svo
