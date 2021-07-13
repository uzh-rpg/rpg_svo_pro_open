// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <svo/common/transformation.h>
#include <svo/backend/smart_factors_fwd.h>

namespace swe {
  class SweGtsamBookeeper;
}
namespace svo {

/// Publish visualisation messages to ROS.
class CeresBackendPublisher
{
public:
  typedef std::shared_ptr<CeresBackendPublisher> Ptr;
  typedef pcl::PointXYZ PointType;
  typedef pcl::PointCloud<PointType> PointCloud;

  static const std::string kWorldFrame;

  ros::NodeHandle pnh_;
  size_t trace_id_;
  std::string trace_dir_;
  ros::Publisher pub_markers_;
  ros::Publisher pub_pc_;
  double vis_scale_;
  std::ofstream ofs_states_;
  std::ofstream ofs_covariance_;

  CeresBackendPublisher(const std::string& trace_dir,
                    const ros::NodeHandle& pnh);

  ~CeresBackendPublisher() = default;

  void visualizeFrames(const gtsam::Values& values);

  void visualizePoints(const gtsam::Values& values);

  void publishPointcloud(const gtsam::Values& values);

  void visualizeSmartFactors(const SmartFactorMap& smart_factors);

  void visualizeVelocity(const gtsam::Values& values);

  void visualizePoseCovariance(
      const Transformation& T_W_B,
      const Eigen::Matrix<double, 6, 6>& covariance);

  void tracePose(std::map<int, int64_t, std::less<int>> frameid_timestamp_map,
                 const gtsam::Values& values);

  void traceStates(const gtsam::Values& values);

  void traceCovariance(
      const Eigen::Matrix<double, 6, 6>& C,
      const int64_t timestamp);
};

} // end namespace svo

