// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <utility>  // std::pair
#include <iostream>

#include <boost/shared_ptr.hpp>

// ros
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/ColorRGBA.h>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <svo/global.h>
#include <svo/common/types.h>

#ifdef SVO_LOOP_CLOSING
#include <svo/online_loopclosing/keyframe.h>
#include <svo/online_loopclosing/loop_closing.h>
#endif

#ifdef SVO_GLOBAL_MAP
#include <svo/global_map.h>
#endif

namespace svo
{
// forward declarations
class FrameHandlerBase;

/// Publish visualisation messages to ROS.
class Visualizer
{
public:
  typedef std::shared_ptr<Visualizer> Ptr;
  typedef pcl::PointXYZI PointType;
  typedef pcl::PointCloud<PointType> PointCloud;

  static std::string kWorldFrame;

  static constexpr double seed_marker_scale_ = 0.03;
  static constexpr double seed_uncertainty_marker_scale_ = 0.03;
  static constexpr double trajectory_marker_scale_ = 0.03;
  static constexpr double point_marker_scale_ = 0.05;

  ros::NodeHandle pnh_;
  size_t trace_id_ = 0;
  std::string trace_dir_;
  size_t img_pub_level_;
  size_t img_pub_nth_;
  size_t dense_pub_nth_;
  bool viz_caption_str_;

  ros::Publisher pub_frames_;
  ros::Publisher pub_points_;
  ros::Publisher pub_imu_pose_;
  ros::Publisher pub_info_;
  ros::Publisher pub_markers_;
  ros::Publisher pub_pc_;
  PointCloud::Ptr pc_;
  std::vector<ros::Publisher> pub_cam_poses_;
  std::vector<ros::Publisher> pub_dense_;
  std::vector<image_transport::Publisher> pub_images_;

  tf::TransformBroadcaster br_;
  bool publish_world_in_cam_frame_;
  bool publish_map_every_frame_;
  ros::Duration publish_points_display_time_;
  bool publish_seeds_;
  bool publish_seeds_uncertainty_;
  bool publish_active_keyframes_;
  bool trace_pointcloud_;
  double vis_scale_;
  std::ofstream ofs_pointcloud_;

#ifdef SVO_LOOP_CLOSING
  PointCloud pose_graph_map_;
  ros::Publisher pub_loop_closure_;
  ros::Publisher pub_pose_graph_;
  ros::Publisher pub_pose_graph_map_;
#endif

#ifdef SVO_GLOBAL_MAP
  ros::Publisher pub_global_map_kfs_opt_;
  ros::Publisher pub_global_map_query_kfs_;
  ros::Publisher pub_global_map_pts_opt_;
  ros::Publisher pub_global_map_vis_;
  ros::Publisher pub_global_map_keypoints_vis_;
  ros::Publisher pub_global_map_matched_points_;
  ros::Publisher pub_global_map_reobserved_points_;
  ros::Publisher pub_global_map_reobserved_points_frontend_;
  ros::Publisher pub_global_map_point_ids_;
#endif
  ros::Publisher pub_visible_fixed_landmarks_;
  
  std::string img_caption_;

  Visualizer(const std::string& trace_dir, const ros::NodeHandle& nh_private,
             const size_t num_cameras);

  ~Visualizer() = default;

  void publishSvoInfo(const svo::FrameHandlerBase* const svo,
                      const int64_t timestamp_nanoseconds);

  void publishImages(const std::vector<cv::Mat>& images,
                     const int64_t timestamp_nanoseconds);

  void publishImagesWithFeatures(const FrameBundlePtr& frame_bundle,
                                 const int64_t timestamp,
                                 const bool draw_boundary);

  void publishImuPose(const Transformation& T_world_imu,
                      const Eigen::Matrix<double, 6, 6> Covariance,
                      const int64_t timestamp_nanoseconds);

  void publishCameraPoses(const FrameBundlePtr& frame_bundle,
                          const int64_t timestamp_nanoseconds);

  void publishBundleFeatureTracks(const FrameBundlePtr frames_ref,
                                  const FrameBundlePtr frames_cur,
                                  int64_t timestamp);

  void publishFeatureTracks(
      const Keypoints& px_ref, const Keypoints& px_cur,
      const std::vector<std::pair<size_t, size_t>>& matches_ref_cur,
      const ImgPyr& img_pyr, const Level& level, const uint64_t timestamp,
      const size_t frame_index);

  void visualizeHexacopter(const Transformation& T_frame_world,
                           const uint64_t timestamp);

  void visualizeQuadrocopter(const Transformation& T_frame_world,
                             const uint64_t timestamp);

  void visualizeMarkers(const FrameBundlePtr& frame_bundle,
                        const std::vector<FramePtr>& close_kfs,
                        const MapPtr& map);

  void publishTrajectoryPoint(const Eigen::Vector3d& pos_in_vision,
                              const uint64_t timestamp, const int id);

  void visualizeMarkersWithUncertainty(const FramePtr& frame,
                                       const std::vector<FramePtr>& close_kfs,
                                       const MapPtr& map,
                                       const float sigma_threshold);

  void publishSeedsBinary(const MapPtr& map, const float sigma_threshold);

  void publishSeeds(const MapPtr& map);

  void publishSeedsAsPointcloud(const Frame& frame, bool only_converged_seeds,
                                bool reset_pc_before_publishing = true);

  void publishVelocity(const Eigen::Vector3d& velocity_imu,
                       const uint64_t timestamp);

  void publishMapRegion(const std::vector<FramePtr>& frames);

  void publishKeyframeWithPoints(const FramePtr& frame,
                                 const uint64_t timestamp,
                                 const double marker_scale = 0.05);

  void publishActiveKeyframes(const std::vector<FramePtr>& active_kfs);

  void exportToDense(const FrameBundlePtr& frame_bundle);

  void publishSeedsUncertainty(const MapPtr& map);

  void visualizeCoordinateFrames(const Transformation& T_world_cam);

#ifdef SVO_LOOP_CLOSING
  void publishLoopClosureInfo(
      const LoopVizInfoVec& loop_viz_info_vec,
      const std::string& ns, const Eigen::Vector3f& color,
      const double scale=1.0);

  bool publishPoseGraph(const std::vector<KeyFramePtr>& kf_list,
                        const bool redo_pointcloud,
                        const size_t ignored_past_frames);
#endif

#ifdef SVO_GLOBAL_MAP
  void visualizeGlobalMap(const GlobalMap& gmap,
                          const std::string ns,
                          const Eigen::Vector3f& color,
                          const double scale);
  void visualizeFixedLandmarks(const FramePtr& frame);
#endif
  void writeCaptionStr(cv::Mat img);
};

}  // end namespace svo
