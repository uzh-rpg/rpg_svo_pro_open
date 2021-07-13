// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#include <svo_ros/backend_visualizer.h>

// ros
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// svo
#include <vikit/timer.h>
#include <vikit/output_helper.h>
#include <vikit/params_helper.h>

// gtsam
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

namespace svo {

const std::string BackendVisualizer::kWorldFrame = "world";

BackendVisualizer::BackendVisualizer(
    const std::string& trace_dir,
    const ros::NodeHandle& pnh)
  : pnh_(pnh)
  , trace_id_(0)
  , trace_dir_(trace_dir)
  , vis_scale_(vk::param<double>(pnh_, "publish_marker_scale", 1.2))
{
  // Init ROS Marker Publishers
  pub_markers_ = pnh_.advertise<visualization_msgs::Marker>("ba_frames", 100);
  pub_pc_ = pnh_.advertise<PointCloud>("ba_pointcloud", 1);
}

void BackendVisualizer::visualizeFrames(const gtsam::Values& values)
{

  if(pub_markers_.getNumSubscribers() == 0)
    return;

  double marker_scale = 0.03*vis_scale_;

  gtsam::Values::ConstFiltered<gtsam::Pose3> frames =
      values.filter<gtsam::Pose3>(gtsam::Symbol::ChrTest('x'));

  // position of each frame
  visualization_msgs::Marker m;
  m.header.frame_id = kWorldFrame;
  m.header.stamp = ros::Time();
  m.ns = "frames";
  m.id = 1;
  m.type = visualization_msgs::Marker::POINTS;
  m.action = 0; // add/modify
  m.scale.x = marker_scale;
  m.scale.y = marker_scale;
  m.scale.z = marker_scale;
  m.color.a = 1.0;
  m.color.r = 1.0;
  m.color.g = 0.0;
  m.color.b = 0.0;
  m.points.reserve(frames.size());
  for(auto it = frames.begin(); it != frames.end(); ++it)
  {
    const gtsam::Point3& xyz = it->value.translation();
    geometry_msgs::Point p;
    p.x = xyz.x();
    p.y = xyz.y();
    p.z = xyz.z();
    m.points.push_back(p);
  }
  pub_markers_.publish(m);
  VLOG(30) << "Viz: Published " << m.points.size() << " frames.";

  // line trip as trajectory
  visualization_msgs::Marker m_traj;
  m_traj.header.frame_id = kWorldFrame;
  m_traj.header.stamp = ros::Time();
  m_traj.ns = "traj";
  m_traj.id = 2;
  m_traj.type = visualization_msgs::Marker::LINE_STRIP;
  m_traj.action = 0; // add/modify
  m_traj.scale.x = marker_scale;
  m_traj.scale.y = marker_scale;
  m_traj.scale.z = marker_scale;
  m_traj.color.a = 1.0;
  m_traj.color.r = 0.0;
  m_traj.color.g = 1.0;
  m_traj.color.b = 0.0;
  m_traj.points.reserve(frames.size());
  for(auto it = frames.begin(); it != frames.end(); ++it)
  {
    const gtsam::Point3& xyz = it->value.translation();
    geometry_msgs::Point p;
    p.x = xyz.x();
    p.y = xyz.y();
    p.z = xyz.z();
    m_traj.points.push_back(p);
  }
  pub_markers_.publish(m_traj);

  // coordinate frame
  int i = 0;
  for(auto it = frames.begin(); it != frames.end(); ++it)
  {
    vk::output_helper::publishFrameMarker(
          pub_markers_,
          it->value.rotation().matrix(),
          it->value.translation().vector(),
          "keyframes",
          ros::Time::now(), (i++)*10, 0, 0.05);
  }
}

void BackendVisualizer::visualizePoints(const gtsam::Values& values)
{
  if(pub_markers_.getNumSubscribers() == 0)
    return;

  double marker_scale = 0.03*vis_scale_;
  visualization_msgs::Marker m;
  m.header.frame_id = kWorldFrame;
  m.header.stamp = ros::Time();
  m.ns = "points";
  m.id = 0;
  m.type = visualization_msgs::Marker::POINTS;
  m.action = 0; // add/modify
  m.scale.x = marker_scale;
  m.scale.y = marker_scale;
  m.scale.z = marker_scale;
  m.color.a = 1.0;
  m.color.r = 0.5;
  m.color.g = 0.0;
  m.color.b = 0.5;
  gtsam::Values::ConstFiltered<gtsam::Point3> points =
      values.filter<gtsam::Point3>(gtsam::Symbol::ChrTest('l'));
  const size_t n_pts = points.size();
  m.points.reserve(n_pts);
  for(auto it = points.begin(); it != points.end(); ++it)
  {
    const gtsam::Point3& xyz = it->value;
    geometry_msgs::Point p;
    p.x = xyz.x();
    p.y = xyz.y();
    p.z = xyz.z();
    m.points.push_back(p);
  }
  pub_markers_.publish(m);
  VLOG(30) << "Viz: Published " << m.points.size() << " points.";
}

void BackendVisualizer::publishPointcloud(const gtsam::Values& values)
{
  if(pub_pc_.getNumSubscribers() == 0)
    return;

  PointCloud::Ptr pc(new PointCloud);
  pc->header.frame_id = kWorldFrame;
  pc->header.stamp = ros::Time::now().toNSec() / 1000ull;

  gtsam::Values::ConstFiltered<gtsam::Point3> points =
      values.filter<gtsam::Point3>(gtsam::Symbol::ChrTest('l'));
  pc->reserve(points.size());
  for(auto it = points.begin(); it != points.end(); ++it)
  {
    const gtsam::Point3& xyz = it->value;
    PointType p;
    p.x = xyz.x();
    p.y = xyz.y();
    p.z = xyz.z();
    pc->push_back(p);
  }
  pub_pc_.publish(pc);
}

void BackendVisualizer::visualizeSmartFactors(const SmartFactorMap& smart_factors)
{
  if(pub_markers_.getNumSubscribers() == 0)
    return;

  double marker_scale = 0.03*vis_scale_;
  visualization_msgs::Marker m;
  m.header.frame_id = kWorldFrame;
  m.header.stamp = ros::Time();
  m.ns = "smart_points";
  m.id = 0;
  m.type = visualization_msgs::Marker::POINTS;
  m.action = 0; // add/modify
  m.scale.x = marker_scale;
  m.scale.y = marker_scale;
  m.scale.z = marker_scale;
  m.color.a = 1.0;
  m.color.r = 0.0;
  m.color.g = 0.7;
  m.color.b = 0.0;
  m.points.reserve(smart_factors.size());
  size_t n_degenerate = 0;
  size_t n_cheirality = 0;
  for(const auto& sf : smart_factors)
  {
    gtsam::Vector3 xyz = sf.second.first->point().get().vector();
    if(!sf.second.first->isDegenerate())
    {
      geometry_msgs::Point p;
      p.x = xyz.x();
      p.y = xyz.y();
      p.z = xyz.z();
      m.points.push_back(p);
    }
    else
    {
      ++n_degenerate;
      if(sf.second.first->isPointBehindCamera())
        ++n_cheirality;
    }
  }
  pub_markers_.publish(m);

  // publish degenerate
  m.points.clear();
  m.id = 1;
  m.color.r = 0.5;
  m.color.g = 0.0;
  m.color.b = 0.0;
  m.points.reserve(smart_factors.size());
  for(const auto& sf : smart_factors)
  {
    gtsam::Vector3 xyz = sf.second.first->point().get().vector();
    if(sf.second.first->isDegenerate())
    {
      geometry_msgs::Point p;
      p.x = xyz.x();
      p.y = xyz.y();
      p.z = xyz.z();
      m.points.push_back(p);
    }
  }
  pub_markers_.publish(m);

  VLOG(30) << "Viz: Published " << m.points.size() << " smart-factors.";
  VLOG(30) << "Viz: Number of degenerate smart-factors:" << n_degenerate;
  VLOG(30) << "Viz: Number of points behind camera smart-factors:" << n_cheirality;
}

void BackendVisualizer::visualizeVelocity(const gtsam::Values& values)
{
  if(pub_markers_.getNumSubscribers() == 0)
    return;

  double marker_scale = 0.02*vis_scale_; // 0.02
  double velocity_scale = 0.1*vis_scale_; // 0.1
  visualization_msgs::Marker m;
  m.header.frame_id = kWorldFrame;
  m.header.stamp = ros::Time();
  m.ns = "velocity";
  m.id = 0;
  m.type = visualization_msgs::Marker::LINE_LIST;
  m.action = 0; // add/modify
  m.scale.x = marker_scale;
  m.scale.y = marker_scale;
  m.scale.z = marker_scale;
  m.color.a = 1.0;
  m.color.r = 0.0;
  m.color.g = 0.0;
  m.color.b = 1.0;
  gtsam::Values::ConstFiltered<gtsam::Pose3> frames =
      values.filter<gtsam::Pose3>(gtsam::Symbol::ChrTest('x'));
  const size_t n_frames = frames.size();
  m.points.reserve(n_frames);
  for(auto it = frames.begin(); it != frames.end(); ++it)
  {
    const gtsam::Pose3& T_world_imu = it->value;
    const gtsam::Point3& xyz = T_world_imu.translation();
    geometry_msgs::Point p;
    p.x = xyz.x();
    p.y = xyz.y();
    p.z = xyz.z();
    m.points.push_back(p);
    const gtsam::Symbol frame_symbol(it->key);
    const gtsam::Symbol vel_symbol('v', frame_symbol.index());
    const gtsam::Vector3 vel = values.at< gtsam::Vector3 >(vel_symbol);
    const gtsam::Vector3 xyz_plus = xyz.vector()+vel*velocity_scale;

    p.x = xyz_plus.x();
    p.y = xyz_plus.y();
    p.z = xyz_plus.z();
    m.points.push_back(p);
  }
  pub_markers_.publish(m);
  VLOG(30) << "Viz: Published " << m.points.size()/2 << " velocities.";
}

void BackendVisualizer::visualizePoseCovariance(
    const Transformation& T_W_B,
    const Eigen::Matrix<double, 6, 6>& covariance)
{
  if(pub_markers_.getNumSubscribers() == 0)
    return;

  vk::output_helper::publishGtsamPoseCovariance(
        pub_markers_, T_W_B.getPosition(), T_W_B.getRotationMatrix(), covariance,
        "covariance", trace_id_, 0, 2.0, Eigen::Vector4d(0, 0, 0.5, 0.5));
}

void BackendVisualizer::tracePose(std::map<int, int64_t, std::less<int>> frameid_timestamp_map,
                                  const gtsam::Values& values)
{
  std::string states_trace_name(trace_dir_+"/traj_estimate.txt");
  ofs_states_.open(states_trace_name.c_str());
  ofs_states_.precision(8);
  CHECK(ofs_states_.is_open());

  gtsam::Values::ConstFiltered<gtsam::imuBias::ConstantBias> biases =
      values.filter<gtsam::imuBias::ConstantBias>(gtsam::Symbol::ChrTest('b'));
  for(std::pair<int, int64_t> keyval : frameid_timestamp_map)
  {
    const gtsam::Pose3& T_wi = values.at<gtsam::Pose3>(gtsam::Symbol('x', keyval.first));
    const Eigen::Quaterniond q(T_wi.rotation().matrix());
    const Eigen::Vector3d p(T_wi.translation().vector());
    ofs_states_ << keyval.second << " "
                << p.x() << " " << p.y() << " " << p.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                << std::endl;
  }

  // Trace pointcloud
  std::string states_pc_name(trace_dir_+"/pointcloud.txt");
  std::ofstream ofs_pc(states_pc_name.c_str());
  CHECK(ofs_pc.is_open());

  gtsam::Values::ConstFiltered<gtsam::Point3> points =
      values.filter<gtsam::Point3>(gtsam::Symbol::ChrTest('l'));
  for(auto it = points.begin(); it != points.end(); ++it)
  {
    const gtsam::Point3& xyz = it->value;
    ofs_pc << xyz.x() << " " << xyz.y() << " " << xyz.z() << std::endl;
  }
  ofs_pc.close();
}

void BackendVisualizer::traceStates(const gtsam::Values& values)
{
  if(!ofs_states_.is_open())
  {
    std::string states_trace_name(trace_dir_+"/trace_states.txt");
    ofs_states_.open(states_trace_name.c_str());
    ofs_states_.precision(10);
  }
  CHECK(ofs_states_.is_open());

  gtsam::Values::ConstFiltered<gtsam::imuBias::ConstantBias> biases =
      values.filter<gtsam::imuBias::ConstantBias>(gtsam::Symbol::ChrTest('b'));
  for(auto b : biases)
  {
    const gtsam::imuBias::ConstantBias& bias = b.value;
    const size_t id = gtsam::Symbol(b.key).index();
    const gtsam::Vector& velocity =
        values.at<gtsam::Vector3>(gtsam::Symbol('v', id));
    const gtsam::Pose3& T_wi =
        values.at<gtsam::Pose3>(gtsam::Symbol('x', id));
    const Eigen::Quaterniond q(T_wi.rotation().matrix());
    ofs_states_ << trace_id_ << " "
                << id << " "
                << bias.accelerometer().x() << " "
                << bias.accelerometer().y() << " "
                << bias.accelerometer().z() << " "
                << bias.gyroscope().x() << " "
                << bias.gyroscope().y() << " "
                << bias.gyroscope().z() << " "
                << velocity.x() << " "
                << velocity.y() << " "
                << velocity.z() << " "
                << T_wi.translation().x() << " "
                << T_wi.translation().y() << " "
                << T_wi.translation().z() << " "
                << q.x() << " "
                << q.y() << " "
                << q.z() << " "
                << q.w() << " "
                << std::endl;
  }
  ++trace_id_;
}

/*
void BackendVisualizer::traceStates(
    const gtsam::Values& values,
    const swe::SweGtsamBookeeper& book)
{
  if(!ofs_states_.is_open())
  {
    std::string states_trace_name(trace_dir_+"/estimate_states.csv");
    ofs_states_.open(states_trace_name.c_str());
    ofs_states_.precision(10);
  }
  CHECK(ofs_states_.is_open());

  if(trace_id_ == 0)
  {
    // Write header.
    ofs_states_ << "# Iteration count, timestamp [ns], bax, bay, baz, bgx, bgy, bgz, vx, "
                << "vy, vz, tx, ty, tz, qx, qy, qz, qw" << std::endl;
  }

  std::vector<int64_t> timestamps;
  book.getAllViNodeTimestamps(&timestamps);
  for(int64_t timestamp : timestamps)
  {
    aslam::NFramesId nframe_id;
    book.getNFrameIdFromTimestamp(timestamp, &nframe_id);
    swe::SweGtsamBookeeper::ViNodeKeys vikeys;
    book.getViNodeKeysFromNFrameId(nframe_id, &vikeys);

    const gtsam::Pose3& T_wi = values.at<gtsam::Pose3>(vikeys.pose);
    const gtsam::Vector& velocity = values.at<gtsam::Vector3>(vikeys.velocity);
    const gtsam::imuBias::ConstantBias& bias =
        values.at<gtsam::imuBias::ConstantBias>(vikeys.imubias);
    const Eigen::Quaterniond q(T_wi.rotation().matrix());
    ofs_states_ << trace_id_ << ", "
                << timestamp << ", "
                << bias.accelerometer().x() << ", "
                << bias.accelerometer().y() << ", "
                << bias.accelerometer().z() << ", "
                << bias.gyroscope().x() << ", "
                << bias.gyroscope().y() << ", "
                << bias.gyroscope().z() << ", "
                << velocity.x() << ", "
                << velocity.y() << ", "
                << velocity.z() << ", "
                << T_wi.translation().x() << ", "
                << T_wi.translation().y() << ", "
                << T_wi.translation().z() << ", "
                << q.x() << ", "
                << q.y() << ", "
                << q.z() << ", "
                << q.w() << std::endl;
  }
  ++trace_id_;
}
*/

void BackendVisualizer::traceCovariance(
    const Eigen::Matrix<double, 6, 6>& C,
    const int64_t timestamp)
{
  if(!ofs_covariance_.is_open())
  {
    std::string states_trace_name(trace_dir_+"/estimate_covariance.csv");
    ofs_covariance_.open(states_trace_name.c_str());
    ofs_covariance_.precision(10);
  }
  CHECK(ofs_covariance_.is_open());

  ofs_covariance_
      << timestamp << ", "
      << C(0,0) << "," << C(0,1) << "," << C(0,2) << "," << C(0,3) << "," << C(0,4) << "," << C(0,5) << ","
      << C(1,0) << "," << C(1,1) << "," << C(1,2) << "," << C(1,3) << "," << C(1,4) << "," << C(1,5) << ","
      << C(2,0) << "," << C(2,1) << "," << C(2,2) << "," << C(2,3) << "," << C(2,4) << "," << C(2,5) << ","
      << C(3,0) << "," << C(3,1) << "," << C(3,2) << "," << C(3,3) << "," << C(3,4) << "," << C(3,5) << ","
      << C(4,0) << "," << C(4,1) << "," << C(4,2) << "," << C(4,3) << "," << C(4,4) << "," << C(4,5) << ","
      << C(5,0) << "," << C(5,1) << "," << C(5,2) << "," << C(5,3) << "," << C(5,4) << "," << C(5,5)
      << std::endl;
}

} // namespace svo
