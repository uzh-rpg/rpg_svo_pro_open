#include "svo/ceres_backend_publisher.hpp"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>

#include <vikit/output_helper.h>
#include <vikit/params_helper.h>

namespace
{
template <typename T>
void normalizeVector(const std::vector<T>& in, std::vector<float>* out)
{
  auto res = std::minmax_element(in.begin(), in.end());
  const T min = *res.first;
  const T max = *res.second;
  const float dist = static_cast<float>(max - min);

  out->resize(in.size());
  for (size_t i = 0; i < in.size(); i++)
  {
    (*out)[i] = (in[i] - min) / dist;
  }
}
}

namespace svo
{
CeresBackendPublisher::CeresBackendPublisher(
    const ros::NodeHandle& nh_private,
    const std::shared_ptr<ceres_backend::Map>& map_ptr)
  : pnh_(nh_private)
  , map_ptr_(map_ptr)
{
  pub_imu_pose_ = pnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
      "backend_pose_imu", 10);
  pub_imu_pose_viz_ =
      pnh_.advertise<geometry_msgs::PoseStamped>("backend_pose_imu_viz", 10);
  pub_points_ =
      pnh_.advertise<PointCloud>("backend_points", 10);
}

void CeresBackendPublisher::publish(const ViNodeState& state,
                                    const int64_t timestamp,
                                    const int32_t seq)
{
  publishImuPose(state, timestamp, seq);
  publishBackendLandmarks(timestamp);
}

void CeresBackendPublisher::publishImuPose(const ViNodeState& state,
                                           const int64_t timestamp,
                                           const int32_t seq)
{
  // Trace state
  state_ = state;

  {
    std::lock_guard<std::mutex> lock(mutex_frame_id_);
    state_frame_id_ = BundleId(seq);
  }

  size_t n_pose_sub = pub_imu_pose_.getNumSubscribers();
  size_t n_pose_viz_sub = pub_imu_pose_viz_.getNumSubscribers();
  if (n_pose_sub == 0 && n_pose_viz_sub == 0)
  {
    return;
  }
  VLOG(100) << "Publish IMU Pose";

  Eigen::Quaterniond q = state.get_T_W_B().getRotation().toImplementation();
  Eigen::Vector3d p = state.get_T_W_B().getPosition();
  ros::Time time = ros::Time().fromNSec(timestamp);

  if (n_pose_sub > 0)
  {
    geometry_msgs::PoseWithCovarianceStampedPtr msg_pose(
        new geometry_msgs::PoseWithCovarianceStamped);
    msg_pose->header.seq = seq;
    msg_pose->header.stamp = time;
    msg_pose->header.frame_id = kWorldFrame;
    msg_pose->pose.pose.position.x = p[0];
    msg_pose->pose.pose.position.y = p[1];
    msg_pose->pose.pose.position.z = p[2];
    msg_pose->pose.pose.orientation.x = q.x();
    msg_pose->pose.pose.orientation.y = q.y();
    msg_pose->pose.pose.orientation.z = q.z();
    msg_pose->pose.pose.orientation.w = q.w();
    for (size_t i = 0; i < 36; ++i)
      msg_pose->pose.covariance[i] = 0;
    pub_imu_pose_.publish(msg_pose);
  }

  if (n_pose_viz_sub > 0)
  {
    geometry_msgs::PoseStampedPtr msg_pose(new geometry_msgs::PoseStamped);
    msg_pose->header.seq = seq;
    msg_pose->header.stamp = time;
    msg_pose->header.frame_id = kWorldFrame;
    msg_pose->pose.position.x = p[0];
    msg_pose->pose.position.y = p[1];
    msg_pose->pose.position.z = p[2];
    msg_pose->pose.orientation.x = q.x();
    msg_pose->pose.orientation.y = q.y();
    msg_pose->pose.orientation.z = q.z();
    msg_pose->pose.orientation.w = q.w();
    pub_imu_pose_viz_.publish(msg_pose);
  }
}

void CeresBackendPublisher::publishBackendLandmarks(
    const int64_t timestamp) const
{
  if (pub_points_.getNumSubscribers() == 0)
  {
    return;
  }

  // get all landmarks
  const std::unordered_map<
      uint64_t, std::shared_ptr<ceres_backend::ParameterBlock> >& idmap =
      map_ptr_->idToParameterBlockMap();
  size_t n_pts = 0;
  std::vector<const double*> landmark_pointers;
  std::vector<uint64_t> point_ids;
  for (auto& it : idmap)
  {
    if (it.second->typeInfo() == "HomogeneousPointParameterBlock" &&
        !it.second->fixed())
    {
      n_pts++;
      landmark_pointers.push_back(it.second->parameters());
      point_ids.push_back(it.second->id());
    }
  }

  if (n_pts < 5)
  {
    return;
  }

  std::vector<float> intensities;
  normalizeVector(point_ids, &intensities);

  // point clound to publish
  PointCloud pc;
  ros::Time pub_time;
  pub_time.fromNSec(timestamp);
  pcl_conversions::toPCL(pub_time, pc.header.stamp);
  pc.header.frame_id = kWorldFrame;
  pc.reserve(n_pts);
  for(size_t i = 0; i < landmark_pointers.size(); i++)
  {
    const auto p = landmark_pointers[i];
    PointType pt;
    pt.intensity = intensities[i];
    pt.x = p[0];
    pt.y = p[1];
    pt.z = p[2];
    pc.push_back(pt);
  }
  pub_points_.publish(pc);
}

}  // namespace svo
