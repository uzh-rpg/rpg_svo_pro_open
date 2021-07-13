#include <ros/ros.h>
#include <svo_ros/svo_nodelet.h>
#include <svo_ros/svo_interface.h>
#include <pluginlib/class_list_macros.h>
#include <vikit/params_helper.h>

PLUGINLIB_EXPORT_CLASS(svo::SvoNodelet, nodelet::Nodelet)

namespace svo {

SvoNodelet::~SvoNodelet()
{
  NODELET_INFO_STREAM("SVO quit");
  svo_interface_->quit_ = true;
}

void SvoNodelet::onInit()
{
  ros::NodeHandle nh(getNodeHandle());
  ros::NodeHandle pnh(getPrivateNodeHandle());

  NODELET_INFO_STREAM("Initialized " <<  getName() << " nodelet.");
  svo::PipelineType type = svo::PipelineType::kMono;
  if(vk::param<bool>(pnh, "pipeline_is_stereo", false))
    type = svo::PipelineType::kStereo;

  svo_interface_.reset(new SvoInterface(type, nh, pnh));
  if(svo_interface_->imu_handler_)
    svo_interface_->subscribeImu();
  svo_interface_->subscribeImage();
  svo_interface_->subscribeRemoteKey();
}

} // namespace svo
