#include <svo/tracker/feature_tracking_types.h>
#include <svo/common/frame.h>

namespace svo {

// -----------------------------------------------------------------------------
FeatureRef::FeatureRef(
    const FrameBundlePtr& frame_bundle, size_t frame_index, size_t feature_index)
  : frame_bundle_(frame_bundle)
  , frame_index_(frame_index)
  , feature_index_(feature_index)
{
  CHECK_LT(frame_index_, frame_bundle_->size());
}

const Eigen::Block<Keypoints, 2, 1> FeatureRef::getPx() const
{
  return frame_bundle_->at(frame_index_)->px_vec_.block<2,1>(0, feature_index_);
}

const Eigen::Block<Bearings, 3, 1> FeatureRef::getBearing() const
{
  return frame_bundle_->at(frame_index_)->f_vec_.block<3,1>(0, feature_index_);
}

const FramePtr FeatureRef::getFrame() const
{
  return frame_bundle_->at(frame_index_);
}

// -----------------------------------------------------------------------------
FeatureTrack::FeatureTrack(int track_id)
  : track_id_(track_id)
{
  feature_track_.reserve(10);
}

double FeatureTrack::getDisparity() const
{
  return (front().getPx() - back().getPx()).norm();
}

} // namespace svo
