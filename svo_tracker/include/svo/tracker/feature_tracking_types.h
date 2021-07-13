#pragma once

#include <glog/logging.h>
#include <svo/common/types.h>

namespace svo {

// -----------------------------------------------------------------------------
struct FeatureTrackerOptions
{
  /// We do the Lucas Kanade tracking in a pyramidal way. max_level specifies the
  /// coarsest pyramidal level to optimize. For an image resolution of (640x480)
  /// we set this variable to 4 if you have an image with double the resolution,
  /// increase this number by one.
  int klt_max_level = 4;

  /// Similar to klt_max_level, this is the coarsest level to search for.
  /// if you have a really high resolution image and you don't extract
  /// features down to the lowest level you can set this number larger than 0.
  int klt_min_level = 0;

  /// Patch-size to use on each pyramid level.
  std::vector<int> klt_patch_sizes = {16, 16, 16, 8, 8};

  /// KLT termination criterion.
  int klt_max_iter = 30;

  /// KLT termination criterion.
  double klt_min_update_squared = 0.001;

  /// Use the first observation as klt template. If set to false, then the
  /// last observation is used, which results in more feature drift.
  bool klt_template_is_first_observation = true;

  /// If number of tracks falls below this threshold, detect new features.
  size_t min_tracks_to_detect_new_features = 50;

  /// Reset tracker before detecting new features. This means that all active
  /// tracks are always the same age.
  bool reset_before_detection = true;
};

// -----------------------------------------------------------------------------
class FeatureRef
{
public:
  FeatureRef() = delete;

  FeatureRef(
      const FrameBundlePtr& frame_bundle, size_t frame_index, size_t feature_index);

  inline const FrameBundlePtr getFrameBundle() const {
    return frame_bundle_;
  }

  inline size_t getFrameIndex() const {
    return frame_index_;
  }

  inline size_t getFeatureIndex() const {
    return feature_index_;
  }

  const Eigen::Block<Keypoints, 2, 1> getPx() const;

  const Eigen::Block<Bearings, 3, 1> getBearing() const;

  const FramePtr getFrame() const;

private:
  FrameBundlePtr frame_bundle_;
  size_t frame_index_;
  size_t feature_index_;
};

typedef std::vector<FeatureRef> FeatureRefList;


// -----------------------------------------------------------------------------
class FeatureTrack
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit FeatureTrack(int track_id);

  inline int getTrackId() const {
    return track_id_;
  }

  inline const FeatureRefList& getFeatureTrack() const {
    return feature_track_;
  }

  inline size_t size() const {
    return feature_track_.size();
  }

  inline bool empty() const {
    return feature_track_.empty();
  }

  /// The feature at the front is the first observed feature.
  inline const FeatureRef& front() const {
    CHECK(!empty()) << "Track empty when calling front().";
    return feature_track_.front();
  }

  /// The feature at the back is the last observed feature.
  inline const FeatureRef& back() const {
    CHECK(!empty()) << "Track empty when calling back().";
    return feature_track_.back();
  }

  inline const FeatureRef& at(size_t i) const {
    CHECK_LT(i, feature_track_.size()) << "Index too large.";
    return feature_track_.at(i);
  }

  /// New observations are always inserted at the back of the vector.
  inline void pushBack(
      const FrameBundlePtr& frame_bundle,
      const size_t frame_index,
      const size_t feature_index) {
    feature_track_.emplace_back(FeatureRef(frame_bundle, frame_index, feature_index));
  }

  double getDisparity() const;

private:
  int track_id_;
  FeatureRefList feature_track_;
};
typedef std::vector<FeatureTrack,
Eigen::aligned_allocator<FeatureTrack> > FeatureTracks;

} // namespace svo
