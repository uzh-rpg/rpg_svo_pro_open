#pragma once

#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/transformation.h>
#include <svo/tracker/feature_tracking_types.h>

namespace svo {

// Forward declarations.
class AbstractDetector;
struct DetectorOptions;

class FeatureTracker
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<FeatureTracker> Ptr;

  FeatureTracker() = delete;

  /// Provide number of frames per frame_bundle
  FeatureTracker(
      const FeatureTrackerOptions& options,
      const DetectorOptions& detector_options,
      const CameraBundlePtr& cams);

  /// Tracks current features and if many tracks terminate, automatically
  /// initializes new tracks.
  void trackAndDetect(const FrameBundlePtr& nframe_kp1);

  /// Tracks a frame bundle. Returns number of tracked features.
  size_t trackFrameBundle(const FrameBundlePtr& nframe_kp1);

  /// Extract new features
  size_t initializeNewTracks(const FrameBundlePtr& nframe_k);

  const FeatureTracks& getActiveTracks(size_t frame_index) const;

  size_t getTotalActiveTracks() const;

  /// pivot_ration needs to be in range(0,1) and if 0.5 the disparity that
  /// is returned per frame is the median. If 0.25 it means that 25% of the
  /// tracks have a higher disparity than returned.
  void getNumTrackedAndDisparityPerFrame(
      double pivot_ratio,
      std::vector<size_t>* num_tracked,
      std::vector<double>* disparity) const;

  FrameBundlePtr getOldestFrameInTrack(size_t frame_index) const;

  void resetActiveTracks();

  void resetTerminatedTracks();

  /// Clear all stored data.
  void reset();

//private:

  FeatureTrackerOptions options_;

  // A vector for each frame in the bundle.
  const size_t bundle_size_;
  std::vector<std::shared_ptr<AbstractDetector>> detectors_;
  std::vector<FeatureTracks,
  Eigen::aligned_allocator<FeatureTracks> > active_tracks_;
  std::vector<FeatureTracks,
  Eigen::aligned_allocator<FeatureTracks> > terminated_tracks_;
};

} // namespace svo
