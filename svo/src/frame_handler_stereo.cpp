// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/frame_handler_stereo.h>
#include <svo/map.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/pose_optimizer.h>
#include <svo/img_align/sparse_img_align.h>
#include <svo/direct/depth_filter.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>
#include <vikit/performance_monitor.h>

namespace svo {

FrameHandlerStereo::FrameHandlerStereo(
    const BaseOptions& base_options,
    const DepthFilterOptions& depth_filter_options,
    const DetectorOptions& feature_detector_options,
    const InitializationOptions& init_options,
    const StereoTriangulationOptions& stereo_options,
    const ReprojectorOptions& reprojector_options,
    const FeatureTrackerOptions& tracker_options,
    const CameraBundle::Ptr& stereo_camera)
  : FrameHandlerBase(
      base_options, reprojector_options, depth_filter_options,
      feature_detector_options, init_options, tracker_options, stereo_camera)
{
  // init initializer
  stereo_triangulation_.reset(
        new StereoTriangulation(
          stereo_options,
          feature_detection_utils::makeDetector(
            feature_detector_options, cams_->getCameraShared(0))));
}

UpdateResult FrameHandlerStereo::processFrameBundle()
{
  UpdateResult res = UpdateResult::kFailure;
  if(stage_ == Stage::kTracking)
    res = processFrame();
  else if(stage_ == Stage::kInitializing)
    res = processFirstFrame();
  return res;
}

void FrameHandlerStereo::addImages(
    const cv::Mat& img_left,
    const cv::Mat& img_right,
    const uint64_t timestamp)
{
  // TODO: deprecated
  addImageBundle({img_left, img_right}, timestamp);
}

UpdateResult FrameHandlerStereo::processFirstFrame()
{
  if(initializer_->addFrameBundle(new_frames_) == InitResult::kFailure)
  {
    SVO_ERROR_STREAM("Initialization failed. Not enough triangulated points.");
    return UpdateResult::kDefault;
  }

  new_frames_->at(0)->setKeyframe();
  map_->addKeyframe(new_frames_->at(0),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  new_frames_->at(1)->setKeyframe();
  map_->addKeyframe(new_frames_->at(1),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);

  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  depth_filter_->addKeyframe(new_frames_->at(0), depth_median_, 0.5*depth_min_, depth_median_*1.5);

  SVO_INFO_STREAM("Init: Selected first frame.");
  stage_ = Stage::kTracking;
  tracking_quality_ = TrackingQuality::kGood;
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerStereo::processFrame()
{
  // ---------------------------------------------------------------------------
  // tracking

  // STEP 1: Sparse Image Align
  size_t n_tracked_features = 0;
  sparseImageAlignment();

  // STEP 2: Map Reprojection & Feature Align
  n_tracked_features = projectMapInFrame();
  if(n_tracked_features < options_.quality_min_fts)
  {
    return makeKeyframe(); // force stereo triangulation to recover
  }

  // STEP 3: Pose & Structure Optimization
  if(bundle_adjustment_type_!=BundleAdjustmentType::kCeres)
  {
    n_tracked_features = optimizePose();
    if(n_tracked_features < options_.quality_min_fts)
    {
      return makeKeyframe(); // force stereo triangulation to recover
    }
    optimizeStructure(new_frames_, options_.structure_optimization_max_pts, 5);
  }
  // return if tracking bad
  setTrackingQuality(n_tracked_features);
  if(tracking_quality_ == TrackingQuality::kInsufficient)
  {
    return makeKeyframe(); // force stereo triangulation to recover
  }

  // ---------------------------------------------------------------------------
  // select keyframe
  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  if(!need_new_kf_(new_frames_->at(0)->T_f_w_))
  {
    for(size_t i=0; i<new_frames_->size(); ++i)
      depth_filter_->updateSeeds(overlap_kfs_.at(i), new_frames_->at(i));
    return UpdateResult::kDefault;
  }
  SVO_DEBUG_STREAM("New keyframe selected.");
  return makeKeyframe();
}

UpdateResult FrameHandlerStereo::makeKeyframe()
{
  static size_t kf_counter = 0;
  const size_t kf_id = kf_counter++ % cams_->numCameras();
  const size_t other_id = kf_counter % cams_->numCameras();
  CHECK(kf_id != other_id);

  // ---------------------------------------------------------------------------
  // add extra features when num tracked is critically low!
  if(new_frames_->numLandmarks() < options_.kfselect_numkfs_lower_thresh)
  {
    setDetectorOccupiedCells(0, stereo_triangulation_->feature_detector_);
    new_frames_->at(other_id)->setKeyframe();
    map_->addKeyframe(new_frames_->at(other_id),
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
    upgradeSeedsToFeatures(new_frames_->at(other_id));
    stereo_triangulation_->compute(new_frames_->at(0), new_frames_->at(1));
  }

  // ---------------------------------------------------------------------------
  // new keyframe selected
  new_frames_->at(kf_id)->setKeyframe();
  map_->addKeyframe(new_frames_->at(kf_id),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  upgradeSeedsToFeatures(new_frames_->at(kf_id));

  // init new depth-filters, set feature-detection grid-cells occupied that
  // already have a feature
  {
    DepthFilter::ulock_t lock(depth_filter_->feature_detector_mut_);
    setDetectorOccupiedCells(kf_id, depth_filter_->feature_detector_);
  } // release lock
  depth_filter_->addKeyframe(
        new_frames_->at(kf_id), depth_median_, 0.5*depth_min_, depth_median_*1.5);
  depth_filter_->updateSeeds(overlap_kfs_.at(0), new_frames_->at(0));
  depth_filter_->updateSeeds(overlap_kfs_.at(1), new_frames_->at(1));

  // TEST
  // {
  if(options_.update_seeds_with_old_keyframes)
  {
    depth_filter_->updateSeeds({ new_frames_->at(0) }, last_frames_->at(0));
    depth_filter_->updateSeeds({ new_frames_->at(0) }, last_frames_->at(1));
    depth_filter_->updateSeeds({ new_frames_->at(1) }, last_frames_->at(0));
    depth_filter_->updateSeeds({ new_frames_->at(1) }, last_frames_->at(1));
    for(const FramePtr& old_keyframe : overlap_kfs_.at(0))
    {
      depth_filter_->updateSeeds({ new_frames_->at(0) }, old_keyframe);
      depth_filter_->updateSeeds({ new_frames_->at(1) }, old_keyframe);
    }
  }
  // }

  // if limited number of keyframes, remove the one furthest apart
  while(map_->size() > options_.max_n_kfs && options_.max_n_kfs > 2)
  {
    if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
    {
      // deal differently with map for ceres backend
      map_->removeOldestKeyframe();
    }
    else
    {
      FramePtr furthest_frame =
          map_->getFurthestKeyframe(new_frames_->at(kf_id)->pos());
      map_->removeKeyframe(furthest_frame->id());
    }
  }
  return UpdateResult::kKeyframe;
}

void FrameHandlerStereo::resetAll()
{
  backend_scale_initialized_ = true;
  resetVisionFrontendCommon();
}

} // namespace svo
