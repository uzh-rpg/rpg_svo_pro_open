// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/frame_handler_array.h>
#include <svo/map.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/pose_optimizer.h>
#include <svo/img_align/sparse_img_align.h>
#include <svo/direct/depth_filter.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>
#include <vikit/performance_monitor.h>

namespace svo {

FrameHandlerArray::FrameHandlerArray(
    const BaseOptions& base_options,
      const DepthFilterOptions& depth_filter_options,
      const DetectorOptions& feature_detector_options,
      const InitializationOptions& init_options,
      const ReprojectorOptions& reprojector_options,
      const FeatureTrackerOptions& tracker_options,
      const CameraBundle::Ptr& cameras)
  : FrameHandlerBase(
      base_options, reprojector_options, depth_filter_options,
      feature_detector_options, init_options, tracker_options, cameras)
{ ; }

UpdateResult FrameHandlerArray::processFrameBundle()
{
  UpdateResult res = UpdateResult::kFailure;
  if(stage_ == Stage::kTracking)
    res = processFrame();
  else if(stage_ == Stage::kInitializing)
    res = processSecondFrame();
  else if(stage_ == Stage::kInitializing)
    res = processFirstFrame();

  return res;
}

void FrameHandlerArray::addImages(
    const std::vector<cv::Mat>& images,
    const uint64_t timestamp)
{
  // TODO: deprecated
  addImageBundle(images, timestamp);
}

UpdateResult FrameHandlerArray::processFirstFrame()
{
  // Add first frame to initializer. It may return a failure if not enough features
  // can be detected, i.e., we are in a texture-less area.
  initializer_->setDepthPrior(options_.init_map_scale);
  if(initializer_->addFrameBundle(new_frames_) == InitResult::kFailure)
    return UpdateResult::kDefault;

  stage_ = Stage::kTracking;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerArray::processSecondFrame()
{
  vk::Timer t;

  initializer_->setDepthPrior(options_.init_map_scale);
  auto res = initializer_->addFrameBundle(new_frames_);
  SVO_INFO_STREAM("Init: Processing took " << t.stop()*1000 << "ms");

  if(res == InitResult::kFailure)
    return UpdateResult::kFailure;
  else if(res == InitResult::kNoKeyframe)
    return UpdateResult::kDefault;

  // make old frame keyframe
  for(const FramePtr& frame : initializer_->frames_ref_->frames_)
  {
    frame->setKeyframe();
    map_->addKeyframe(frame,
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  }

  // make new frame keyframe
  for(const FramePtr& frame : new_frames_->frames_)
  {
    frame->setKeyframe();
    frame_utils::getSceneDepth(frame, depth_median_, depth_min_, depth_max_);
    depth_filter_->addKeyframe(
                frame, depth_median_, 0.5*depth_min_, depth_median_*1.5);
    map_->addKeyframe(frame,
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  }

  stage_ = Stage::kTracking;
  initializer_->reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerArray::processFrame()
{
  // ---------------------------------------------------------------------------
  // tracking

  // STEP 1: Sparse Image Align
  size_t n_tracked_features = 0;
  sparseImageAlignment();

  // STEP 2: Map Reprojection & Feature Align
  n_tracked_features = projectMapInFrame();
  if(n_tracked_features < options_.quality_min_fts)
    return UpdateResult::kFailure;

  // STEP 3: Pose & Structure Optimization
  n_tracked_features = optimizePose();
  if(n_tracked_features < options_.quality_min_fts)
    return UpdateResult::kFailure;
  optimizeStructure(new_frames_, options_.structure_optimization_max_pts, 5);

  // return if tracking bad
  setTrackingQuality(n_tracked_features);
  if(tracking_quality_ == TrackingQuality::kInsufficient)
    return UpdateResult::kFailure;

  // ---------------------------------------------------------------------------
  // select keyframe
  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  //if(!need_new_kf_(new_frames_->at(0)->T_f_w_))
  if(frame_counter_ % 4 != 0)
  {
    for(size_t i=0; i<new_frames_->size(); ++i)
      depth_filter_->updateSeeds(overlap_kfs_.at(i), new_frames_->at(i));
    return UpdateResult::kDefault;
  }
  SVO_DEBUG_STREAM("New keyframe selected.");

  for(size_t i = 0; i<new_frames_->size(); ++i)
    makeKeyframe(i);

  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerArray::makeKeyframe(const size_t camera_id)
{
  const FramePtr& frame = new_frames_->at(camera_id);

  // ---------------------------------------------------------------------------
  // new keyframe selected
  frame->setKeyframe();
  map_->addKeyframe(frame,
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  upgradeSeedsToFeatures(frame);

  // init new depth-filters, set feature-detection grid-cells occupied that
  // already have a feature
  {
    DepthFilter::ulock_t lock(depth_filter_->feature_detector_mut_);
    setDetectorOccupiedCells(camera_id, depth_filter_->feature_detector_);
  } // release lock
  double depth_median = -1, depth_min, depth_max;
  if(!frame_utils::getSceneDepth(frame, depth_median, depth_min, depth_max))
  {
    depth_min = 0.2; depth_median = 3.0; depth_max = 100;
  }
  SVO_DEBUG_STREAM("Average Depth " << frame->cam()->getLabel() << ": " << depth_median);
  depth_filter_->addKeyframe(
        new_frames_->at(camera_id), depth_median, 0.5*depth_min, depth_median*1.5);
  depth_filter_->updateSeeds(overlap_kfs_.at(camera_id), frame);

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
      FramePtr furthest_frame = map_->getFurthestKeyframe(frame->pos());
      map_->removeKeyframe(furthest_frame->id());
    }
  }
  return UpdateResult::kKeyframe;
}

void FrameHandlerArray::resetAll()
{
  backend_scale_initialized_ = true;
  resetVisionFrontendCommon();
  depth_filter_->reset();
}

} // namespace svo
