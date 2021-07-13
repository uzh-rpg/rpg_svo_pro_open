// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/frame_handler_base.h>

namespace svo {

class FrameHandlerArray : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<FrameHandlerArray> Ptr;

  /// Default constructor
  FrameHandlerArray(
      const BaseOptions& base_options,
      const DepthFilterOptions& depth_filter_options,
      const DetectorOptions& feature_detector_options,
      const InitializationOptions& init_options,
      const ReprojectorOptions& reprojector_options,
      const FeatureTrackerOptions& tracker_options,
      const CameraBundle::Ptr& cameras);

  virtual ~FrameHandlerArray() = default;

  // deprecated. use addImageBundle().
  void addImages(
      const std::vector<cv::Mat>& images,
      const uint64_t timestamp);

  const FrameBundlePtr& lastFrames() const
  {
    return last_frames_;
  }
  
protected:

  /// Pipeline implementation. Called by base class.
  virtual UpdateResult processFrameBundle() override;

  UpdateResult processFirstFrame();

  UpdateResult processSecondFrame();

  UpdateResult processFrame();

  UpdateResult makeKeyframe(const size_t camera_id);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll() override;
};

} // namespace svo
