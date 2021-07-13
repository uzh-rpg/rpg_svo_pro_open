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

// forward declaration
class StereoTriangulationOptions;
class StereoTriangulation;
typedef std::shared_ptr<StereoTriangulation> StereoTriangulationPtr;

/// Monocular Semi-Direct Visual Odometry Pipeline
///
/// References:
/// [1] Christian Forster, Matia Pizzoli, Davide Scaramuzza, "SVO: Semi-Direct
/// Monocular Visual Odometry for Micro Aerial Vehicles", Proc. IEEE International
/// Conference on Robotics and Automation, Honkong 2015.
///
/// This is the main interface class of the VO. It derives from FrameHandlerBase
/// which maintains the state machine (start, stop, reset).
class FrameHandlerStereo : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<FrameHandlerStereo> Ptr;

  /// Default constructor
  FrameHandlerStereo(
      const BaseOptions& base_options,
      const DepthFilterOptions& depth_filter_options,
      const DetectorOptions& feature_detector_options,
      const InitializationOptions& init_options,
      const StereoTriangulationOptions& stereo_options,
      const ReprojectorOptions& reprojector_options,
      const FeatureTrackerOptions& tracker_options,
      const CameraBundle::Ptr& stereo_camera);

  virtual ~FrameHandlerStereo() = default;

  /// @name Main Interface
  ///
  /// These are the main functions that you will need to interface
  /// with the visual odometry pipeline. Other important functions like
  /// start(), reset(), stage() are provided in the FrameHandlerBase class which
  /// maintains the state-machine.
  ///
  /// @{

  /// Provide an image to the odometry pipeline
  /// \param img A 8bit grayscale OpenCV image.
  /// \param timestamp Corresponding timestamp of the image. Is saved in the
  /// frame class.
  /// \param R_origin_imu If you use a IMU gyroscope and you integrate it outside
  /// of SVO, you can provide the orientation here. In order to be used, you have
  /// to set option->use_provided_attitude to true. If you set the priors in the
  /// options larger than one the gyroscope measurement is used as weighted prior
  /// in the optimizations. Note that you have to provide the orientation of the
  /// IMU frame and not the camera. The relative transformation between imu and
  /// camera is set inside cam_. Another way to provide an orientation estimate
  /// is to use set the imuHandler and to provide it with rotation rate
  /// measurements.
  /// \param custom_id If you would like to save an identifier key with the the
  /// frame you can provide it here. It is not used by SVO for anything but can
  /// be accessed with frame->customId().
  // deprecated. use addImageBundle().
  void addImages(
      const cv::Mat& img_left,
      const cv::Mat& img_right,
      const uint64_t timestamp);

  /// After adding an image to SVO with addImage(), the image is saved in a
  /// Frame class, the frame is processed and it's pose computed. Most likely you
  /// want to know the pose of the cam. Therefore, use this function to access
  /// the pose of the last provided image (e.g., vo->lastFrame()->T_world_cam()).
  /// \return FramePtr !!!IMPORTANT!!! can be nullptr if something did not work
  /// well or if you call this function when pipeline is not running, i.e.,
  /// vo->stage()==STAGE_PAUSED.
  const FrameBundlePtr& lastFrames() const
  {
    return last_frames_;
  }

  /// @}

  /// @name Debug Interface
  /// These parameters should be private but are currently not for easier debugging.
  /// It is unlikely that you need them.
  ///
  /// @{

  // SVO Modules:
  StereoTriangulationPtr stereo_triangulation_;

  /// @}

protected:

  /// Pipeline implementation. Called by base class.
  virtual UpdateResult processFrameBundle() override;

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  virtual UpdateResult makeKeyframe();

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll() override;
};

} // namespace svo
