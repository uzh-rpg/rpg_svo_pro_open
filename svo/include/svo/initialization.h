// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <svo/common/camera_fwd.h>

namespace svo {

// forward declarations
class StereoTriangulation;
class FeatureTracker;
using FeatureTrackerUniquePtr = std::unique_ptr<FeatureTracker>;
class AbstractDetector;
struct DetectorOptions;
struct FeatureTrackerOptions;

enum class InitializerType {
  kHomography,       ///< Estimates a plane from the first two views
  kTwoPoint,         ///< Assumes known rotation from IMU and estimates translation
  kFivePoint,        ///< Estimate relative pose of two cameras using 5pt RANSAC
  kOneShot,          ///< Initialize points on a plane with given depth
  kStereo,           ///< Triangulate from two views with known pose
  kArrayGeometric,   ///< Estimate relative pose of two camera arrays, using 17pt RANSAC
  kArrayOptimization ///< Estimate relative pose of two camera arrays using GTSAM
};

/// Common options for all initializers
/// Options marked with (!) are more important.
struct InitializationOptions
{

  /// Initializer method. See options above.
  InitializerType init_type = InitializerType::kHomography;

  /// (!) Minimum disparity (length of feature tracks) required to select the
  /// second frame. After that we triangulate the first pointcloud. For easy
  /// initialization you want to make this small (minimum 20px) but actually
  /// it is much better to have more disparity to ensure the initial pointcloud
  /// is good.
  double init_min_disparity = 50.0;

  /// When checking whether the disparity of the tracked features are large enough,
  /// we check that a certain percentage of tracked features have disparities large
  /// than init_min_disparity. The default percentage is 0.5, which means the median
  /// is checked. For example, if this parameter is set to 0.25, it means we go for
  /// realtive pose estimation only when at least 25% of the tracked features have
  ///  disparities large than init_min_disparity
  double init_disparity_pivot_ratio = 0.5;

  /// (!) If less features than init_min_features can be extracted at the
  /// first frame, the first frame is not accepted and we check the next frame.
  size_t init_min_features = 100;

  /// (!) This threshold defines how many features should be tracked in the
  /// first place. Basically, the initializer tries to extract and track
  /// init_min_features*init_min_features_factor features.
  double init_min_features_factor = 2.5;

  /// (!) If number of tracked features during initialization falls below this
  /// threshold than the initializer returns FAILURE.
  size_t init_min_tracked = 50;

  /// (!) At the end of initialization, we triangulate the first pointcloud
  /// and check the quality of the triangulation by evaluating the reprojection
  /// errors. All points that have more reprojection error than reproj_error_thresh
  /// are considered outliers. Only return SUCCESS if we have more inliers
  /// than  init_min_inliers.
  size_t init_min_inliers = 40;

  /// Reprojection threshold in pixels. The same as we also use in pose optimizer.
  double reproj_error_thresh = 2.0;


  // TODO: what are these for? (introduced by Art)
  double expected_avg_depth = 1.0;
  double init_min_depth_error = 1.0;
};

enum class InitResult
{
  kFailure,
  kNoKeyframe,
  kTracking,
  kSuccess
};

/// Bootstrapping the map from the first two views.
class AbstractInitialization
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using BearingVectors = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
  using Ptr = std::shared_ptr<AbstractInitialization>;
  using UniquePtr = std::unique_ptr<AbstractInitialization>;
  using InlierMask = Eigen::Matrix<bool, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using FeatureMatches = std::vector<std::pair<size_t, size_t>>;

  InitializationOptions options_;       //!< Initializer options
  FeatureTrackerUniquePtr tracker_;     //!< Feature tracker
  FrameBundlePtr frames_ref_;           //!< reference frames
  Transformation T_cur_from_ref_;       //!< computed transformation between the first two frames.
  Quaternion R_ref_world_;
  Quaternion R_cur_world_;              //!< Absolute orientation prior.
  Eigen::Vector3d t_ref_cur_;           //!< Translation prior
  bool have_rotation_prior_ = false;    //!< Do we have a rotation prior?
  bool have_translation_prior_ = false; //!< Do we have a translation prior?
  bool have_depth_prior_ = false;       //!< Do we have a depth prior?
  double depth_at_current_frame_ = 1.0; //!< Depth prior of 3D points in current frame

  AbstractInitialization(
      const InitializationOptions& init_options,
      const FeatureTrackerOptions& tracker_options,
      const DetectorOptions& detector_options,
      const CameraBundlePtr& cams);

  virtual ~AbstractInitialization();

  bool trackFeaturesAndCheckDisparity(const FrameBundlePtr& frames);

  virtual InitResult addFrameBundle(const FrameBundlePtr& frames_cur) = 0;

  virtual void reset();

  inline void setAbsoluteOrientationPrior(const Quaternion& R_cam_world)
  {
    R_cur_world_ = R_cam_world;
    have_rotation_prior_ = true;
  }

  inline void setTranslationPrior(const Eigen::Vector3d& t_ref_cur) {
    t_ref_cur_ = t_ref_cur;
    have_translation_prior_ = true;
  }

  inline void setDepthPrior(double depth) {
    depth_at_current_frame_ = depth;
    have_depth_prior_ = true;
  }

};

/// Tracks features using Lucas-Kanade tracker and then estimates a homography.
class HomographyInit : public AbstractInitialization
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using AbstractInitialization::AbstractInitialization;
  virtual ~HomographyInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

/// Tracks features using KLT and then uses the IMU rotation prior to estimate
/// the relative pose using a 2Pt RANSAC
class TwoPointInit : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~TwoPointInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

/// Tracks features using KLT and then estimate the relative pose using 5pt RANSAC
class FivePointInit : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~FivePointInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

/// Assumes horizontal ground and given depth. Initializes all features in the
/// plane. Used for autonomous take-off.
class OneShotInit : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~OneShotInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

class StereoInit : public AbstractInitialization
{
public:

  StereoInit(
      const InitializationOptions& init_options,
      const FeatureTrackerOptions& tracker_options,
      const DetectorOptions& detector_options,
      const CameraBundlePtr& cams);

  virtual ~StereoInit() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;

  std::unique_ptr<StereoTriangulation> stereo_;
  std::shared_ptr<AbstractDetector> detector_;
};

class ArrayInitGeometric : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~ArrayInitGeometric() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};

class ArrayInitOptimization : public AbstractInitialization
{
public:
  using AbstractInitialization::AbstractInitialization;
  virtual ~ArrayInitOptimization() = default;

  virtual InitResult addFrameBundle(
      const FrameBundlePtr& frames_cur) override;
};


namespace initialization_utils {

bool triangulateAndInitializePoints(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref,
    const Transformation& T_cur_ref,
    const double reprojection_threshold,
    const double depth_at_current_frame,
    const size_t min_inliers_threshold,
    AbstractInitialization::FeatureMatches& matches_cur_ref);

void triangulatePoints(
    const Frame& frame_cur,
    const Frame& frame_ref,
    const Transformation& T_cur_ref,
    const double reprojection_threshold,
    AbstractInitialization::FeatureMatches& matches_cur_ref,
    Positions& points_in_cur);

void rescaleAndInitializePoints(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref,
    const AbstractInitialization::FeatureMatches& matches_cur_ref,
    const Positions& points_in_cur,
    const Transformation& T_cur_ref,
    const double depth_at_current_frame);

void displayFeatureTracks(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref);

AbstractInitialization::UniquePtr makeInitializer(
    const InitializationOptions& init_options,
    const FeatureTrackerOptions& tracker_options,
    const DetectorOptions& detector_options,
    const std::shared_ptr<CameraBundle>& camera_array);

void copyBearingVectors(
    const Frame& frame_cur,
    const Frame& frame_ref,
    const AbstractInitialization::FeatureMatches& matches_cur_ref,
    AbstractInitialization::BearingVectors* f_cur,
    AbstractInitialization::BearingVectors* f_ref);

inline double angleError(const Eigen::Vector3d& f1, const Eigen::Vector3d& f2)
{
  return std::acos(f1.dot(f2) / (f1.norm()*f2.norm()));
}

} // namespace initialization_utils

} // namespace svo
