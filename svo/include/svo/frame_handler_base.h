// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <mutex>
#include <queue>
#include <functional>
#include <unordered_map>

#include <vikit/timer.h>
#include <vikit/ringbuffer.h>

#include <vikit/params_helper.h>
#include <vikit/cameras/ncamera.h>
#include <rpg_common/callback_host.h>

#include "svo/common/frame.h"
#include "svo/map.h"
#include "svo/global.h"

// forward declarations:
namespace vk
{
class PerformanceMonitor;
}

namespace svo
{
/// Keyframe Selection Criterion
enum class KeyframeCriterion {
  DOWNLOOKING, FORWARD
};

/// Options for base frame handler module. Sets tracing and quality options.
struct BaseOptions
{
  /// The VO only keeps a list of N keyframes in the map. If a new keyframe
  /// is selected, then one furthest away from the current position is removed
  /// Default is 10. Set to 0 if all keyframes should remain in the map.
  /// More keyframes may reduce drift but since no loop-closures are actively
  /// detected and closed it is not beneficial to accumulate all keyframes.
  size_t max_n_kfs = 10;

  /// Keyframe selection criterion: If we have a downlooking camera (e.g. on
  /// quadrotor), select DOWNLOOKING. Otherwise, select FORWARD.
  KeyframeCriterion kfselect_criterion = KeyframeCriterion::DOWNLOOKING;

  /// (!) Parameter for DOWNLOOKING keyframe criterion. We select a new KF
  /// whenever we move kfselect_min_dist of the average depth away from the
  /// closest keyframe.
  double kfselect_min_dist = 0.12;

  /// Keyframe selection for FORWARD: If we are tracking more than this amount
  /// of features, then we don't take a new keyframe.
  size_t kfselect_numkfs_upper_thresh = 110;

  /// Keyframe selection for FORWARD : If we have less than this amount of
  /// features we always select a new keyframe.
  size_t kfselect_numkfs_lower_thresh = 80;

  /// Keyframe selection for FORWARD : Minimum distance in meters (set initial
  /// scale!) before a new keyframe is selected.
  double kfselect_min_dist_metric = 0.5;

  /// Keyframe selection for FORWARD: Minimum angle in degrees to closest KF
  double kfselect_min_angle = 5.0;

  int kfselect_min_num_frames_between_kfs = 2;
  double kfselect_min_disparity = -1;

  // maximum duration allowed between keyframes with backend
  double kfselect_backend_max_time_sec = 3.0;

  /// By default, when the VO initializes, it sets the average depth to this
  /// value. This is because from monocular views, we can't estimate the scale.
  /// If you initialize SVO with a downward-looking camera at 1.5m height over
  /// a flat surface, then set this value to 1.5 and the resulting map will
  /// be approximately in the right scale.
  double init_map_scale = 1.0;

  /// By default, the orientation of the first camera-frame is set to be the
  /// identity (camera facing in z-direction, camera-right is in x-direction,
  /// camera-down is in y-direction) and the map scale is initialized with the
  /// option init_map_scale. However, if you would like to provide another
  /// initial orientation and inital scene depth, then activate this flag.
  /// If activated, you need to provide attitude and depth measurements using
  /// the functions addAttitudeMeasurement(), addDepthMeasurement() in the
  /// base class.
  /// This option is useful for in-flight initialization of SVO.
  bool init_use_att_and_depth = false;

  /// (!) During sparse image alignment (see [1]), we find the pose relative
  /// to the last frame by minimizing the photometric error between the frames.
  /// This KLT-style optimizer is pyramidal to allow more motion between two
  /// frames.
  /// Depending on the image size, you should increase this value.
  /// For images of the size 640x480 we set this value to 4. If your image is
  /// double the resolution, increase to 5, and so on.
  size_t img_align_max_level = 4;

  /// (!) During sparse image alignment, we stop optimizing at this level for
  /// efficiency reasons. If you have the time, you can go down to the zero'th
  /// level.
  /// Depending on the image size, you should increase this value.
  /// For images of the size 640x480 we set this value to 2. If your image is
  /// double the resolution, increase to 3, and so on.
  size_t img_align_min_level = 2;

  /// control whether to use robustification in image alignment
  bool img_align_robustification = false;

  /// If you are using a gyroscope and provide an attitude estimate
  /// together with the image in the function addImage() then this parameter
  /// specifies how much you trust your gyroscope. If you set it to 2.0
  /// (default) it means that the gyroscope attitude is valued two times more
  /// than the actualy orientation estimate from the visual measurements.
  double img_align_prior_lambda_rot = 0.0;

  /// Internally, we have a very basic constant velocity motion model.
  /// similarly to lambda_rot, this parameter trades-off the visual measurements
  /// with the constant velocity prior. By default this weight is way below 1.
  double img_align_prior_lambda_trans = 0.0;

  /// If you choose to extract many features then sparse image alignment may
  /// become too slow. You can limit the number of features for this step with
  /// this parameter to randomly sample N features that are used for alignment.
  size_t img_align_max_num_features = 0;

  /// Whether or not to include the distortion when calculating the jacobian.
  /// For small FoV pinhole projection, it is safe to leave it as false.
  /// For fisheye lens, set this to true.
  bool img_align_use_distortion_jacobian = false;
  
  /// Estimate an affine transformation for illumination/exposure change.
  /// If you observe bad tracking because of illumination/exposure change,
  /// enabling these parameters might help.
  /// Normally it is OK to leave them as default.
  bool img_align_est_illumination_gain = false;
  bool img_align_est_illumination_offset = false;

  /// (!) This parameter is the reprojection error threshold during pose
  /// optimization. If the distance between a feature and the projected pixel
  /// position of the corresponding 3D point is further than this threshold
  /// appart (on the zero'th level pyramid), then the feature is removed from
  /// the frame. With a good camera and image resolution of 640x480 a threshold
  /// of 2.0 is typically ok. If you use shitty cameras (rolling shutter),
  /// higher resolution cameras, cameras with imperfect calibration etc. you
  /// might increase this threshold. But first, check the tracefile for the
  /// average reprojection threshold. We made the experice that with GoPro
  /// cameras, we had to increase this threshold.
  double poseoptim_thresh = 2.0;

  /// This is the same parameter as img_align_prior_lambda_rot but for
  /// the pose optimization instead of the sparse image alignment. Only used
  /// if you provide gyroscope measurements.
  double poseoptim_prior_lambda = 0.0;

  /// This parameter controls whether the pose optimizer works on:
  /// - unit plane: preferable for pinhole model
  /// - unit sphere: omnidirectional camera model(e.g. fisheye, catadioptric)
  bool poseoptim_using_unit_sphere = false;

  /// By default SVO does not do bundle adjustment but it optimizes the pose
  /// and the structure separately. This is not optimal but much faster. This
  /// parameters specifies how many 3D points should be randomly selected at
  /// every frame and be optimized. For speed reasons, we don't optimize all
  /// points at every iteration. Set to -1 if you want to do so anyway.
  int structure_optimization_max_pts = 20;

  /// Location where the tracefile is saved.
  std::string trace_dir = "/tmp";

  /// Minimum number of features that should be tracked. If the number falls
  /// bellow then the stage is set to STAGE_RELOCALIZING and the tracking
  /// quality to TRACKING_INSUFFICIENT until we find more features again.
  size_t quality_min_fts = 50;

  /// If from one frame to the other we suddenly track much less features,
  /// this can be an indication that something is wrong and we set the stage
  /// to STAGE_RELOCALIZING and the tracking quality to TRACKING_INSUFFICIENT.
  int quality_max_fts_drop = 40;

  /// Once we are in relocalization mode, we allow a fixed number of images
  /// to try and relocalize before we reset() and set to STAGE_PAUSED.
  size_t relocalization_max_trials = 100;

  /// EXPERIMENTAL Should IMU measurements be used.
  bool use_imu = false;

  /// EXPERIMENTAL Update seeds with old keyframes.
  bool update_seeds_with_old_keyframes = false;

  /// EXPERIMENTAL Asynchronous reprojection (for multi-camera svo)
  bool use_async_reprojectors = false;

  /// Trace statistics for benchmarking
  bool trace_statistics = false;

  /// we check whether the backend scale has stablized
  double backend_scale_stable_thresh = 0.02;

  /// EXPERIMENTAL:
  /// If the time from the last good tracking frame till the first frame that
  /// the system is initialized, we still think the relative pose with respect
  /// to the global map / loop closure database is still good.
  double global_map_lc_timeout_sec_ = 3.0;
};

enum class Stage {
  kPaused,           ///< Stage at the beginning and after reset
  kInitializing,     ///< Stage until the first frame with enough features is found
  kTracking,         ///< Stage when SVO is running and everything is well
  kRelocalization    ///< Stage when SVO looses tracking and it tries to relocalize
};
extern const std::unordered_map<svo::Stage, std::string, EnumClassHash>
kStageName;

enum class TrackingQuality {
  kInsufficient,
  kBad,
  kGood
};
extern const std::unordered_map<svo::TrackingQuality, std::string,
EnumClassHash>  kTrackingQualityName;

enum class UpdateResult {
  kDefault,
  kKeyframe,
  kFailure
};
extern const std::unordered_map<svo::UpdateResult, std::string, EnumClassHash>
kUpdateResultName;


/// Base class for various VO pipelines. Manages the map and the state machine.
class FrameHandlerBase : public rpg_common::CallbackHost<const FrameBundlePtr&>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::mutex mutex_t;
  typedef std::unique_lock<mutex_t> ulock_t;
  typedef std::function<bool(const Transformation& pose)> NewKeyFrameCriteria;

  /// Default constructor
  FrameHandlerBase(
      const BaseOptions& base_options,
      const ReprojectorOptions& reprojector_options,
      const DepthFilterOptions& depthfilter_options,
      const DetectorOptions& detector_options,
      const InitializationOptions& init_options,
      const FeatureTrackerOptions& tracker_options,
      const CameraBundle::Ptr& cameras);

  virtual ~FrameHandlerBase();

  // no copy
  FrameHandlerBase(const FrameHandlerBase&) = delete;
  FrameHandlerBase& operator=(const FrameHandlerBase&) = delete;

  /// @name Main Interface
  ///
  /// These are the main functions to be used in this class when interfacing with
  /// the odometry pipeline.
  ///
  /// @{

  /// Start processing. Call this function at the beginning or to restart once
  /// the stage is set to paused.
  void start() { set_start_ = true; }

  /// Will reset the map as soon as the current frame is finished processing.
  void reset() { set_reset_ = true; }

  /// Get the current stage of the algorithm.
  inline Stage stage() const { return stage_; }
  inline std::string stageStr() const { return kStageName.at(stage_); }

  bool addFrameBundle(const FrameBundlePtr& frame_bundle);

  bool addImageBundle(
        const std::vector<cv::Mat>& imgs,
        const uint64_t timestamp);

  void setRotationPrior(const Quaternion& R_imu_world);
  void setRotationIncrementPrior(const Quaternion& R_lastimu_newimu);

  inline bool isBackendValid() const
  {
    const bool ptr_valid = bundle_adjustment_? true:false;
    return ptr_valid && bundle_adjustment_type_ != BundleAdjustmentType::kNone;
  }

  inline bool isBackendScaleInitialised() const
  {
    return backend_scale_initialized_;
  }

  inline bool hasGlobalMap() const
  {
#ifdef SVO_GLOBAL_MAP
    return global_map_? true : false;
#else
    return false;
#endif

  }

  inline bool doesGlobalMapHaveInitialBA() const
  {
    return global_map_has_initial_ba_;
  }

  FrameBundlePtr getLastFrames() const { return last_frames_; }

  /// @}

  /// Has the pipeline started? (set_start_ is set to false when started).
  inline bool hasStarted() const { return !set_start_; }

  /// Get the current map.
  inline const MapPtr& map() const { return map_; }

  /// Get camera bundle.
  inline const CameraBundle::Ptr& getNCamera() const { return cams_; }

  /// Get tracking quality.
  inline TrackingQuality trackingQuality() const { return tracking_quality_; }
  inline std::string trackingQualityStr() const
  { return kTrackingQualityName.at(tracking_quality_); }

  /// Get update result
  inline UpdateResult updateResult() const { return update_res_; }
  inline std::string UpdateResultStr() const
  { return kUpdateResultName.at(update_res_); }

  /// Get the processing time of the previous iteration.
  inline double lastProcessingTime() const { return timer_.getTime(); }

  /// Get the number of feature observations of the last frame.
  inline size_t lastNumObservations() const { return num_obs_last_; }

  /// EXPERIMENTAL Set Bundle-Adjuster handler
  void setBundleAdjuster(const AbstractBundleAdjustmentPtr& ba);

  inline const AbstractBundleAdjustmentPtr& getBundleAdjuster() const
  {
    return bundle_adjustment_;
  }

  /// Set pose of first frame by specifying the IMU pose
  inline void setInitialImuPose(const Transformation& T_world_imu)
  {
    T_world_imuinit = T_world_imu;
  }

  /// Set pose of first frame by specifying the camera pose
  inline void setInitialCamPose(const Transformation& T_world_cam, size_t cam_index = 0)
  {
    T_world_imuinit = T_world_cam*cams_->get_T_C_B(cam_index);
  }

  /// Get the set of spatially closest keyframes of the last frame.
  std::vector<FramePtr> closeKeyframes() const;

  /// Set compensation parameters.
  void setCompensation(const bool do_compensation);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  virtual void setFirstFrames(const std::vector<FramePtr>& first_frames);

  /// @name Debug Interface
  /// These parameters should be private but are currently not for easier debugging.
  /// It is unlikely that you need them.
  ///
  /// @{

  /// Options for BaseFrameHandler module
  BaseOptions options_;

  /// Camera model, can be ATAN, Pinhole or Ocam (see vikit)
  CameraBundle::Ptr cams_;

  /// Current frame-bundle that is being processed
  FrameBundlePtr new_frames_;

  /// Last frame-bundle that was processed. Can be nullptr.
  FrameBundlePtr last_frames_;

  /// Custom callback to check if new keyframe is required
  NewKeyFrameCriteria need_new_kf_;

  /// Default keyframe selection criterion.
  virtual bool needNewKf(const Transformation& T_f_w);

  /// Translation prior computed from simple constant velocity assumption
  Vector3d t_lastimu_newimu_;

  /// Initial orientation
  Transformation T_world_imuinit;

  // SVO Modules
  SparseImgAlignBasePtr sparse_img_align_;
  std::vector<ReprojectorPtr> reprojectors_;
  PoseOptimizerPtr pose_optimizer_;
  DepthFilterPtr depth_filter_;
  InitializerPtr initializer_;
  ImuHandlerPtr imu_handler_;
#ifdef SVO_LOOP_CLOSING
  LoopClosingPtr lc_;
  size_t loop_closing_counter_ = 0;
#endif
#ifdef SVO_GLOBAL_MAP
  GlobalMapPtr global_map_;
#endif
  /// @}

protected:
  Stage stage_;                 //!< Current stage of the algorithm.
  bool set_reset_;              //!< Flag that the user can set. Will reset the system before the next iteration.
  bool set_start_;              //!< Flag the user can set to start the system when the next image is received.
  MapPtr map_;                  //!< Map of keyframes created by the slam system.
  vk::Timer timer_;             //!< Stopwatch to measure time to process frame.
  vk::RingBuffer<double> acc_frame_timings_;    //!< Total processing time of the last 10 frames, used to give some user feedback on the performance.
  vk::RingBuffer<size_t> acc_num_obs_;          //!< Number of observed features of the last 10 frames, used to give some user feedback on the tracking performance.
  size_t num_obs_last_;                         //!< Number of observations in the previous frame.
  TrackingQuality tracking_quality_;            //!< An estimate of the tracking quality based on the number of tracked features.
  UpdateResult update_res_;                     //!< Update result of last frame bundle
  size_t frame_counter_ = 0; //!< Number of frames processed since started
  double depth_median_;                 //!< Median depth at last frame
  double depth_min_;                    //!< Min depth at last frame
  double depth_max_;

  std::vector<std::vector<FramePtr>> overlap_kfs_;

  // Rotation prior.
  bool have_rotation_prior_ = false;
  Quaternion R_imu_world_;
  Quaternion R_imulast_world_;

  // 6 DoF Motion prior
  bool have_motion_prior_ = false;
  Transformation T_newimu_lastimu_prior_;

  // backend related
  AbstractBundleAdjustmentPtr bundle_adjustment_;
  BundleAdjustmentType bundle_adjustment_type_ = BundleAdjustmentType::kNone;
  Eigen::Matrix<double, 9, 1> speed_bias_backend_latest_;
  Transformation T_WS_backend_latest_;
  double timestamp_backend_latest_;
  bool backend_reinit_ = false;

  // relocalization
  FramePtr reloc_keyframe_;
  size_t relocalization_n_trials_;      //!< With how many frames did we try to relocalize?

  void setInitialPose(const FrameBundlePtr& frame_bundle) const;

  size_t sparseImageAlignment();

  size_t projectMapInFrame();

  size_t optimizePose();

  void optimizeStructure(
      const FrameBundlePtr& frames,
      int max_n_pts,
      int max_iter);

  void upgradeSeedsToFeatures(const FramePtr& frame);

  /// Reset the map and frame handler to start from scratch.
  void resetVisionFrontendCommon();

  /// Change the states of relevant modules to indicate recovery mode
  void setRecovery(const bool recovery);

  inline bool isInRecovery() const
  {
    return loss_without_correction_;
  }

  /// Pipeline implementation in derived class.
  virtual UpdateResult processFrameBundle() = 0;

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll() { resetVisionFrontendCommon(); }

  /// Reset backend
  virtual void resetBackend();

  /// Set the tracking quality based on the number of tracked features.
  virtual void setTrackingQuality(const size_t num_observations);

  /// Set all cells in detector to occupied in which a point reprojected during
  /// the reprojection stage.
  virtual void setDetectorOccupiedCells(
      const size_t reprojector_grid_idx,
      const DetectorPtr& feature_detector);

  /// Get motion prior, between last and new frames expressed in IMU frame.
  virtual void getMotionPrior(const bool use_velocity_in_frame);


  /// Helpers for ceres backend
  bool backend_scale_initialized_ = false;
  double last_kf_time_sec_ = -1.0;

  // global map related
  bool global_map_has_initial_ba_ = false;

  // status for loss
  bool loss_without_correction_ = false;
  double last_good_tracking_time_sec_ = -1.0;
};

} // namespace nslam
