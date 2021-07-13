#pragma once

#include <thread>
#include <condition_variable>
#include <math.h>

#include <svo/abstract_bundle_adjustment.h>
#include <svo/vio_common/backend_types.hpp>

#include "svo/ceres_backend/estimator.hpp"
#include "svo/ceres_backend_publisher.hpp"

namespace svo
{
// fwd
struct MotionDetectorOptions;
class MotionDetector;
class OutlierRejection;
class ImuHandler;

struct CeresBackendOptions
{
  // Optimization settings.
  /// (!) Maximum optimization time: this is only a soft limit, not guaranteed
  double max_iteration_time = -1;
  /// (!) How many iterations should be performed
  size_t num_iterations = 3u;
  /// (!) How many threads should be used for optimization
  size_t num_threads = 2u;
  /// (!) Verbose optimization output
  bool verbose = false;

  // Marginalization settings.
  /// (!) Marginalize?
  bool marginalize = true;
  /// (!) Number of keyframes in the map. Must be identical to number of
  /// keyframes in map of SVO frontend. Is set from same ROS parameter.
  size_t num_keyframes = 5u;
  /// (!) Number of frames in the temporal window of the optimization
  size_t num_imu_frames = 3u;

  // Loop closing settings.
  /// (!) Should the marginalization term be removed after loop closure?
  /// EXPERIMENTAL: so far it seems that we need to do this, which is equivalent
  /// to discard all past information when we have a loop closure. If we have
  /// good quality loop closure infos this is ok. If the offset between the
  /// updated pose using the loop and before is small, this also works if we
  /// keep this term.
  bool remove_marginalization_term_after_correction_ = false;

  // since we are using a 6 DoF corrrection, the IMU error terms are not
  // consistent with the states anymore after the correction.
  bool recalculate_imu_terms_after_loop = false;

  // minimum of fixed landmarks that is needed to remove pose fixation
  size_t remove_fixation_min_num_fixed_landmarks_ = 10u;
  size_t max_fixed_lm_in_ceres_ = 50u;
};

struct CeresBackendInterfaceOptions
{
  /// (!) Only add landmarks to backend with minimum number of observations
  size_t min_num_obs = 2u;

  /// (!) CURRENTLY INACTIVE: only introduce observation with a minimum
  /// parallax angle
  double min_parallax_thresh = 2.0 / 180 * M_PI;

  /// (!) Only add corner features to backend?
  bool only_use_corners = false;

  /// (!) use the zero motion detection?
  bool use_zero_motion_detection = true;

  /// (!) After how many non moving frames should we introduce zero motion prior
  size_t backend_zero_motion_check_n_frames = 5;

  /// (!) use the outlier rejection module
  bool use_outlier_rejection = true;

  /// (!) This parameter is the reprojection error threshold after
  /// optimization. If the distance between a feature and the projected pixel
  /// position of the corresponding 3D point is further than this threshold
  /// appart (on the zero'th level pyramid), then the feature is removed from
  /// the frame. With a good camera and image resolution of 640x480 a threshold
  /// of 2.0 is typically ok. If you use shitty cameras (rolling shutter),
  /// higher resolution cameras, cameras with imperfect calibration etc. you
  /// might increase this threshold. We made the experice that with GoPro
  /// cameras, we had to increase this threshold.
  double outlier_rejection_px_threshold = 2.0;

  // EXPERIMENTAL: special treatment when tracking is bad
  /// (!) when the newly added measurements are below this number,
  /// tracking is considered bad
  size_t min_added_measurements = 10u;

  /// (!) skip optimization when tracking is bad.
  /// Pure IMU integration is used to propagate the new state.
  bool skip_optimization_when_tracking_bad = false;

  // refine extrinsics
  bool refine_extrinsics = false;
  double extrinsics_pos_sigma_meter = 0.05;
  double extrinsics_rot_sigma_rad = 5.0 / 180.0 * M_PI;
};

typedef std::pair<size_t, size_t> CorrespondencePair;
typedef std::vector<CorrespondencePair> CorrespondIds;

class CeresBackendInterface : public AbstractBundleAdjustment
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<CeresBackendInterface> Ptr;

  CeresBackendInterfaceOptions options_;
  CeresBackendOptions optimizer_options_;


  CeresBackendInterface(const CeresBackendInterfaceOptions& options,
                        const CeresBackendOptions& optimizer_options,
                        const MotionDetectorOptions& motion_detector_options,
                        const CameraBundlePtr& camera_bundle);

  ~CeresBackendInterface();

  /**
   * @brief This is called BEFORE frame is processed in frontend
   * @param new_frames Pose prior is written into transformation
   * @param last_frames Optimized pose is written into transformation
   * @param map All keyframe poses and landmark positions are updated
   * @param have_motion_prior flags if we succesfully obtained pose prior
   *        for new frames
   */
  void loadMapFromBundleAdjustment(const FrameBundlePtr& new_frames,
                                   const FrameBundlePtr& last_frames,
                                   const MapPtr& map,
                                   bool& have_motion_prior) override;

  /**
   * @brief This is called AFTER frame_bundle was processed by the frontend.
   *        All landmark observations of the frame bundle are added to backend.
   *        If this is a keyframe new landmarks are introduced, and the frame
   *        bundle is set to a keyframe in the backend.
   * @param[in] frame_bundle_vec Frame bundles that were already processed by
   * the
   *            frontend.
   */
  void bundleAdjustment(const FrameBundlePtr& frame_bundle) override;

  /**
   * @brief Reset the backend: NOT IMPLEMENTED
   */
  void reset() override;

  /**
   * @brief Start the ceres optimization thread.
   */
  void startThread() override;

  /**
   * @brief Stop and join ceres optimization thread.
   */
  void quitThread() override;

  // set correction transformation to be applied
  void setCorrectionInWorld(const Transformation& w_T_correction) override;

  void getAllActiveKeyframes(std::vector<FramePtr>* keyframes) override;

  /**
   * @brief Make a publisher node for the backend
   * @param[in]   nh_private Node handle to be used to publish
   * @param[out]  publisher Publisher handle, shared with frontend for
   *              benchmarking purposes.
   */
  void makePublisher(const ros::NodeHandle& nh_private,
                     std::shared_ptr<CeresBackendPublisher>& publisher)
  {
    publisher_.reset(new CeresBackendPublisher(nh_private, backend_.getMap()));
    publisher = publisher_;
  }

  /**
   * @brief Set the IMU in backend
   * @param imu_handler Shared ImuHandler with frontend
   */
  void setImu(const std::shared_ptr<ImuHandler> imu_handler);

  /**
   * @brief Initialize a performance monitor for benchmarking
   * @param trace_dir[in] path to folder where data should be dumped
   */
  void setPerformanceMonitor(const std::string& trace_dir) override;

  /**
   * @brief Start a timer to measure time from frame arrival to the end of
   *        first optimization (called in frontend for benchmarking)
   * @param[in] bundle_id of the frame, of which time is measured.
   */
  void startTimer(const BundleId bundle_id) override;

  /**
   * @brief Get the pose and speed bias of the IMU as per the latest IMU frame
   * @param[in] None
   * @param[out] The speed bias and pose of the latest imu frame
   */
  void getLatestSpeedBiasPose(Eigen::Matrix<double, 9, 1>* speed_bias,
                              Transformation* T_WS,
                              double* timestamp) const override;

  void setReinitStartValues(const Eigen::Matrix<double, 9, 1>& sb,
                            const Transformation& Tws,
                            const double timestamp) override;

  /**
   * @brief getNumFrames returns the number of frames in backend
   * @return
   */
  inline int getNumFrames() const override
  {
    return static_cast<int>(backend_.numFrames());
  }

  inline bool isFixedToGlobalMap() const override
  {
    return lock_to_fixed_landmarks_;
  }

  inline BundleId lastOptimizedBundleId() const override
  {
    return last_optimized_nframe_.load();
  }

  inline void getLastState(ViNodeState* state) const override
  {
    *state = last_state_;
  }

  std::string getStationaryStatusStr() const;

protected:
  // modules
  Estimator backend_;
  std::shared_ptr<ImuHandler> imu_handler_;
  std::unique_ptr<MotionDetector> motion_detector_;
  size_t no_motion_counter_;
  std::unique_ptr<OutlierRejection> outlier_rejection_;

  /**
   * @brief Adds landmarks and observation of frame to the backend. Adding of
   *        landmarks is subject to certain criteria.
   * @param[in] frame
   */
  void addLandmarksAndObservationsToBackend(const FramePtr& frame);

  /**
   * @brief Introduce state for frame_bundle in backend, and add the
   *        corresponding IMU measurements.
   * @param frame_bundle to be added
   * @return true if successful
   */
  bool
  addStatesAndInertialMeasurementsToBackend(const FrameBundlePtr& frame_bundle);

  /**
   * @brief Loop of optimization thread
   */
  void optimizationLoop();

  /**
   * @brief Update the active keyframes, i.e. the keyframes that are contained
   *        in the backend.
   */
  void updateActiveKeyframes();

  /**
   * @brief Update the state stored in the frame (bundle) with the backend
   */
  void updateFrameStateWithBackend(const FramePtr& frame,
                                   const bool get_speed_bias);
  void updateBundleStateWithBackend(const FrameBundlePtr& frames,
                                    const bool get_speed_bias);

  // Threading
  mutable std::condition_variable wait_condition_;
  mutable std::mutex mutex_backend_;
  mutable std::mutex loopclosinginfo_;
  std::unique_ptr<std::thread> thread_;
  std::atomic_bool stop_thread_{ false };

  // state
  // bundle id for which the IMU messages are added
  BundleId last_added_nframe_imu_ = -1;
  // bundle id for which the images and observations are added (after frontend)
  BundleId last_added_nframe_images_ = -1;
  // book keeping for the time
  int64_t last_added_frame_stamp_ns_ = 0;
  // the bundle id for which the optimized states have been 
  // updated in the fronend
  BundleId last_updated_nframe_ = -1;

  // the bundle id for which the backend has finished optimization
  std::atomic<BundleId> last_optimized_nframe_ {-1};

  // variables for handling optimization choices
  bool skip_optimization_once_ = false;

  /// keyframes that are contained in backend. Oldest keyframe in front
  std::deque<FramePtr> active_keyframes_;

  // visualization
  ViNodeState last_state_;
  CeresBackendPublisher::Ptr publisher_;

  // loop closing
  std::mutex w_T_correction_mut_;
  Transformation w_T_correction_to_apply_;
  bool is_w_T_valid_ = false;

  // benchmarking
  PerformanceMonitorPtr g_permon_backend_;
  std::unordered_map<BundleId, vk::Timer> timers_;

  // fixation
  std::atomic_bool lock_to_fixed_landmarks_ {false};
  uint64_t global_landmark_value_version_ = 0u;
  bool image_motion_detector_stationary_ = false;
  bool imu_motion_detector_stationary_ = false;
};

}  // namespace svo
