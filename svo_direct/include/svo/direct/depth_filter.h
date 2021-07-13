// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <queue>
#include <memory> // std::shared_ptr
#include <mutex>
#include <thread>
#include <condition_variable>
#include <vikit/performance_monitor.h>
#include <svo/direct/matcher.h>

namespace svo {

// forward declarations
class AbstractDetector;
typedef std::shared_ptr<AbstractDetector> DetectorPtr;
struct DetectorOptions;

/// Depth-filter config parameters
struct DepthFilterOptions
{
  /// Threshold for the uncertainty of the seed. If seed's sigma2 is thresh
  /// smaller than the inital sigma, it is considered as converged.
  /// Default value is 200. If seeds should converge quicker, set it to 50 or
  /// if you want very precise 3d points, set it higher.
  double seed_convergence_sigma2_thresh = 200.0;

  /// Threshold for map point seeds convergence. Should be higher to make sure
  /// we have an accurate map (for loop closing).
  double mappoint_convergence_sigma2_thresh = 500.0;

  /// Use inverse-depth parametrization for seeds.
  /// Default is true. Set to false if you are using the depth-filter to
  /// reconstruct small objects with fixed depth range.
  bool use_inverse_depth = true;

  /// Specify the max pyramid level for the matcher.
  /// Normally, you don't need to change this parameters.
  size_t max_search_level = 2;

  /// Show additional debug output
  bool verbose = false;

  /// Start separate thread for seed updates
  bool use_threaded_depthfilter = true;

  /// Update the 3D point linked by the feature in the seed (false for REMODE-CPU)
  bool update_3d_point = true;

  /// Do epipolar search on unit sphere
  bool scan_epi_unit_sphere = false;

  /// Restrict number of features per frame.
  size_t max_n_seeds_per_frame = 200;

  size_t max_map_seeds_per_frame = 200;

  /// use affine model to compensate for brightness change
  bool affine_est_offset = true;
  bool affine_est_gain = false;

  ///
  bool extra_map_points = false;
};

/// Depth filter implements the Bayesian Update proposed in:
/// "Video-based, Real-Time Multi View Stereo" by G. Vogiatzis and C. Hern??ndez.
/// In Image and Vision Computing, 29(7):434-441, 2011.
class DepthFilter
{
protected:

  /// Job for multi-threading. can either be to update a seed with the frame
  /// or to initialize new seeds in the frame
  struct Job
  {
    enum Type { UPDATE, SEED_INIT } type;
    FramePtr cur_frame;
    FramePtr ref_frame;
    size_t ref_frame_seed_index;
    double min_depth, max_depth, mean_depth;

    /// Default constructor
    Job()
      : cur_frame(nullptr), ref_frame(nullptr)
    {}

    /// Constructor for seed update
    Job(const FramePtr& _cur_frame, const FramePtr& _ref_frame, const size_t _ref_index)
      : type(UPDATE)
      , cur_frame(_cur_frame)
      , ref_frame(_ref_frame)
      , ref_frame_seed_index(_ref_index)
    {}

    /// Constructor for seed initialization
    Job(const FramePtr& f, double min_d, double max_d, double mean_d)
      : type(SEED_INIT), cur_frame(f), ref_frame(nullptr)
      , min_depth(min_d), max_depth(max_d), mean_depth(mean_d)
    {}
  };

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<DepthFilter> Ptr;
  typedef std::mutex mutex_t;
  typedef std::unique_lock<mutex_t> ulock_t;
  typedef std::queue<Job> JobQueue;

  DepthFilterOptions options_;

  /// Default Constructor
  DepthFilter(
      const DepthFilterOptions& options,
      const DetectorOptions& detector,
      const std::shared_ptr<CameraBundle>& cams);

  /// Constructor for REMODE-CPU
  DepthFilter(
      const DepthFilterOptions& options);

  /// Destructor stops thread if necessary.
  virtual ~DepthFilter();

  /// Start this thread when seed updating should be in a parallel thread.
  void startThread();

  /// Stop the parallel thread that is running.
  void stopThread();

  /// Add new keyframe to the queue, depth_max is only required for direct-depth
  void addKeyframe(
      const FramePtr& frame,
      const double depth_mean,
      const double depth_min,
      const double depth_max);

  /// Resets all jobs of the parallel thread
  void reset();

  /// test
  Matcher& getMatcher() { return *matcher_; }

  /// Update seeds
  /// \param frames_with_seeds List of frames which contain seeds that should be
  /// updated with new_frame
  /// \param new_frame The new frame that is used as observation to update all
  /// seeds in frames_with_seeds
  size_t updateSeeds(
      const std::vector<FramePtr>& frames_with_seeds,
      const FramePtr& new_frame);

  // need public access to set grid occupancy
  mutex_t feature_detector_mut_;
  DetectorPtr feature_detector_;
  DetectorPtr sec_feature_detector_; // for extra points used for loop closing

protected:
  mutex_t jobs_mut_;
  JobQueue jobs_;
  std::condition_variable jobs_condvar_;
  std::unique_ptr<std::thread> thread_;
  bool quit_thread_ = false;
  Matcher::Ptr matcher_;

  /// A thread that is continuously updating the seeds.
  void updateSeedsLoop();
};

namespace depth_filter_utils {

/// Initialize new seeds from a frame.
void initializeSeeds(
    const FramePtr& frame,
    const DetectorPtr& feature_detector,
    const size_t max_n_seeds,
    const float min_depth,
    const float max_depth,
    const float mean_depth);

/// Update Seed
bool updateSeed(
    const Frame& cur_frame,
    Frame& ref_frame,
    const size_t& seed_index,
    Matcher& matcher,
    const FloatType sigma2_convergence_threshold,
    const bool check_visibility = true,
    const bool check_convergence = false,
    const bool use_vogiatzis_update = true);

bool updateFilterVogiatzis(
    const FloatType z,
    const FloatType tau2,
    const FloatType z_range,
    Eigen::Ref<SeedState>& seed);

bool updateFilterGaussian(
    const FloatType z,
    const FloatType tau2,
    Eigen::Ref<SeedState>& seed);

/// Compute the uncertainty of the measurement.
double computeTau(
    const Transformation& T_ref_cur,
    const BearingVector& f,
    const FloatType z,
    const FloatType px_error_angle);

double computeEpiGradAngle(
    const Transformation& T_cur_ref,
    const BearingVector& f_ref,
    const GradientVector& grad,
    const FloatType depth_estimate);

#ifdef SVO_USE_PHOTOMETRIC_DISPARITY_ERROR
bool setSeedCovariance(
    const int halfpatch_size,
    const double image_noise2,
    SeedImplementation::Ptr seed);

double getSeedDisparityUncertainty(
    const SeedImplementation::Ptr& seed,
    const Transformation& T_cur_ref);
#endif

} // namespace depth_filter_utils

} // namespace svo
