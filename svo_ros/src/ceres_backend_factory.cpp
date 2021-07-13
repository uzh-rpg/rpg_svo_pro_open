#include <svo_ros/ceres_backend_factory.h>

#include <memory>

#include <ros/package.h>
#include <vikit/params_helper.h>

#include <svo/imu_handler.h>
#include <svo/motion_detector.hpp>

namespace svo {
namespace ceres_backend_factory {

MotionDetectorOptions loadMotionDetectorOptions(const ros::NodeHandle& pnh)
{
  MotionDetectorOptions o;
  o.px_diff_threshold =
      vk::param<double>(pnh, "zero_motion_px_diff_threshold", 0.5);
  o.ratio_moving_pixels_threshold =
      vk::param<double>(pnh, "zero_motion_ratio_moving_pixels_threshold", 0.1);
  o.min_number_correspondences =
      vk::param<int>(pnh, "zero_motion_min_number_correspondences", 5);
  o.max_features_to_check =
      vk::param<int>(pnh, "zero_motion_max_features_to_check", 100);
  DEBUG_CHECK(o.max_features_to_check>0) << "max_features_to_check must be > 0";
  o.sigma = vk::param<double>(pnh, "zero_motion_sigma", 0.05);
  return o;
}

CeresBackendInterface::Ptr makeBackend(const ros::NodeHandle& pnh,
                                       const CameraBundlePtr& camera_bundle)
{
  VLOG(1) << "Initialize Backend.";
  CeresBackendOptions backend_options;
  /// Output ceres optimization progress
  backend_options.verbose =
      vk::param<bool>(pnh, "ceres_verbose", false);
  /// Apply marginalization?
  backend_options.marginalize =
      vk::param<bool>(pnh, "ceres_marginalize", true);
  /// Maximum number of iterations
  backend_options.num_iterations =
      static_cast<size_t>(vk::param<int>(pnh, "ceres_num_iterations", 3));
  backend_options.num_threads =
      static_cast<size_t>(vk::param<int>(pnh, "ceres_num_threads", 2));
  /// Sliding window size of ceres backend
  backend_options.num_imu_frames =
      static_cast<size_t>(vk::param<int>(pnh, "ceres_num_imu_frames", 3));
  /// Number of keyframes of ceres backend
  backend_options.num_keyframes =
      static_cast<size_t>(vk::param<int>(pnh, "max_n_kfs", 5));
  /// Maximum time used to optimize [s]. Set negative to always do the
  /// maximum number of iterations.
  backend_options.max_iteration_time =
      static_cast<double>(
        vk::param<double>(pnh, "ceres_max_iteration_time", -1.0));
  /// Should marginalization term be discarded after loop? Experimental
  backend_options.remove_marginalization_term_after_correction_ = vk::param<bool>(
        pnh, "ceres_remove_marginalization_term_after_loop", false);
  backend_options.recalculate_imu_terms_after_loop = vk::param<bool>(
        pnh, "ceres_recalculate_imu_terms_after_loop", false);
  backend_options.remove_fixation_min_num_fixed_landmarks_ = static_cast<size_t>
      (vk::param<int>(pnh, "ceres_remove_fixation_min_num_fixed_points", 10u));
  backend_options.max_fixed_lm_in_ceres_  = static_cast<size_t>
      (vk::param<int>(pnh, "ceres_max_fixed_landmarks", 50u));

  CeresBackendInterfaceOptions ba_interface_options;
  ba_interface_options.min_num_obs =
      static_cast<size_t>(vk::param<int>(pnh, "backend_min_num_obs", 2));
  ba_interface_options.min_parallax_thresh =
      vk::param<double>(pnh, "backend_min_parallax_deg", 2.0)/180.0*M_PI;
  ba_interface_options.only_use_corners =
      vk::param<bool>(pnh, "backend_only_use_corners", false);
  ba_interface_options.use_zero_motion_detection =
      vk::param<bool>(pnh, "backend_use_zero_motion_detection", true);
  ba_interface_options.backend_zero_motion_check_n_frames = static_cast<int>(
        vk::param<int>(pnh, "backend_zero_motion_check_n_frames", 5));
  ba_interface_options.use_outlier_rejection =
      vk::param<bool>(pnh, "use_outlier_rejection", true);
  ba_interface_options.outlier_rejection_px_threshold =
      vk::param<double>(pnh, "outlier_rejection_px_threshold", 2.0);
  // EXPERIMENTAL: when tracking is bad
  ba_interface_options.skip_optimization_when_tracking_bad =
      vk::param<bool>(pnh, "skip_optimization_when_tracking_bad", false);
  ba_interface_options.min_added_measurements=
      vk::param<int>(pnh, "skip_optim_min_obs", 20);

  // extrinsics refinement
  ba_interface_options.refine_extrinsics =
        vk::param<bool>(pnh, "backend_refine_extrinsics", false);
  ba_interface_options.extrinsics_pos_sigma_meter =
        vk::param<double>(pnh, "backend_extrinsics_pos_sigma_meter", 0.05);
  ba_interface_options.extrinsics_pos_sigma_meter =
        vk::param<double>(pnh, "backend_extrinsics_rot_sigma_deg", 5.0) / 180.0 * M_PI;

  CeresBackendInterface::Ptr ba_interface =
      std::make_shared<CeresBackendInterface>(ba_interface_options,
                                              backend_options,
                                              loadMotionDetectorOptions(pnh),
                                              camera_bundle);

  if(vk::param<bool>(pnh, "ba_parallelized", true))
    ba_interface->startThread();

  return ba_interface;
}

} // namespace vin_factory
} // namespace svo

