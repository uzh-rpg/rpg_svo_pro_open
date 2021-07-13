#include <ros/package.h>

#include <svo/svo.h>
#include <svo/common/imu_calibration.h>
#include <svo_ros/svo_factory.h>
#include <svo/frame_handler_mono.h>
#include <svo/frame_handler_stereo.h>
#include <vikit/params_helper.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop

#ifdef SVO_USE_VIN_BACKEND
#include <svo_gtsam/backend_optimizer.h>
#include <svo_gtsam/backend_interface.h>
#include <svo_gtsam/graph_manager.h>
#endif

#ifdef SVO_LOOP_CLOSING
#include <svo/online_loopclosing/loop_closing.h>
#include <svo/online_loopclosing/map_alignment.h>
#endif

#ifdef SVO_GLOBAL_MAP
#include <svo/global_map.h>
#include <svo/gtsam/graph_manager.h>
#include <svo/gtsam/gtsam_optimizer.h>
#endif

namespace svo {
namespace factory {

BaseOptions loadBaseOptions(const ros::NodeHandle& pnh, bool forward_default)
{
  BaseOptions o;
  o.max_n_kfs = vk::param<int>(pnh, "max_n_kfs", 5);
  o.use_imu = vk::param<bool>(pnh, "use_imu", false);
  o.trace_dir = vk::param<std::string>(pnh, "trace_dir", ros::package::getPath("svo")+"/trace");
  o.quality_min_fts = vk::param<int>(pnh, "quality_min_fts", 50);
  o.quality_max_fts_drop = vk::param<int>(pnh, "quality_max_drop_fts", 40);
  o.relocalization_max_trials = vk::param<int>(pnh, "relocalization_max_trials", 50);
  o.poseoptim_prior_lambda = vk::param<double>(pnh, "poseoptim_prior_lambda", 0.0);
  o.poseoptim_using_unit_sphere = vk::param<bool>(pnh, "poseoptim_using_unit_sphere", false);
  o.img_align_prior_lambda_rot = vk::param<double>(pnh, "img_align_prior_lambda_rot", 0.0);
  o.img_align_prior_lambda_trans = vk::param<double>(pnh, "img_align_prior_lambda_trans", 0.0);
  o.structure_optimization_max_pts = vk::param<int>(pnh, "structure_optimization_max_pts", 20);
  o.init_map_scale = vk::param<double>(pnh, "map_scale", 1.0);
  std::string default_kf_criterion = forward_default ? "FORWARD" : "DOWNLOOKING";
  if(vk::param<std::string>(pnh, "kfselect_criterion", default_kf_criterion) == "FORWARD")
    o.kfselect_criterion = KeyframeCriterion::FORWARD;
  else
    o.kfselect_criterion = KeyframeCriterion::DOWNLOOKING;
  o.kfselect_min_dist = vk::param<double>(pnh, "kfselect_min_dist", 0.12);
  o.kfselect_numkfs_upper_thresh = vk::param<int>(pnh, "kfselect_numkfs_upper_thresh", 120);
  o.kfselect_numkfs_lower_thresh = vk::param<double>(pnh, "kfselect_numkfs_lower_thresh", 70);
  o.kfselect_min_dist_metric = vk::param<double>(pnh, "kfselect_min_dist_metric", 0.01);
  o.kfselect_min_angle = vk::param<double>(pnh, "kfselect_min_angle", 20);
  o.kfselect_min_disparity = vk::param<double>(pnh, "kfselect_min_disparity", 40);
  o.kfselect_min_num_frames_between_kfs = vk::param<int>(pnh, "kfselect_min_num_frames_between_kfs", 2);
  o.kfselect_backend_max_time_sec = vk::param<double>(pnh, "kfselect_backend_max_time_sec", 3.0);
  o.img_align_max_level = vk::param<int>(pnh, "img_align_max_level", 4);
  o.img_align_min_level = vk::param<int>(pnh, "img_align_min_level", 2);
  o.img_align_robustification = vk::param<bool>(pnh, "img_align_robustification", false);
  o.img_align_use_distortion_jacobian =
      vk::param<bool>(pnh, "img_align_use_distortion_jacobian", false);
  o.img_align_est_illumination_gain =
      vk::param<bool>(pnh, "img_align_est_illumination_gain", false);
  o.img_align_est_illumination_offset =
      vk::param<bool>(pnh, "img_align_est_illumination_offset", false);
  o.poseoptim_thresh = vk::param<double>(pnh, "poseoptim_thresh", 2.0);
  o.update_seeds_with_old_keyframes =
      vk::param<bool>(pnh, "update_seeds_with_old_keyframes", true);
  o.use_async_reprojectors = vk::param<bool>(pnh, "use_async_reprojectors", false);
  o.trace_statistics = vk::param<bool>(pnh, "trace_statistics", false);
  o.backend_scale_stable_thresh =
      vk::param<double>(pnh, "backend_scale_stable_thresh", 0.02);
  o.global_map_lc_timeout_sec_ =
      vk::param<double>(pnh, "global_map_timeout_sec", 2.0);
  return o;
}

DetectorOptions loadDetectorOptions(const ros::NodeHandle& pnh)
{
  DetectorOptions o;
  o.cell_size = vk::param<int>(pnh, "grid_size", 35);
  o.max_level = vk::param<int>(pnh, "n_pyr_levels", 3) - 1;
  o.threshold_primary = vk::param<int>(pnh, "detector_threshold_primary", 10);
  o.threshold_secondary = vk::param<int>(pnh, "detector_threshold_secondary", 200);
  o.threshold_shitomasi = vk::param<int>(pnh, "detector_threshold_shitomasi", 100);
  if(vk::param<bool>(pnh, "use_edgelets", true))
    o.detector_type = DetectorType::kFastGrad;
  else
    o.detector_type = DetectorType::kFast;
  return o;
}

DepthFilterOptions loadDepthFilterOptions(const ros::NodeHandle& pnh)
{
  DepthFilterOptions o;
  o.max_search_level = vk::param<int>(pnh, "n_pyr_levels", 3) - 1;
  o.use_threaded_depthfilter =
      vk::param<bool>(pnh, "use_threaded_depthfilter", true);
  o.seed_convergence_sigma2_thresh =
      vk::param<double>(pnh, "seed_convergence_sigma2_thresh", 200.0);
  o.mappoint_convergence_sigma2_thresh =
      vk::param<double>(pnh, "mappoint_convergence_sigma2_thresh", 500.0);
  o.scan_epi_unit_sphere = vk::param<bool>(pnh, "scan_epi_unit_sphere", false);
  o.affine_est_offset= vk::param<bool>(pnh, "depth_filter_affine_est_offset", true);
  o.affine_est_gain = vk::param<bool>(pnh, "depth_filter_affine_est_gain", false);
  o.max_n_seeds_per_frame = static_cast<size_t>(
        static_cast<double>(vk::param<int>(pnh, "max_fts", 120))
        * vk::param<double>(pnh, "max_seeds_ratio", 3.0));
  o.max_map_seeds_per_frame = static_cast<size_t>(
        static_cast<double>(vk::param<int>(pnh, "max_map_fts", 120)));
  o.extra_map_points =
      vk::param<bool>(pnh, "depth_filter_extra_map_points", false);
  if(vk::param<bool>(pnh, "runlc", false) && !o.extra_map_points)
  {
    LOG(WARNING) << "Loop closure requires extra map points, "
                 << " but the option is not set, overriding to true.";
    o.extra_map_points = true;
  }
  return o;
}

InitializationOptions loadInitializationOptions(const ros::NodeHandle& pnh)
{
  InitializationOptions o;
  o.init_min_features = vk::param<int>(pnh, "init_min_features", 100);
  o.init_min_tracked = vk::param<int>(pnh, "init_min_tracked", 80);
  o.init_min_inliers = vk::param<int>(pnh, "init_min_inliers", 70);
  o.init_min_disparity = vk::param<double>(pnh, "init_min_disparity", 40.0);
  o.init_min_features_factor = vk::param<double>(pnh, "init_min_features_factor", 2.0);
  o.reproj_error_thresh = vk::param<double>(pnh, "reproj_err_thresh", 2.0);
  o.init_disparity_pivot_ratio = vk::param<double>(pnh, "init_disparity_pivot_ratio", 0.5);
  std::string init_method = vk::param<std::string>(pnh, "init_method", "FivePoint");
  if(init_method == "Homography")
    o.init_type = InitializerType::kHomography;
  else if(init_method == "TwoPoint")
    o.init_type = InitializerType::kTwoPoint;
  else if(init_method == "FivePoint")
    o.init_type = InitializerType::kFivePoint;
  else if(init_method == "OneShot")
    o.init_type = InitializerType::kOneShot;
  else
    SVO_ERROR_STREAM("Initialization Method not supported: " << init_method);
  return o;
}

FeatureTrackerOptions loadTrackerOptions(const ros::NodeHandle& pnh)
{
  FeatureTrackerOptions o;
  o.klt_max_level = vk::param<int>(pnh, "klt_max_level", 4);
  o.klt_min_level = vk::param<int>(pnh, "klt_min_level", 0.001);
  return o;
}

ReprojectorOptions loadReprojectorOptions(const ros::NodeHandle& pnh)
{
  ReprojectorOptions o;
  o.max_n_kfs = vk::param<int>(pnh, "reprojector_max_n_kfs", 5);
  o.max_n_features_per_frame = vk::param<int>(pnh, "max_fts", 160);
  o.cell_size = vk::param<int>(pnh, "grid_size", 35);
  o.reproject_unconverged_seeds =
      vk::param<bool>(pnh, "reproject_unconverged_seeds", true);
  o.max_unconverged_seeds_ratio =
      vk::param<double>(pnh, "max_unconverged_seeds_ratio", -1.0);
  o.min_required_features =
      vk::param<int>(pnh, "quality_min_fts", 50);
  o.seed_sigma2_thresh =
      vk::param<double>(pnh, "seed_convergence_sigma2_thresh", 200.0);

  o.affine_est_offset =
      vk::param<bool>(pnh, "reprojector_affine_est_offset", true);
  o.affine_est_gain =
      vk::param<bool>(pnh, "reprojector_affine_est_gain", false);
  o.max_fixed_landmarks =
      vk::param<int>(pnh, "reprojector_max_fixed_landmarks", 50);
  o.max_n_global_kfs =
      vk::param<int>(pnh, "reprojector_max_n_global_kfs", 20);
  o.use_kfs_from_global_map =
      vk::param<bool>(pnh, "reprojector_use_kfs_from_global_map", false);
  o.fixed_lm_grid_size =
      vk::param<int>(pnh, "reprojector_fixed_lm_grid_size", 50);

  return o;
}

CameraBundle::Ptr loadCameraFromYaml(const ros::NodeHandle& pnh)
{
  std::string calib_file = vk::param<std::string>(pnh, "calib_file", "~/cam.yaml");
  CameraBundle::Ptr ncam = CameraBundle::loadFromYaml(calib_file);
  std::cout << "loaded " << ncam->numCameras() << " cameras";
  for(const auto& cam : ncam->getCameraVector())
    cam->printParameters(std::cout, "");
  return ncam;
}

#ifdef SVO_LOOP_CLOSING
MapAlignmentOptions loadMapAlignmentOptions(const ros::NodeHandle& pnh)
{
  MapAlignmentOptions o;
  o.ransac3d_inlier_percent =
      vk::param<double>(pnh, "ransac3d_inlier_percent", 40.0);
  o.ransac3d_min_pts = vk::param<int>(pnh, "ransac3d_min_pts", 8);

  return o;
}

LoopClosureOptions loadLoopClosureOptions(const ros::NodeHandle& pnh)
{
  LoopClosureOptions o;
  o.runlc = vk::param<bool>(pnh, "runlc", false);
  o.voc_name = vk::param<std::string>(pnh, "voc_name", "test");
  std::string default_voc_path =
      ros::package::getPath("svo_online_loopclosing") + "/vocabularies/";
  o.voc_path = vk::param<std::string>(pnh, "voc_path", default_voc_path);
  o.alpha = vk::param<double>(pnh, "alpha", 40.0);
  o.beta = vk::param<double>(pnh, "beta", 20.0);
  o.ignored_past_frames = vk::param<int>(pnh, "frames_to_ignore", 5);
  o.scale_ret_app =
      vk::param<std::string>(pnh, "scale_retrieval_approach", "None");
  o.global_map_type =
      vk::param<std::string>(pnh, "global_map_type", "BuiltInPoseGraph");
  o.bowthresh = vk::param<double>(pnh, "bow_thresh", 0.6);
  o.gv_3d_inlier_thresh = vk::param<double>(pnh, "gv_3d_inlier_thresh", 0.4);
  o.min_num_3d = vk::param<int>(pnh, "min_num_3d", 4);
  o.orb_dist_thresh = vk::param<double>(pnh, "orb_dist_thresh", 32.0);
  o.gv_2d_match_thresh = vk::param<double>(pnh, "gv_2d_match_thresh", 0.10);
  o.use_opengv = vk::param<bool>(pnh, "use_opengv", false);
  o.enable_image_logging = vk::param<bool>(pnh, "enable_image_logging", false);
  o.image_log_base_path = vk::param<std::string>(pnh, "image_log_base_path", "/home/");
  o.proximity_dist_ratio = vk::param<double>(pnh, "proximity_dist_ratio", 0.01);
  o.proximity_offset = vk::param<double>(pnh, "proximity_offset", 0.1);
  o.force_correction_dist_thresh_meter =  vk::param<double>(pnh, "force_correction_dist_thresh_meter", 0.1);

  return o;
}

LoopClosingPtr getLoopClosingModule(const ros::NodeHandle &pnh,
                                    const CameraBundlePtr& cam)
{
  // Create camera
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);

  LoopClosureOptions lc_options = loadLoopClosureOptions(pnh);
  LoopClosingPtr lc = std::make_shared<LoopClosing>(lc_options, ncam);
  lc->map_alignment_se3_ =
      std::make_shared<MapAlignmentSE3>(loadMapAlignmentOptions(pnh));
  return lc;
}
#endif

#ifdef SVO_GLOBAL_MAP
GlobalMapPtr getGlobalMap(const ros::NodeHandle& pnh,
                          const CameraBundlePtr& ncams)
{
  GlobalMapOptions o;
  CameraBundle::Ptr cams = (ncams) ? ncams : loadCameraFromYaml(pnh);

  o.init_min_frame_before_opt =
      static_cast<size_t>(vk::param<int>(pnh, "gm_init_min_frame_opt", 10));
  o.init_min_lm_obs_frame = static_cast<size_t>(
        vk::param<int>(pnh, "gm_init_min_lm_obs_frame", 5));
  o.debug_fix_all_poses =
      vk::param<bool>(pnh, "gm_debug_fix_all_poses", false);
  o.print_stdcout =
      vk::param<bool>(pnh, "gm_print_stdcout", true);
  o.use_smart_factor =
      vk::param<bool>(pnh, "gm_use_smart_factor", false);
  o.use_imu =
      vk::param<bool>(pnh, "gm_use_imu", true);
  o.verbose =
      vk::param<bool>(pnh, "gm_verbose", false);
  o.use_thread =
      vk::param<bool>(pnh, "gm_use_thread", false);
  o.only_add_every_n =
        vk::param<int>(pnh, "gm_only_add_every_n", -1);
  o.optimize_every_n =
        vk::param<int>(pnh, "gm_optimize_every_n", 5);
  o.ignore_redundant_landmarks =
      vk::param<bool>(pnh, "gm_ignore_redundant_landmarks", false);
  o.ignore_grid_size =
        vk::param<int>(pnh, "gm_ignore_grid_size", 30);
  o.point_stable_thresh_meter =
        vk::param<double>(pnh, "gm_point_stable_thresh_meter", 0.05);
  o.inc_version_stable_ratio =
        vk::param<double>(pnh, "gm_inc_version_stable_ratio", 0.9);
  o.refine_extrinsics =
      vk::param<bool>(pnh, "refine_extrinsics", false);

  o.add_reprojected_fixed_landmark =
      vk::param<bool>(pnh, "gm_add_reprojected_fixed_landmark", false);
  o.max_reprojected_fixed_landmark_each_frame = static_cast<size_t>(
      vk::param<int>(pnh, "gm_max_reprojected_fixed_landmark_each_frame", 20));
  o.fix_lm_reobs_threshold =
      vk::param<int>(pnh, "gm_fix_lm_reobs_threshold", 5);

  o.min_num_obs =
      static_cast<size_t>(vk::param<int>(pnh, "gm_min_num_obs", 2));
  o.pt_cond_thresh =
      vk::param<double>(pnh, "gm_pt_cond_thresh", -1);
  o.pt_sigma_thresh =
      vk::param<double>(pnh, "gm_pt_sigma_thresh", 0.1);
  o.add_range_prior_to_good_pt =
      vk::param<bool>(pnh, "gm_add_range_prior_to_good_pt", true);
  o.max_point_age = vk::param<int>(pnh, "gm_max_point_age", 20);
  o.min_parallax_thresh_rad =
      vk::param<double>(pnh, "gm_min_parallax_deg", 5.0)/180.0*M_PI;
  o.use_edgelet = vk::param<bool>(pnh, "gm_use_edgelet", false);
  o.use_map_point = vk::param<bool>(pnh, "gm_use_map_point", false);
  o.mappoint_min_obs =
      static_cast<size_t>(vk::param<int>(pnh, "gm_map_point_min_obs", 4));

  o.use_relative_pose_fixation =
      vk::param<bool>(pnh, "gm_use_relative_pose_fixation", true);
  o.relative_prior_with_imu_min_time_sec =
      vk::param<double>(pnh, "gm_relative_prior_with_imu_min_time_sec", 1.0);
  o.frame_fixation_release_check =
      vk::param<bool>(pnh, "gm_frame_fixation_release_check", true);
  o.relative_min_common = static_cast<size_t>(
      vk::param<int>(pnh, "gm_relative_min_common_obs", 6));
  o.relative_release_min_common = static_cast<size_t>(
      vk::param<int>(pnh, "gm_relative_release_min_common_obs", 30));
  o.relative_pos_sigma_meter =
      vk::param<double>(pnh, "gm_relative_pos_sigma_meter", 0.1);
  o.relative_rot_sigma_rad =
      vk::param<double>(pnh, "gm_relative_rot_sigma_deg", 5.0)/180.0 * M_PI;

  o.query_frame_min_3d_points = static_cast<size_t>(
      vk::param<int>(pnh, "gm_query_frame_min_3d_points", 20));

  GraphManagerOptions gm_o;
  gm_o.use_bearing_factor =
      vk::param<bool>(pnh, "ba_use_bearing_factor", true);
  gm_o.reproj_error_ns_px =
      vk::param<double>(pnh, "ba_reproj_error_noise_px", 1.0);
  gm_o.smart_reproj_outlier_thresh_px =
      vk::param<double>(pnh, "ba_smart_reproj_outlier_thresh_px", 3.0);
  gm_o.use_robust_px_noise =
      vk::param<bool>(pnh, "ba_use_robust_px_noise", true);
  // prior factors
  gm_o.position_prior_sigma_meter =
      vk::param<double>(pnh, "ba_pos_sigma_meter", 0.001);
  gm_o.init_pos_sigma_meter = gm_o.position_prior_sigma_meter;
  gm_o.rotation_prior_sigma_rad =
      vk::param<double>(pnh, "ba_rot_sigma_degree", 1.0)/180.0*M_PI;
  gm_o.init_yaw_sigma_rad = gm_o.rotation_prior_sigma_rad;
  gm_o.init_roll_pitch_sigma_rad =
      vk::param<double>(pnh, "ba_init_roll_pitch_sigma_degree", 45)/180.0*M_PI;
  gm_o.point_prior_sigma_meter =
      vk::param<double>(pnh, "ba_point_sigma_meter", 0.001);

  GTSAMOptimizerOptions gtsam_opt_options;
  gtsam_opt_options.verbose =
      vk::param<bool>(pnh, "ba_verbose", false);
  gtsam_opt_options.max_iterations_per_update =
      vk::param<int>(pnh, "ba_max_iterations_per_update", 10);
  gtsam_opt_options.min_iterations_per_update =
      vk::param<int>(pnh, "ba_min_iterations_per_update", 3);
  gtsam_opt_options.max_time_sec_per_update =
      vk::param<int>(pnh, "ba_max_time_sec_per_update", 10.0);
  gtsam_opt_options.output_errors =
      vk::param<bool>(pnh, "ba_output_errors", false);
  gtsam_opt_options.trace_dir =
      vk::param<std::string>(pnh, "trace_dir",
                             ros::package::getPath("svo")+"/trace");
  gtsam_opt_options.optim_method =
      vk::param<std::string>(pnh, "isam_optimization_method",
                        std::string("GaussNewton"));
  gtsam_opt_options.isam_relinearize_thresh =
      vk::param<double>(pnh, "isam_relinearize_thresh", 0.1);
  gtsam_opt_options.isam_relinearize_skip =
      vk::param<int>(pnh, "isam_relinearize_skip", 10);
  gtsam_opt_options.isam_wildfire_thresh =
      vk::param<double>(pnh, "isam_wildfire_thresh", 0.001);
  gtsam_opt_options.isam_detailed_results =
      vk::param<bool>(pnh, "isam_detailed_results", false);

  GlobalMapPtr gm_ptr = std::make_shared<GlobalMap>(o, gm_o, gtsam_opt_options,
                                                    cams);

  return gm_ptr;
}
#endif

StereoTriangulationOptions loadStereoOptions(const ros::NodeHandle& pnh)
{
  StereoTriangulationOptions o;
  o.triangulate_n_features = vk::param<int>(pnh, "max_fts", 120);
  o.max_depth_inv = vk::param<double>(pnh, "max_depth_inv", 1.0/50.0);
  o.min_depth_inv = vk::param<double>(pnh, "min_depth_inv", 1.0/0.5);
  o.mean_depth_inv = vk::param<double>(pnh, "mean_depth_inv", 1.0/2.0);
  return o;
}

ImuHandler::Ptr getImuHandler(const ros::NodeHandle& pnh)
{
  std::string calib_file = vk::param<std::string>(pnh, "calib_file", "");
  ImuCalibration imu_calib = ImuHandler::loadCalibrationFromFile(calib_file);
  imu_calib.print("Loaded IMU Calibration");
  ImuInitialization imu_init = ImuHandler::loadInitializationFromFile(calib_file);
  imu_init.print("Loaded IMU Initialization");
  IMUHandlerOptions options;
  options.temporal_stationary_check =
      vk::param<bool>(pnh, "imu_temporal_stationary_check", false);
  options.temporal_window_length_sec_ =
      vk::param<double>(pnh, "imu_temporal_window_length_sec", 0.5);
  options.stationary_acc_sigma_thresh_ =
      vk::param<double>(pnh, "stationary_acc_sigma_thresh", 0.0);
  options.stationary_gyr_sigma_thresh_ =
      vk::param<double>(pnh, "stationary_gyr_sigma_thresh", 0.0);
  ImuHandler::Ptr imu_handler(new ImuHandler(imu_calib, imu_init, options));
  return imu_handler;
}

void setInitialPose(const ros::NodeHandle& pnh, FrameHandlerBase& vo)
{
  Transformation T_world_imuinit(
        Quaternion(vk::param<double>(pnh, "T_world_imuinit/qw", 1.0),
                   vk::param<double>(pnh, "T_world_imuinit/qx", 0.0),
                   vk::param<double>(pnh, "T_world_imuinit/qy", 0.0),
                   vk::param<double>(pnh, "T_world_imuinit/qz", 0.0)),
        Vector3d(vk::param<double>(pnh, "T_world_imuinit/tx", 0.0),
                 vk::param<double>(pnh, "T_world_imuinit/ty", 0.0),
                 vk::param<double>(pnh, "T_world_imuinit/tz", 0.0)));
  vo.setInitialImuPose(T_world_imuinit);
}


FrameHandlerMono::Ptr makeMono(const ros::NodeHandle& pnh, const CameraBundlePtr& cam)
{
  // Create camera
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);
  if (ncam->numCameras() > 1)
  {
    LOG(WARNING) << "Load more cameras than needed, will erase from the end.";
    ncam->keepFirstNCams(1);
  }

  // Init VO
  FrameHandlerMono::Ptr vo =
      std::make_shared<FrameHandlerMono>(
        loadBaseOptions(pnh, false),
        loadDepthFilterOptions(pnh),
        loadDetectorOptions(pnh),
        loadInitializationOptions(pnh),
        loadReprojectorOptions(pnh),
        loadTrackerOptions(pnh),
        ncam);

  // Get initial position and orientation of IMU
  setInitialPose(pnh, *vo);

  return vo;
}

FrameHandlerStereo::Ptr makeStereo(const ros::NodeHandle& pnh, const CameraBundlePtr& cam)
{
  // Load cameras
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);
  if (ncam->numCameras() > 2)
  {
    LOG(WARNING) << "Load more cameras than needed, will erase from the end.";
    ncam->keepFirstNCams(2);
  }

  // Init VO
  InitializationOptions init_options = loadInitializationOptions(pnh);
  init_options.init_type = InitializerType::kStereo;
  FrameHandlerStereo::Ptr vo =
      std::make_shared<FrameHandlerStereo>(
        loadBaseOptions(pnh, true),
        loadDepthFilterOptions(pnh),
        loadDetectorOptions(pnh),
        init_options,
        loadStereoOptions(pnh),
        loadReprojectorOptions(pnh),
        loadTrackerOptions(pnh),
        ncam);

  // Get initial position and orientation of IMU
  setInitialPose(pnh, *vo);

  return vo;
}

FrameHandlerArray::Ptr makeArray(const ros::NodeHandle& pnh, const CameraBundlePtr& cam)
{
  // Load cameras
  CameraBundle::Ptr ncam = (cam) ? cam : loadCameraFromYaml(pnh);

  // Init VO
  InitializationOptions init_options = loadInitializationOptions(pnh);
  init_options.init_type = InitializerType::kArrayGeometric;
  init_options.init_min_disparity = 25;
  DepthFilterOptions depth_filter_options = loadDepthFilterOptions(pnh);
  depth_filter_options.verbose = true;
  FrameHandlerArray::Ptr vo =
      std::make_shared<FrameHandlerArray>(
        loadBaseOptions(pnh, true),
        depth_filter_options,
        loadDetectorOptions(pnh),
        init_options,
        loadReprojectorOptions(pnh),
        loadTrackerOptions(pnh),
        ncam);

  // Get initial position and orientation of IMU
  setInitialPose(pnh, *vo);

  return vo;
}

} // namespace factory
} // namespace svo
