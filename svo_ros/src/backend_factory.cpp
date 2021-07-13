#include <svo_ros/backend_factory.h>

#include <memory>

#include <ros/package.h>
#include <vikit/params_helper.h>

#include <svo/imu_handler.h>
#include <svo/backend/backend_optimizer.h>
#include <svo/backend/backend_interface.h>
#include <svo/backend/graph_manager.h>

namespace svo {
namespace backend_factory {

BackendInterface::Ptr makeBackend(const ros::NodeHandle& pnh)
{
  GraphManagerOptions graph_manager_options;
  graph_manager_options.reproj_error_noise =
      vk::param<double>(pnh, "ba_reproj_error_noise", 1.0/370.0);
  graph_manager_options.smart_reproj_threshold =
      vk::param<double>(pnh, "ba_smart_reproj_threshold", 3.0/370.0);
  graph_manager_options.min_parallax_thresh =
      vk::param<double>(pnh, "ba_min_parallax_deg", 5.0)/180.0*M_PI;
  graph_manager_options.trace_tracks =
      vk::param<bool>(pnh, "ba_trace_tracks", false);
  graph_manager_options.use_robust_px_noise =
      vk::param<bool>(pnh, "ba_use_robust_px_noise", false);
  graph_manager_options.init_pos_sigma =
      vk::param<double>(pnh, "ba_init_pos_sigma_meter", 0.001);
  graph_manager_options.init_roll_pitch_sigma =
      vk::param<double>(pnh, "ba_init_roll_pitch_sigma_degree", 45)/180.0*M_PI;
  graph_manager_options.init_yaw_sigma =
      vk::param<double>(pnh, "ba_init_yaw_sigma_degree", 2)/180.0*M_PI;

  OptimizerBackendOptions backend_options;
  backend_options.verbose =
      vk::param<bool>(pnh, "ba_verbose", false);
  backend_options.max_iterations_per_update =
      vk::param<int>(pnh, "ba_max_iterations_per_update", 10);
  backend_options.output_errors =
      vk::param<bool>(pnh, "ba_output_errors", false);
  backend_options.trace_dir =
      vk::param<std::string>(pnh, "trace_dir", ros::package::getPath("svo")+"/trace");
  backend_options.isam_relinearize_thresh =
      vk::param<double>(pnh, "isam_relinearize_thresh", 0.1);
  backend_options.isam_relinearize_skip =
      vk::param<int>(pnh, "isam_relinearize_skip", 1);
  backend_options.isam_wildfire_thresh =
      vk::param<double>(pnh, "isam_wildfire_thresh", 0.001);
  backend_options.isam_detailed_results =
      vk::param<double>(pnh, "isam_detailed_results", false);

  BackendInterfaceOptions ba_interface_options;
  ba_interface_options.trace_dir = backend_options.trace_dir;
  ba_interface_options.isam_wait_time_ms =
      vk::param<int>(pnh, "isam_wait_time_ms", 2);
  ba_interface_options.use_smart_factors =
      vk::param<bool>(pnh, "ba_use_smart_factors", false);
  ba_interface_options.add_imu_factors =
      vk::param<bool>(pnh, "ba_add_imu_factors", false);
  ba_interface_options.min_num_obs =
      vk::param<int>(pnh, "ba_min_num_obs", 2);
  ba_interface_options.n_frames_in_init_ba =
      vk::param<int>(pnh, "ba_n_frames_in_init_ba", 8);

  BackendInterface::Ptr ba_interface =
      std::make_shared<BackendInterface>(
        ba_interface_options, graph_manager_options, backend_options);

  if(vk::param<bool>(pnh, "ba_parallelized", false))
    ba_interface->startThread();

  return ba_interface;
}

} // namespace vin_factory
} // namespace svo
