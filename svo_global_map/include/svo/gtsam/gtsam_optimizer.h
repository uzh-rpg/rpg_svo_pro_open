#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <unordered_map>

#include <svo/common/types.h>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>

namespace svo {

// fwd
class GraphManager;

struct GTSAMOptimizerOptions
{
  size_t max_iterations_per_update = 10;
  size_t min_iterations_per_update = 3;
  double max_time_sec_per_update = 10.0;

  bool verbose = false;
  bool output_errors = false;
  std::string trace_dir = "/tmp";
  double isam_relinearize_thresh = 0.1;
  double isam_relinearize_skip = 10.0;
  double isam_wildfire_thresh = 0.001;
  bool isam_detailed_results = false;
  std::string optim_method = "GaussNewton";
};

class GTSAMOptimizer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<GTSAMOptimizer> Ptr;

  GTSAMOptimizer(
      const GTSAMOptimizerOptions& options,
      std::shared_ptr<GraphManager>& graph);

  virtual ~GTSAMOptimizer();

  void reset();

  void initialize();

  void optimize();

  GTSAMOptimizerOptions options_;
  std::shared_ptr<gtsam::ISAM2> isam_;
  std::shared_ptr<GraphManager> graph_;

  // last computed estimate
  std::mutex estimate_mut_;
  gtsam::Values estimate_;
  std::atomic<BundleId> latest_estimate_state_index_ {-1}; // Id of last state in estimate.

  // thread
  std::mutex optimizer_cond_var_mut_;
  std::condition_variable optimizer_condition_var_;
  std::shared_ptr<std::thread> thread_;
  std::atomic<bool> quit_thread_;
  void startThread();
  void quitThread();

// protected -------------------------------------------------------------------

  void threadLoop();
  bool optimizeImpl();
};

} // namespace svo
