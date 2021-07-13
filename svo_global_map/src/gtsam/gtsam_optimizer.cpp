// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include "svo/gtsam/gtsam_optimizer.h"

// svo
#include <svo/common/frame.h>

// boost
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

// gtsam
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <rpg_common/timer.h>

#include "svo/gtsam/graph_manager.h"

namespace svo
{
GTSAMOptimizer::GTSAMOptimizer(const GTSAMOptimizerOptions& options,
                               std::shared_ptr<GraphManager>& graph)
  : options_(options), graph_(graph)
{
}

GTSAMOptimizer::~GTSAMOptimizer()
{
  quitThread();
}

void GTSAMOptimizer::reset()
{
  VLOG(3) << "Backend: Optimizer reset";
  latest_estimate_state_index_ = -1;
  estimate_.clear();
  isam_.reset();
}

void GTSAMOptimizer::initialize()
{
  VLOG(3) << "Backend: Optimizer init";
  gtsam::ISAM2Params isam_param;
  if (options_.optim_method == std::string("GaussNewton"))
  {
    gtsam::ISAM2GaussNewtonParams gauss_newton_params;
    gauss_newton_params.setWildfireThreshold(options_.isam_wildfire_thresh);
    isam_param.optimizationParams = gauss_newton_params;
  }
  else if (options_.optim_method == std::string("Dogleg"))
  {
    gtsam::ISAM2DoglegParams dogleg_params;
    dogleg_params.setWildfireThreshold(options_.isam_wildfire_thresh);
    isam_param.optimizationParams = dogleg_params;
  }
  else
  {
    LOG(FATAL) << "Unknown optimization method.";
  }

  // Initialize iSAM2
  isam_param.relinearizeThreshold = options_.isam_relinearize_thresh;
  isam_param.relinearizeSkip = options_.isam_relinearize_skip;
  isam_param.enableDetailedResults = options_.isam_detailed_results;  // TODO
  isam_param.factorization = gtsam::ISAM2Params::CHOLESKY;
  // isam_param.evaluateNonlinearError = true; // TODO only for debug
  isam_ = std::make_shared<gtsam::ISAM2>(isam_param);
}

void GTSAMOptimizer::optimize()
{
  // optimize
  if (thread_ == nullptr)
  {
    optimizeImpl();
  }
  else
  {
    optimizer_condition_var_.notify_one();  // notify the optimizer thread
  }
}

bool GTSAMOptimizer::optimizeImpl()
{
  rpg::Timer timer;
  timer.start();
  CHECK_NOTNULL(isam_.get());

  // ---------------------------------------------------------------------------
  // get copy of graph and value updates so the other thread is not
  // blocked during the isam update.
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_states;
  gtsam::FactorIndices delete_slots;
  std::vector<int> new_smart_factor_point_ids;
  BundleId last_added_state_id_before_optimization = -1;
  {
    std::lock_guard<std::mutex> lock(graph_->graph_mut_);
    graph_->getUpdatesCopy(&new_factors, &new_states, &delete_slots,
                           &new_smart_factor_point_ids);
    last_added_state_id_before_optimization = graph_->last_added_state_index_;
    VLOG(1) << ">>> === Graph update since last optimization started:\n"
            << "- all new factors: " << new_factors.size() << "; "
            << "new smart factors: " << new_smart_factor_point_ids.size()
            << "; "
            << "new states: " << new_states.size() << "; "
            << "deleted factors: " << delete_slots.size() << std::endl;
  }
  if (new_factors.empty() && new_states.empty() && delete_slots.empty())
  {
    VLOG(1) << "Nothing to update. This function should not be called.";
    return true;
  }

  if (new_smart_factor_point_ids.size() > 0)
  {
    size_t n_triang = 0;
    gtsam::Values all_values = isam_->getLinearizationPoint();
    all_values.insert(new_states);
    for (size_t i = 0; i < new_smart_factor_point_ids.size(); i++)
    {
      SmartFactorPtr s_ptr =
          boost::dynamic_pointer_cast<SmartFactor>(new_factors.at(i));
      CHECK_NOTNULL(s_ptr.get());
      if (!s_ptr->point().is_initialized() || !s_ptr->isValid())
      {
        if (s_ptr->triangulateSafe(s_ptr->cameras(all_values)).valid())
        {
          n_triang++;
        }
      }
    }
    std::cout << "Triangulated " << n_triang << " points before optimization."
              << std::endl;
  }

  // ---------------------------------------------------------------------------
  // compute update
  gtsam::ISAM2Result result;
  size_t n_iter = 0;
  VLOG(1) << "iSAM2 update with " << new_factors.size() << " graph updates"
          << " and " << new_states.size() << " new values "
          << " and " << delete_slots.size() << " delete indices";
  try
  {
    result = isam_->update(new_factors, new_states, delete_slots);
  }
  catch (const gtsam::IndeterminantLinearSystemException& e)
  {
    std::cerr << e.what() << std::endl;
    std::cerr << "crash in first update." << std::endl;
    throw;
  }
  ++n_iter;
  {
    std::lock_guard<std::mutex> lock(graph_->graph_mut_);
    graph_->updateSlotInfo(result.newFactorsIndices,
                           new_smart_factor_point_ids);
  }

  for (; n_iter < options_.max_iterations_per_update; ++n_iter)
  {
    try
    {
      result = isam_->update();
    }
    catch (const gtsam::IndeterminantLinearSystemException& e)
    {
      std::cerr << e.what() << std::endl;
      std::cerr << "crash in update: " << n_iter << std::endl;
      throw;
    }
    timer.stop();
    double cur_t = timer.getAccumulated();
    timer.resume();
    if (cur_t > options_.max_time_sec_per_update &&
        n_iter >= options_.min_iterations_per_update)
    {
      VLOG(1) << "Terminate optimization before max iterations:"
                 " already take too long time for "
              << n_iter << " iterations" << std::endl;
      break;
    }
  }

  // ---------------------------------------------------------------------------
  // copy estimate of the system
  {
    std::lock_guard<std::mutex> lock(estimate_mut_);
    try
    {
      estimate_ = isam_->calculateEstimate();
    }
    catch (const gtsam::IndeterminantLinearSystemException& e)
    {
      std::cerr << e.what() << std::endl;
      std::cerr << "crash after in calculateEstimate." << std::endl;
      gtsam::Key var = e.nearbyVariable();
      gtsam::Symbol symb(var);
      std::cerr << "Variable has type '" << symb.chr() << "' "
                << "and index " << symb.index() << std::endl;
      throw;
    }
    latest_estimate_state_index_ = last_added_state_id_before_optimization;
    if (options_.output_errors)
      VLOG(40) << "Final Error = "
               << isam_->getFactorsUnsafe().error(estimate_);
  }
  timer.stop();
  VLOG(1) << "Optimization took " << timer.getAccumulated() << " seconds."
          << std::endl;
  return true;
}

void GTSAMOptimizer::startThread()
{
  VLOG(3) << "BA: Started bundle adjustment thread";
  quit_thread_ = false;
  thread_ = std::make_shared<std::thread>(&GTSAMOptimizer::threadLoop, this);
}

void GTSAMOptimizer::quitThread()
{
  if (thread_)
  {
    quit_thread_ = true;
    optimizer_condition_var_.notify_all();
    thread_->join();
  }
  thread_ = nullptr;
}

void GTSAMOptimizer::threadLoop()
{
  while (!quit_thread_)
  {
    // optimize returns true when we have finished optimizing
    // it returns false, when we have to run another iteration
    if (optimizeImpl())
    {
      VLOG(10) << "Optimizer thread waiting ...";
      std::unique_lock<std::mutex> lock(optimizer_cond_var_mut_);
      optimizer_condition_var_.wait(lock);
      VLOG(10) << "Optimizer thread finished waiting.";
    }
  }
}

}  // namespace svo
