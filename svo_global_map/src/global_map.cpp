#include "svo/global_map.h"

#include <iomanip>

#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <svo/common/camera.h>
#include <svo/common/imu_calibration.h>
#include <gtsam/geometry/Pose3.h>

#include "svo/gtsam/graph_manager.h"
#include "svo/gtsam/gtsam_optimizer.h"
#include "svo/null_stream.h"

namespace
{
void calculateUnitBearingFromPoint(const svo::FramePtr& frame,
                                   const svo::PointPtr& pt, Eigen::Vector3d* f)
{
  (*f) = frame->pos() - pt->pos();
  f->normalize();
}
}

namespace svo
{
double CachedPointObservations::min_parallax_rad = 5.0 / 180 * M_PI;
size_t CachedPointObservations::min_num_obs = 3;
size_t CachedPointObservations::min_map_point_num_obs = 6;
double CachedPointObservations::cond_thresh = 30.0;
double CachedPointObservations::fim_det_cbr_minimum = 10.0;

void CachedPointObservations::addObservation(const FramePtr& f,
                                             const size_t ftr_idx)
{
  Eigen::Vector3d unit_bearing;
  calculateUnitBearingFromPoint(f, this->point_, &unit_bearing);
  ob_frames_.push_back(f);
  ftr_indices_.push_back(ftr_idx);
  obs_unit_bearing_from_pt_.push_back(unit_bearing);
  type_ = f->type_vec_[ftr_idx];

  if (!is_parallax_good_ && obs_unit_bearing_from_pt_.size() >= 2)
  {
    for (size_t i = 0; i < obs_unit_bearing_from_pt_.size() - 1; i++)
    {
      const double parallax = std::acos(
          obs_unit_bearing_from_pt_[i].dot(obs_unit_bearing_from_pt_.back()));
      if (parallax > min_parallax_rad)
      {
        is_parallax_good_ = true;
        break;
      }
    }
  }

  Eigen::Matrix<double, 2, 3> J;
  Eigen::Vector3d pt_in_f = f->T_f_w_.transform(point_->pos());
  Point::jacobian_xyz2uv(pt_in_f, f->T_f_w_.getRotationMatrix(), J);
  fim_ +=
      J.transpose() * J * (1.0 / std::pow(1.0 / f->getErrorMultiplier(), 2));
  Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(fim_);
  if (lu_decomp.rank() == 3)
  {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(fim_);
    cur_cond_num_ = svd.singularValues()(0) /
                    svd.singularValues()(svd.singularValues().size() - 1);
    cur_mean_det_ = std::cbrt(fim_.determinant());
  }
}

void CachedPointObservations::setAllObservationsToOutlier()
{
  for (size_t idx = 0; idx < ob_frames_.size(); idx++)
  {
    FramePtr& f = ob_frames_[idx];
    f->landmark_vec_[ftr_indices_[idx]] = nullptr;
    f->type_vec_[ftr_indices_[idx]] = FeatureType::kOutlier;
  }
}


GlobalMap::GlobalMap(const GlobalMapOptions& options,
                     const GraphManagerOptions& graph_manager_options,
                     const GTSAMOptimizerOptions& gtsam_opt_options,
                     const CameraBundlePtr& ncams)
  : options_(options)
  , out_((options_.print_stdcout ? std::cout : kNullOutput))
  , ncams_(ncams)
{
  if (options_.use_smart_factor && ncams_->getNumCameras() != 1)
  {
    LOG(FATAL) << "Smart factor only supports monocular case.";
  }
  graph_.reset(new GraphManager(graph_manager_options));
  optimizer_.reset(new GTSAMOptimizer(gtsam_opt_options, graph_));

  if (options_.pt_cond_thresh < 0)
  {
    std::cout << "Conditional number threshold for points are negative"
                 ", will not use the conditional number as a criteria.\n";
  }

  if (options_.use_imu && options_.use_relative_pose_fixation)
  {
    std::cout << "Use both IMU and relative constraints."
                 " Will only be used when the integration time is too long.\n";
  }

  if (options_.use_smart_factor)
  {
    std::cout << "With smart factor, we cannot use priors on point."
              << "Will disable range priors." << std::endl;
    options_.add_range_prior_to_good_pt = false;
  }

  CachedPointObservations::min_parallax_rad = options_.min_parallax_thresh_rad;
  CachedPointObservations::min_num_obs = options_.min_num_obs;
  CachedPointObservations::min_map_point_num_obs = options_.mappoint_min_obs;
  CachedPointObservations::cond_thresh = options_.pt_cond_thresh;
  CHECK_GT(options_.pt_sigma_thresh, 0);
  CachedPointObservations::fim_det_cbr_minimum = 1.0 / options_.pt_sigma_thresh;

  LOG(ERROR) << "Use the focal length of the first camera to scale error."
             << "Should implement for individual camera in the future.";
  graph_->initialize((ncams_->getCamera(0)).errorMultiplier());
  optimizer_->initialize();
  if (options_.use_thread)
  {
    optimizer_->startThread();
  }

  relative_pose_prior_ =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6)
               << gtsam::Vector3::Constant(options_.relative_rot_sigma_rad),
           gtsam::Vector3::Constant(options_.relative_pos_sigma_meter))
              .finished());
  std::cout << "GlobalMap: Initialized noise for immature states:\n";
  std::cout << "- Relative Pose:\n";
  relative_pose_prior_->print();
}

void GlobalMap::initializeIMUParams(const ImuCalibration& imu_calib,
                                    const ImuInitialization& imu_init)
{
  graph_->initializeImuNoiseModels(imu_calib, imu_init);
}

void GlobalMap::addKeyframe(const FramePtr& new_frame)
{
  static size_t kCount = 0;
  if (options_.only_add_every_n > 0)
  {
    kCount++;
    if (kCount % static_cast<size_t>(options_.only_add_every_n) != 0)
    {
      return;
    }
  }
  CHECK(new_frame);

  if (options_.ignore_redundant_landmarks && !reprojected_landmark_grid_)
  {
    const int cell_size = options_.ignore_grid_size;
    reprojected_landmark_grid_.reset(new OccupandyGrid2D(
        cell_size,
        OccupandyGrid2D::getNCell(new_frame->cam()->imageWidth(), cell_size),
        OccupandyGrid2D::getNCell(new_frame->cam()->imageHeight(), cell_size)));
  }

  // add to the id - keyframe map
  {
    auto it = frame_id_to_kf_in_graph_map_.find(new_frame->id());
    CHECK(it == frame_id_to_kf_in_graph_map_.end())
        << "Keyframe " << new_frame->id() << " already added.";
    if (options_.use_imu)
    {
      if (from_id_to_imu_meas_map_.find(new_frame->id()) ==
              from_id_to_imu_meas_map_.end() &&
          imu_cur_start_kf_id_ != new_frame->id())
      {
        out_ << "\n>>> Cannot find IMU measurements slot from frame id "
             << new_frame->id() << ", will not add to the graph." << std::endl;
        return;
      }
    }

    frame_id_to_kf_in_graph_map_[new_frame->id()] = new_frame;
    frame_id_to_vis_pt_ids_in_graph_map_[new_frame->id()] = UniquePointIds{};
    frame_id_to_vis_ftr_idx_in_graph_map_[new_frame->id()] = PtIdToFtrIdx{};
    prev_added_frame_id_ = newly_added_frame_id_;
    newly_added_frame_id_ = new_frame->id();
    frame_id_to_unoptimized_pose_map_[new_frame->id()] =
        new_frame->T_world_imu();
  }
  const int cur_bundle_id = new_frame->bundleId();
  const int cur_frame_id = new_frame->id();
  skip_fixed_lm_from_frontend_ =
      (matched_new_frame_ids_.find(new_frame->id()) !=
       matched_new_frame_ids_.end());

  // check whether we already have the keyframes from the same bundle
  bool same_bundle_already_added_ = false;
  {
    auto it = bundle_id_to_frame_id_map_.find(cur_bundle_id);
    if (it != bundle_id_to_frame_id_map_.end())
    {
      same_bundle_already_added_ = true;
      it->second.insert(cur_frame_id);
    }
    else
    {
      same_bundle_already_added_ = false;
      bundle_id_to_frame_id_map_[cur_bundle_id] =
          UniqueFrameIds{ cur_frame_id };
    }
  }

  // update the graph
  {
    std::lock_guard<std::mutex> lock(graph_->graph_mut_);
    // 1. add states to the graph
    if (!same_bundle_already_added_)
    {
      graph_->augmentStateWithPose(cur_bundle_id, new_frame->T_world_imu());
      out_ << "\n>>> Added frame id " << cur_bundle_id << " with time (sec) "
           << std::setprecision(std::numeric_limits<double>::digits10 + 1)
           << new_frame->getTimestampSec() << std::endl;
      VLOG(5) << "T_w_imu: " << new_frame->T_world_imu();
      VLOG(5) << "T_imu_cam: " << new_frame->T_imu_cam();

      if (options_.refine_extrinsics)
      {
        const size_t cam_id = static_cast<size_t>(new_frame->nframe_index_);
        if (!isTbcAdded(cam_id))
        {
          this->addTbc(new_frame->T_body_cam_, cam_id);
        }
      }

      // if we have IMU, also add other constraints
      if (options_.use_imu)
      {
        CHECK(frame_id_with_inertial_states_.find(new_frame->id()) ==
              frame_id_with_inertial_states_.end());
        graph_->augmentStateWithVelocityAndBias(
            new_frame->bundleId(), new_frame->imu_vel_w_,
            new_frame->imu_acc_bias_, new_frame->imu_gyr_bias_);
        frame_id_with_inertial_states_.insert(new_frame->id());

        if (prev_added_frame_id_ != -1)
        {
          ImuMeasurements imu_meas_from_prev;
          getIMUMeasurementsBetween(prev_added_frame_id_, newly_added_frame_id_,
                                    true, &imu_meas_from_prev);

          CHECK(imu_constraints_frame_id_pairs_.find(prev_added_frame_id_) ==
                imu_constraints_frame_id_pairs_.end());
          const FramePtr& prev_f =
              frame_id_to_kf_in_graph_map_[prev_added_frame_id_];
          int prev_bundle_id = prev_f->bundleId();
          graph_->addCombinedImuFactor(
              prev_bundle_id, cur_bundle_id, prev_f->imu_acc_bias_,
              prev_f->imu_gyr_bias_, new_frame->getTimestampSec(),
              imu_meas_from_prev);
          imu_constraints_frame_id_pairs_[prev_added_frame_id_] =
              newly_added_frame_id_;
        }
      }
    }
    else
    {
      LOG(ERROR) << "Encountered keyframe from the same bundle. "
                 << "Will not use add new bundle state.";
    }

    // 2. add visual measurements
    out_ << "Adding visual measurements...\n";
    if (options_.verbose)
    {
      out_ << "Added observations {frame - landmark}:" << std::endl;
    }
    if (options_.add_reprojected_fixed_landmark)
    {
      addReobservedLandmarkToGraph(new_frame);
    }
    addVisualMeasurementsToGraph(new_frame);
    if (options_.fix_lm_reobs_threshold > 0)
    {
      updateFixedLandmarks();
    }
    out_ << "Last added keyframe has "
         << frame_id_to_vis_pt_ids_in_graph_map_[cur_frame_id].size()
         << " observations." << std::endl;
    if (options_.verbose)
    {
      out_ << "The observations are:\n";
      for (const auto& v : frame_id_to_vis_pt_ids_in_graph_map_[cur_frame_id])
      {
        out_ << v << ", ";
      }
      out_ << std::endl;
    }

    // 3. Point matches
    addMatchingPointsToGraph();

    // 4. fixation and priors
    if (frame_id_to_kf_in_graph_map_.size() >=
        options_.init_min_frame_before_opt)
    {
      if (!initial_ba_fixed_)
      {
        initial_ba_fixed_ = checkAndFixInitialBA();
      }
      else
      {
        updatePriors();
      }
    }
  }  // manipulate graph
  n_since_last_optimization_++;

  //  new_frame->img_pyr_.clear();
  new_frame->original_color_image_.release();

  // optimize
  if (initial_ba_fixed_)
  {
    out_ << ">>> === GLOBAL MAP INFO ===" << std::endl
         << "- " << frame_id_to_kf_in_graph_map_.size() << " frames"
         << "; "
         << "- " << frame_id_with_inertial_states_.size() << " inertial states"
         << "; "
         << "- " << frame_ids_to_query_from_.size() << " to query from"
         << "; "
         << "- " << last_queried_kf_ids_.size() << " queried last time"
         << "; "
         << "- " << pt_id_to_lm_in_graph_map_.size() << " landmarks"
         << std::endl
         << "- " << fixed_frame_ids_other_than_intial_.size()
         << " frames (excluding the initial one) and " << fixed_pts_all_.size()
         << " points with priors; "
         << "- " << fixed_relative_frame_id_pairs_.size()
         << " relative pose constraints." << std::endl
         << "- " << n_matched_points_in_graph_
         << " matched points (diff ids) added in graph." << std::endl
         << "- " << n_reobserved_landmarks_in_graph_
         << " reobservations (same ids) added in graph." << std::endl
         << "- " << n_reobserved_landmarks_in_graph_frontend_
         << " reobservations from frontend added in graph." << std::endl
         << "- " << this->numMatchesToAdd()
         << " point matches waited to be added from "
         << match_points_info_to_add_.size() << " frame matches." << std::endl
         << "- " << imu_constraints_frame_id_pairs_.size() << " IMU constraints"
         << "; "
         << " - " << from_id_to_imu_meas_map_.size()
         << " imu measurement segments saved." << std::endl;
    if (needToOptimize())
    {
      this->optimize();
    }

    updateStates();

    if (options_.use_smart_factor)
    {
      graph_->updateSmartFactorStatistics();
      graph_->smart_factor_stats_.print();
    }
  }
}

void GlobalMap::optimize()
{
  try
  {
    optimizer_->optimize();
    n_since_last_optimization_ = 0;
    has_unoptimized_point_matches_ = false;
  }
  catch (const gtsam::IndeterminantLinearSystemException& e)
  {
    const gtsam::Values& estimate = optimizer_->estimate_;
    std::cerr << "Indeterminant:\n";
    std::cerr << "All points and # visible frames:\n";
    for (const auto& p : point_id_to_vis_frame_ids_in_graph_map_)
    {
      std::cerr << p.first << " - " << p.second.size() << ", ";
    }
    std::cerr << std::endl;

    std::cerr << "Poses input:\n";
    for (const auto& v : frame_id_to_kf_in_graph_map_)
    {
      std::cerr << "x" << v.first << ": " << v.second->pos().x() << ", "
                << v.second->pos().y() << ", " << v.second->pos().z() << "; ";
    }
    std::cerr << std::endl;

    std::cerr << "Points input:\n";
    for (const auto& v : pt_id_to_lm_in_graph_map_)
    {
      std::cerr << "l" << v.first << ": " << v.second->pos().x() << ", "
                << v.second->pos().y() << ", " << v.second->pos().z() << "; ";
    }
    std::cerr << std::endl;

    std::cerr << "Points in graph:\n";
    for (const auto& v : pt_id_to_lm_in_graph_map_)
    {
      Eigen::Vector3d pt = estimate
                               .at<gtsam::Point3>(gtsam::Symbol(
                                   'l', static_cast<size_t>(v.first)))
                               .vector();
      std::cerr << "l" << v.first << ": " << pt.x() << ", " << pt.y() << ", "
                << pt.z() << "; ";
    }
    std::cerr << std::endl;

    std::cerr << "Poses in graph:\n";
    for (const auto& v : frame_id_to_kf_in_graph_map_)
    {
      const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(
          gtsam::Symbol('x', static_cast<size_t>(v.second->bundleId())));
      std::cerr << "x" << v.first << ": " << pose.translation().x() << ", "
                << pose.translation().y() << ", " << pose.translation().z()
                << "; ";
    }
    std::cerr << std::endl;

    throw;
  }
}

void GlobalMap::updateStates()
{
  if (optimizer_->latest_estimate_state_index_ > last_updated_bundle_id_)
  {
    std::lock_guard<std::mutex> estimate_lock(optimizer_->estimate_mut_);
    updateStatesWithEstimate(optimizer_->estimate_,
                             optimizer_->latest_estimate_state_index_.load());
    last_updated_bundle_id_ = optimizer_->latest_estimate_state_index_.load();
  }
}

void GlobalMap::quitOptimizerThread()
{
  optimizer_->quitThread();
}

void GlobalMap::getIMUMeasurementsBetween(const int start_f_id,
                                          const int end_f_id, const bool remove,
                                          ImuMeasurements* imu_meas_from_start)
{
  CHECK_LT(start_f_id, end_f_id);
  auto it_s = frame_id_to_kf_in_graph_map_.find(start_f_id);
  CHECK(it_s != frame_id_to_kf_in_graph_map_.end());
  auto it_e = frame_id_to_kf_in_graph_map_.find(end_f_id);
  CHECK(it_e != frame_id_to_kf_in_graph_map_.end());

  const FramePtr& end_f = it_e->second;

  auto imu_it = from_id_to_imu_meas_map_.find(start_f_id);
  CHECK(imu_it != from_id_to_imu_meas_map_.end());
  for (; imu_it != from_id_to_imu_meas_map_.end();)
  {
    const ImuMeasurements& imu_meas_i = imu_it->second;
    if (imu_it->second.front().timestamp_ <= end_f->getTimestampSec())
    {
      imu_meas_from_start->insert(imu_meas_from_start->begin(),
                                  imu_meas_i.begin(), imu_meas_i.end());
      if (remove)
      {
        imu_it = from_id_to_imu_meas_map_.erase(imu_it);
      }
      else
      {
        imu_it++;
      }
    }
    else
    {
      CHECK_EQ(imu_it->first, end_f->id());
      break;
    }
  }

  out_ << std::setprecision(std::numeric_limits<double>::digits10 + 1)
       << "Found " << imu_meas_from_start->size() << " IMU measurements from "
       << start_f_id << "(" << imu_meas_from_start->back().timestamp_ << ")"
       << " to " << end_f_id << "(" << imu_meas_from_start->front().timestamp_
       << ")" << std::endl;
}

bool GlobalMap::checkAndFixInitialBA()
{
  if (options_.debug_fix_all_poses)
  {
    out_ << "Debug: Fix all frames in the initial structure." << std::endl;
    size_t idx = 0;
    for (auto it = frame_id_to_kf_in_graph_map_.cbegin();
         it != frame_id_to_kf_in_graph_map_.cend(); it++)
    {
      const FramePtr& f = it->second;
      fixFramePoseUnsafe(f, idx != 0);
      idx++;
    }
    return true;
  }

  out_ << "Global Map initialization: "
          "check and fix the initial BA problem."
       << std::endl;
  int max_frame_id = -1;
  size_t max_ftrs = 0;
  for (auto it = frame_id_to_kf_in_graph_map_.cbegin();
       it != frame_id_to_kf_in_graph_map_.cend(); it++)
  {
    const FramePtr& f = it->second;
    const UniquePointIds& vis_pts_in_ba =
        frame_id_to_vis_pt_ids_in_graph_map_[f->id()];
    out_ << "- Bundle id " << f->bundleId() << " (frame " << f->id() << ") has "
         << vis_pts_in_ba.size() << " visible landmarks." << std::endl;
    if (vis_pts_in_ba.size() > max_ftrs)
    {
      max_frame_id = it->first;
      max_ftrs = vis_pts_in_ba.size();
    }
  }

  if (max_ftrs < options_.init_min_lm_obs_frame)
  {
    out_ << "Initial structure not ready to be optimized yet." << std::endl;
    return false;
  }

  // pose and/or points fixation
  {
    if (options_.use_imu)
    {
      const Frame& init_f = *(frame_id_to_kf_in_graph_map_.begin()->second);
      const int init_bundle_id = init_f.bundleId();
      out_ << "=> Fixing frame state (position, yaw, bias, velocity) for "
           << init_bundle_id << "(" << init_f.id() << ")..." << std::endl;
      graph_->addInitalPosePriorVI(init_bundle_id, init_f.T_world_imu());
      graph_->addBiasPriorFactor(init_bundle_id, init_f.imu_acc_bias_,
                                 init_f.imu_gyr_bias_);
      graph_->addVelocityPriorFactor(init_bundle_id, init_f.imu_vel_w_);
    }
    else
    {
      size_t n_fixed_landmarks = 0;
      const FramePtr& max_f = frame_id_to_kf_in_graph_map_[max_frame_id];
      out_ << "=> Fixing frame pose" << max_frame_id << std::endl;
      const UniquePointIds& max_vis_pts_in_ba =
          frame_id_to_vis_pt_ids_in_graph_map_[max_frame_id];
      fixFramePoseUnsafe(max_f, false);
      out_ << "=> Fixing points: ";
      for (const int pid : max_vis_pts_in_ba)
      {
        out_ << pid;
        const PointPtr& pt = pt_id_to_lm_in_graph_map_[pid];
        // we fix it again anyway with a strong prior
        graph_->addPointPriorFactor(pt->id(), pt->pos(), -1.0);
        if (fixed_pts_all_.find(pid) != fixed_pts_all_.end())
        {
          out_ << "(already fixed); ";
        }
        else
        {
          registerPointFixed(pt->id());
        }
        out_ << ", ";
        n_fixed_landmarks++;
      }
      out_ << std::endl;
      CHECK_GE(n_fixed_landmarks, options_.init_min_lm_obs_frame);
      out_ << "Fixed " << n_fixed_landmarks << " points in initial BA."
           << std::endl;
    }
  }

  if (options_.use_relative_pose_fixation)
  {
    int prev_frame_id = -1;
    size_t n_fixed_relative = 0;
    for (auto it = frame_id_to_kf_in_graph_map_.cbegin();
         it != frame_id_to_kf_in_graph_map_.cend(); it++)
    {
      if (prev_frame_id != -1)
      {
        if (checkAndFixRelativePose(prev_frame_id, it->first))
        {
          n_fixed_relative++;
        }
      }
      prev_frame_id = it->first;
    }
    out_ << "Fixed " << n_fixed_relative
         << " relative pose pairs in initial BA." << std::endl;
  }

  return true;
}

void GlobalMap::updatePriors()
{
  if (options_.debug_fix_all_poses)
  {
    const FramePtr& new_f = frame_id_to_kf_in_graph_map_[newly_added_frame_id_];
    fixFramePoseUnsafe(new_f, true);
    return;
  }

  // sanity check on the fixation
  CHECK(initial_ba_fixed_);
  if (!options_.use_imu)
  {
    CHECK_GT(fixed_pts_all_.size(), 0u) << "We should have initial fixation "
                                           "already at this moment.";
  }

  if (options_.frame_fixation_release_check &&
      options_.use_relative_pose_fixation)
  {
    checkAndRemoveRelativePosePriors();
  }

  // check the common landmarks, potentially we can do something here
  if (options_.use_relative_pose_fixation)
  {
    if (frame_id_to_kf_in_graph_map_.size() > 2)
    {
      checkAndFixRelativePose(prev_added_frame_id_, newly_added_frame_id_);
    }
  }
}

bool GlobalMap::checkAndFixRelativePose(const int prev_id, const int cur_id)
{
  const FramePtr& prev_f = frame_id_to_kf_in_graph_map_[prev_id];
  const FramePtr& cur_f = frame_id_to_kf_in_graph_map_[cur_id];

  if (options_.use_imu)
  {
    const double dt = cur_f->getTimestampSec() - prev_f->getTimestampSec();
    if (dt < options_.relative_prior_with_imu_min_time_sec)
    {
      return false;
    }
  }

  bool add_rel_constraint = false;
  const UniquePointIds& prev_vis_pts_in_graph =
      frame_id_to_vis_pt_ids_in_graph_map_[prev_id];
  const UniquePointIds& cur_vis_pts_in_graph =
      frame_id_to_vis_pt_ids_in_graph_map_[cur_id];
  size_t n_common_lm =
      numCommonPoints(cur_vis_pts_in_graph, prev_vis_pts_in_graph);
  out_ << "There are " << n_common_lm << " common landmarks between " << prev_id
       << " and " << cur_id << std::endl;
  add_rel_constraint = n_common_lm < options_.relative_min_common;

  if (add_rel_constraint)
  {
    out_ << "Too few common landmarks,"
            " will add a relative constraint: "
         << prev_f->bundleId() << "(" << prev_f->id() << ")"
         << " - " << cur_f->bundleId() << "(" << cur_f->id() << ")"
         << std::endl;
    fixRelativePoseUnsafe(prev_f, cur_f, relative_pose_prior_);
    return true;
  }
  else
  {
    return false;
  }
}

void GlobalMap::addRangePriorUnsafe(const Point& pt, const Frame& f,
                                    const double ratio_dist)
{
  const double dist = (pt.pos() - f.T_world_cam().getPosition()).norm();
  gtsam::SharedNoiseModel range_noies =
      gtsam::noiseModel::Isotropic::Sigma(1, ratio_dist * dist);
  graph_->addRangePriorFactor(f.bundleId(), pt.id(), dist, f.T_imu_cam(),
                              range_noies);
}

void GlobalMap::fixFramePoseUnsafe(const FramePtr& f, const bool bookkeeping,
                                   const gtsam::SharedNoiseModel& noise)
{
  if (noise.get())
  {
    out_ << "Fix frame " << f->id() << " with given noise.\n";
    graph_->addPosePriorFactor(f->bundleId(), f->T_world_imu(), noise);
  }
  else
  {
    out_ << "Fix frame " << f->id() << " with default noise.\n";
    graph_->addPosePriorFactor(f->bundleId(), f->T_world_imu(), -1, -1);
  }
  if (bookkeeping)
  {
    CHECK(fixed_frame_ids_other_than_intial_.find(f->id()) ==
          fixed_frame_ids_other_than_intial_.end());
    fixed_frame_ids_other_than_intial_.insert(f->id());
  }
}

void GlobalMap::fixRelativePoseUnsafe(const FramePtr& f1, const FramePtr& f2,
                                      const gtsam::SharedNoiseModel& noise)
{
  CHECK_NOTNULL(noise.get());

  Transformation T_1_2 = frame_id_to_unoptimized_pose_map_[f1->id()].inverse() *
                         frame_id_to_unoptimized_pose_map_[f2->id()];
  graph_->addRelativePosePriorFactor(f1->bundleId(), f2->bundleId(), T_1_2,
                                     noise);
  CHECK(fixed_relative_frame_id_pairs_.find(f1->id()) ==
        fixed_relative_frame_id_pairs_.end());
  fixed_relative_frame_id_pairs_[f1->id()] = f2->id();
}

void GlobalMap::checkAndRemoveRelativePosePriors()
{
  for (auto it = fixed_relative_frame_id_pairs_.begin();
       it != fixed_relative_frame_id_pairs_.end();)
  {
    const int id1 = it->first;
    const int id2 = it->second;
    size_t n_common_lm =
        numCommonPoints(frame_id_to_vis_pt_ids_in_graph_map_[id1],
                        frame_id_to_vis_pt_ids_in_graph_map_[id2]);
    if (n_common_lm > options_.relative_release_min_common)
    {
      const int bid1 = frame_id_to_kf_in_graph_map_[id1]->bundleId();
      const int bid2 = frame_id_to_kf_in_graph_map_[id2]->bundleId();
      graph_->removeRelativePosePriorFactor(bid1, bid2);
      it = fixed_relative_frame_id_pairs_.erase(it);
      out_ << "Released the fixation for frame pair: " << id1 << " and " << id2
           << std::endl;
    }
    else
    {
      it++;
    }
  }
}

void GlobalMap::addVisualMeasurementsToGraph(const FramePtr& new_frame)
{
  size_t n_map_points = 0;
  size_t n_lm_points = 0;
  size_t n_fixed_lm = 0;

  size_t n_ignored_points = 0;
  size_t n_new_3D_points = 0;

  for (size_t ftr_idx = 0; ftr_idx < new_frame->numFeatures(); ftr_idx++)
  {
    FeatureType& ftr_type_ref = new_frame->type_vec_[ftr_idx];
    if (!isFeatureGoodToAdd(*new_frame, ftr_idx))
    {
      ftr_type_ref = FeatureType::kOutlier;
      new_frame->landmark_vec_[ftr_idx] = nullptr;
      n_ignored_points++;
      continue;
    }
    PointPtr& point = new_frame->landmark_vec_.at(ftr_idx);
    const int pt_id = point->id();

    if (isFixedLandmark(ftr_type_ref))
    {
      continue;
    }

    // statistics
    if (isMapPoint(ftr_type_ref))
    {
      if (options_.use_map_point)
      {
        n_map_points++;
      }
      else
      {
        continue;
      }
    }
    else
    {
      n_lm_points++;
    }

    // we set the type to fixed landmark after it is optimized
    ftr_type_ref = FeatureType::kOutlier;
    // We cache the point first until it gets enough observations
    auto it_cached = pt_id_to_cached_pt_obs_map_.find(pt_id);
    auto it_in_ba = pt_id_to_lm_in_graph_map_.find(pt_id);

    if (it_in_ba != pt_id_to_lm_in_graph_map_.end())
    {  // if it is already added to the graph
      CHECK(it_cached == pt_id_to_cached_pt_obs_map_.end());
      {
        this->addObservationToLandmark(new_frame, ftr_idx);
      }
    }
    else if (it_cached == pt_id_to_cached_pt_obs_map_.end())
    {  // if it is not in the graph and not cahced yet
      CachedPointObservations cached_pt_obs;
      cached_pt_obs.point_ = point;
      cached_pt_obs.addObservation(new_frame, ftr_idx);

      pt_id_to_cached_pt_obs_map_[pt_id] = cached_pt_obs;

      n_new_3D_points++;
    }
    else
    {
      CachedPointObservations& cur_cached_pt = it_cached->second;
      cur_cached_pt.addObservation(new_frame, ftr_idx);
    }
  }

  size_t n_added_good_landmarks = 0;
  size_t n_deleted_landmarks = 0;
  UniquePointIds good_lm_ids;
  {
    for (auto it = pt_id_to_cached_pt_obs_map_.begin();
         it != pt_id_to_cached_pt_obs_map_.cend();)
    {
      // sanity check
      CHECK_EQ(it->first, it->second.point_->id());
      CHECK(pt_id_to_lm_in_graph_map_.find(it->first) ==
            pt_id_to_lm_in_graph_map_.end());

      it->second.age_++;
      if (it->second.isPointGood())
      {
        CHECK_EQ(it->first, it->second.point_->id());
        good_lm_ids.insert(it->first);
        this->addCachedPointAndObservations(it->second);
        if (options_.add_range_prior_to_good_pt && !options_.use_smart_factor)
        {
          addRangePriorUnsafe(*(it->second.point_),
                              *(it->second.ob_frames_.front()), 0.8);
        }
        // bookkeeping
        it = pt_id_to_cached_pt_obs_map_.erase(it);
        n_added_good_landmarks++;
      }
      else if (options_.max_point_age > 0 &&
               it->second.age_ >= options_.max_point_age)
      {
        it->second.setAllObservationsToOutlier();
        it = pt_id_to_cached_pt_obs_map_.erase(it);
        n_deleted_landmarks++;
      }
      else
      {
        ++it;
      }
    }
  }
  out_ << "- Total " << n_lm_points << " normal points, "
       << "and " << n_map_points << " map points added,"
       << " ignored " << n_ignored_points << " points." << std::endl;
  out_ << "- Points cached but not added: "
       << pt_id_to_cached_pt_obs_map_.size() << std::endl;
  out_ << "- Added " << n_added_good_landmarks << " good landmarks, "
       << " and trashed " << n_deleted_landmarks << " landmarks.\n";
  out_ << " - New 3D points: " << n_new_3D_points << std::endl;

  if (n_new_3D_points > options_.query_frame_min_3d_points)
  {
    frame_ids_to_query_from_.insert(new_frame->id());
  }
  else
  {
    out_ << "Not enough new 3D points,"
            " will not add to the query frame.\n";
  }

  if (options_.verbose)
  {
    out_ << "Good landmarks are:\n";
    for (const auto id : good_lm_ids)
    {
      const Point& p = *(pt_id_to_lm_in_graph_map_[id]);
      out_ << id << ": " << p.pos().x() << ", " << p.pos().y() << ", "
           << p.pos().z() << "; ";
    }
    out_ << std::endl;
  }
}

void GlobalMap::addReobservedLandmarkToGraph(const FramePtr& new_frame)
{
  using FtrInfo = std::array<int, 2>;
  constexpr size_t kOldest = 0u;
  constexpr size_t kNumPrevObs = 1u;
  std::vector<std::pair<size_t, FtrInfo>> ftr_idx_to_n_observed;

  // select the good ones
  for (size_t ftr_idx = 0; ftr_idx < new_frame->numFeatures(); ftr_idx++)
  {
    if (new_frame->landmark_vec_[ftr_idx] == nullptr ||
        (!isFixedLandmark(new_frame->type_vec_[ftr_idx])))
    {
      continue;
    }
    const PointPtr& pt = new_frame->landmark_vec_[ftr_idx];
    FtrInfo ftr_info;
    ftr_info[kOldest] = pt->obs_.front().frame_id;
    ftr_info[kNumPrevObs] = std::count_if(pt->obs_.begin(), pt->obs_.end(),
                                          [&](const KeypointIdentifier& v) {
                                            return v.frame_id < new_frame->id();
                                          });
    ftr_idx_to_n_observed.emplace_back(std::make_pair(ftr_idx, ftr_info));
  }

  if (ftr_idx_to_n_observed.size() >
      options_.max_reprojected_fixed_landmark_each_frame)
  {
    std::nth_element(ftr_idx_to_n_observed.begin(),
                     ftr_idx_to_n_observed.begin() +
                         options_.max_reprojected_fixed_landmark_each_frame,
                     ftr_idx_to_n_observed.end(),
                     [&](const std::pair<size_t, FtrInfo>& lhs,
                         const std::pair<size_t, FtrInfo>& rhs) {
                       return (lhs.second[kOldest] < rhs.second[kOldest]) ||
                              (lhs.second[kOldest] == rhs.second[kOldest] &&
                               lhs.second[kNumPrevObs] >
                                   rhs.second[kNumPrevObs]);
                     });
  }

  size_t n_added_fixed = 0;
  if (reprojected_landmark_grid_)
  {
    reprojected_landmark_grid_->reset();
  }
  for (size_t idx = 0;
       idx < options_.max_reprojected_fixed_landmark_each_frame &&
       idx < ftr_idx_to_n_observed.size();
       idx++)
  {
    const auto& ftr_obs = ftr_idx_to_n_observed[idx];
    const size_t ftr_idx = ftr_obs.first;
    if (!hasFutureObservations(*new_frame, ftr_idx) ||
        new_frame->landmark_vec_[ftr_idx]->obs_.size() <= 2)
    {
      continue;
    }
    addObservationToLandmark(new_frame, ftr_idx);
    registerFrameRelatedPoints(new_frame->id(),
                               new_frame->landmark_vec_[ftr_idx]->id(),
                               &frame_id_to_reobs_pts_ids_frontend_);
    registerReobservedLandmarks(new_frame->landmark_vec_[ftr_idx]->id());
    n_reobserved_landmarks_in_graph_frontend_++;
    n_added_fixed++;
    if (reprojected_landmark_grid_)
    {
      reprojected_landmark_grid_->fillWithKeypoints(
          new_frame->px_vec_.col(static_cast<int>(ftr_idx)));
    }
  }

  out_ << "- Added " << n_added_fixed << " reobserved fixed landmarks "
                                         "from the frontend.\n";
}

void GlobalMap::updateFixedLandmarks()
{
  size_t n_new_fixed_lms = 0;
  for (auto it = lm_id_to_num_reobs_map_.begin();
       it != lm_id_to_num_reobs_map_.end();)
  {
    if (it->second >= static_cast<size_t>(options_.fix_lm_reobs_threshold) &&
        isPointOptimized(it->first))
    {
      graph_->addPointPriorFactor(
          it->first, pt_id_to_lm_in_graph_map_[it->first]->pos(), -1);
      registerPointFixed(it->first);
      fixed_landmarks_in_graph_.insert(it->first);
      it = lm_id_to_num_reobs_map_.erase(it);
      n_new_fixed_lms++;
    }
    else
    {
      it++;
    }
  }
  out_ << "- newly fixed " << n_new_fixed_lms << " points in graph.\n";
}

void GlobalMap::addMatchingPointsToGraph()
{
  size_t n_diff_id_matches = 0;
  size_t n_same_id_matches = 0;
  size_t n_same_id_trials = 0;
  for (auto match_it = match_points_info_to_add_.begin();
       match_it != match_points_info_to_add_.end();)
  {
    MatchedPointsInfo& cur_match_info = *match_it;
    if (!frameInGraph(cur_match_info.lc_kf_id_) ||
        !frameInGraph(cur_match_info.cur_kf_id_))
    {
      match_it++;
      continue;
    }

    for (auto pit = cur_match_info.pt_id_matches_.begin();
         pit != cur_match_info.pt_id_matches_.end();)
    {
      if (!pointInGraph(pit->first) || !pointInGraph(pit->second))
      {
        pit++;
        continue;
      }

      n_same_id_trials++;
      const int first_id = pit->first;
      const int second_id = pit->second;
      if (first_id == second_id)
      {
        bool all_in_graph = true;
        for (const int f_id :
             { cur_match_info.lc_kf_id_, cur_match_info.cur_kf_id_ })
        {
          if (pointVisibleInFrameInGraph(first_id, f_id))
          {
            continue;
          }
          bool res = addObservationToLandmarkByPointId(
              frame_id_to_kf_in_graph_map_[f_id], first_id);
          n_same_id_matches += (res ? 1 : 0);
          all_in_graph &= res;
        }

        if (all_in_graph)
        {
          n_reobserved_landmarks_in_graph_++;
          for (const int f_id :
               { cur_match_info.lc_kf_id_, cur_match_info.cur_kf_id_ })
          {
            registerFrameRelatedPoints(f_id, first_id,
                                       &frame_id_to_reobs_pts_ids_);
          }
        }
        pit = (all_in_graph ? cur_match_info.pt_id_matches_.erase(pit) :
                              std::next(pit));
      }
      else
      {
        graph_->addPointMatchFactor(first_id, second_id);
        registerFrameRelatedPoints(cur_match_info.lc_kf_id_, first_id,
                                   &frame_id_to_matched_pts_id_);
        registerFrameRelatedPoints(cur_match_info.cur_kf_id_, second_id,
                                   &frame_id_to_matched_pts_id_);
        n_matched_points_in_graph_++;
        n_diff_id_matches++;
        pit = cur_match_info.pt_id_matches_.erase(pit);
      }
    }

    // move the next one
    match_it = (cur_match_info.pt_id_matches_.empty() ?
                    match_points_info_to_add_.erase(match_it) :
                    std::next(match_it));
  }
  const size_t n_matched_total = n_diff_id_matches + n_same_id_matches;
  has_unoptimized_point_matches_ &= (n_matched_total > 0u);
  if (n_matched_total > 0)
  {
    out_ << "Newly added point matches:\n"
         << "- " << n_diff_id_matches << " with different ids\n"
         << "- " << n_same_id_matches << " out of " << n_same_id_trials
         << " with same ids (reobserve)." << std::endl;
  }
}

bool GlobalMap::isFeatureGoodToAdd(const Frame& f, const size_t ftr_idx)
{
  // check type
  const FeatureType& ft = f.type_vec_[ftr_idx];

  // fixed landmark will be handled differently
  if (isFixedLandmark(ft))
  {
    return true;
  }

  if (ft == FeatureType::kOutlier || (!options_.use_edgelet && isEdgelet(ft)))
  {
    return false;
  }
  // point
  if (f.landmark_vec_[ftr_idx] == nullptr)
  {
    return false;
  }
  // different observations check
  const PointPtr& point = f.landmark_vec_.at(ftr_idx);
  if (!f.isVisible(point->pos()))
  {
    return false;
  }

  if (point->n_succeeded_reproj_ < point->n_failed_reproj_ ||
      point->n_failed_reproj_ > 10)
  {
    return false;
  }

  if (reprojected_landmark_grid_)
  {
    const size_t cur_cell_idx = reprojected_landmark_grid_->getCellIndex(
        f.px_vec_.col(static_cast<int>(ftr_idx)));
    if (reprojected_landmark_grid_->isOccupied(cur_cell_idx))
    {
      return false;
    }
  }

  return true;
}

void GlobalMap::addTbc(svo::Transformation& Tbc, const size_t cam_id)
{
  graph_->addExtrinsics(Tbc, cam_id);
  cam_id_to_Tbc_in_graph_map_.emplace(std::make_pair(cam_id, Tbc));
}

void GlobalMap::addCachedPointAndObservations(
    const CachedPointObservations& cached)
{
  CHECK_EQ(cached.point_->nRefs(), 0u);
  this->addLandmark(*(cached.point_), *(cached.ob_frames_.front()));
  for (size_t idx = 0; idx < cached.ob_frames_.size(); idx++)
  {
    const FramePtr& frame_i = cached.ob_frames_[idx];
    this->addObservationToLandmark(frame_i, cached.ftr_indices_[idx]);
  }

  CHECK(pt_id_to_lm_in_graph_map_.find(cached.point_->id()) ==
        pt_id_to_lm_in_graph_map_.end());
  pt_id_to_lm_in_graph_map_[cached.point_->id()] = cached.point_;
}

void GlobalMap::addLandmark(svo::Point& pt, const Frame& f)
{
  if (options_.use_smart_factor && initial_ba_fixed_)
  {
    graph_->addSmartLandmark(pt, f.T_imu_cam());
  }
  else
  {
    graph_->addLandmark(pt);
  }
  pt.n_failed_reproj_ = 0;
  pt.in_ba_graph_ = false;
  pt.n_succeeded_reproj_ = 0;
  CHECK(point_id_to_vis_frame_ids_in_graph_map_.find(pt.id()) ==
        point_id_to_vis_frame_ids_in_graph_map_.end());
  point_id_to_vis_frame_ids_in_graph_map_[pt.id()] = UniqueFrameIds{};
}

void GlobalMap::addObservationToLandmark(const FramePtr& f,
                                         const size_t ftr_idx)
{
  if (graph_->isPointIDSmartLandmark(f->landmark_vec_[ftr_idx]->id()))
  {
    graph_->addObservationToSmartLandmark(*f, ftr_idx);
  }
  else
  {
    graph_->addObservationToLandmark(*f, ftr_idx, isTbcAdded(f->nframe_index_));
  }
  if (isPointOptimized(f->landmark_vec_[ftr_idx]->id()))
  {
    f->type_vec_[ftr_idx] = FeatureType::kFixedLandmark;
  }
  else
  {
    f->type_vec_[ftr_idx] = FeatureType::kOutlier;
  }
  f->landmark_vec_[ftr_idx]->addObservation(f, ftr_idx);
  if (options_.verbose)
  {
    out_ << f->id() << ": " << f->landmark_vec_[ftr_idx]->id() << ", ";
  }
  registerPointVisibleToFrame(f->id(), f->landmark_vec_[ftr_idx]->id(),
                              ftr_idx);
}

bool GlobalMap::addObservationToLandmarkByPointId(const FramePtr& f,
                                                  const int point_id)
{
  CHECK(f != nullptr);
  bool suc = false;
  for (size_t ftr_idx = 0; ftr_idx < f->numFeatures(); ftr_idx++)
  {
    if (f->landmark_vec_[ftr_idx] != nullptr &&
        f->landmark_vec_[ftr_idx]->id() == point_id)
    {
      this->addObservationToLandmark(f, ftr_idx);
      suc = true;
      break;
    }
  }

  return suc;
}

void GlobalMap::updateStatesWithEstimate(
    const gtsam::Values& estimate, const BundleId last_optimized_state_index)
{
  updated_point_ids_.clear();
  stable_point_ids_.clear();

  if (options_.refine_extrinsics)
  {
    CHECK(!cam_id_to_Tbc_in_graph_map_.empty());
    for (auto& v : cam_id_to_Tbc_in_graph_map_)
    {
      gtsam::Key Tbc_key = gtsam::Symbol('C', v.first);
      if (estimate.exists(Tbc_key))
      {
        const gtsam::Pose3 Tbc = estimate.at<gtsam::Pose3>(Tbc_key);
        v.second = Transformation(Tbc.matrix());
      }
    }
  }

  for (auto& v : frame_id_to_kf_in_graph_map_)
  {
    FramePtr& f = v.second;
    if (f->bundleId() > last_optimized_state_index)
    {
      continue;
    }
    gtsam::Key pose_key =
        gtsam::Symbol('x', static_cast<size_t>(f->bundleId()));
    CHECK(estimate.exists(pose_key));
    const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(pose_key);
    const size_t cam_id = static_cast<size_t>(f->nframe_index_);
    if (isTbcAdded(cam_id))
    {
      f->set_T_cam_imu(cam_id_to_Tbc_in_graph_map_[cam_id].inverse());
    }
    f->set_T_w_imu(Transformation(pose.matrix()));

    auto it_vis = frame_id_to_vis_pt_ids_in_graph_map_.find(f->id());
    CHECK(it_vis != frame_id_to_vis_pt_ids_in_graph_map_.end());
    for (const auto pt_id : it_vis->second)
    {
      const size_t ftr_idx = ftrIdx(pt_id, f->id());
      CHECK(f->landmark_vec_[ftr_idx])
          << "Feature type: " << str(f->type_vec_[ftr_idx])
          << "; frame id: " << f->id();

      if (isPointOptimized(pt_id))
      {
        f->type_vec_[ftr_idx] = isPointStable(pt_id) ?
                                    FeatureType::kFixedLandmark :
                                    FeatureType::kOutlier;
        continue;
      }
      Eigen::Vector3d new_pos;
      const PointPtr& pt = pt_id_to_lm_in_graph_map_[pt_id];
      if (getPointFromGraphByIdUnsafe(pt_id, estimate, &new_pos))
      {
        if (options_.point_stable_thresh_meter > 0 &&
            (new_pos - pt->pos_).norm() > options_.point_stable_thresh_meter)
        {
          f->type_vec_[ftr_idx] = FeatureType::kOutlier;
        }
        else
        {
          stable_point_ids_.insert(pt_id);
          f->type_vec_[ftr_idx] = FeatureType::kFixedLandmark;
        }
      }
      else
      {
        f->type_vec_[ftr_idx] = FeatureType::kOutlier;
      }
      pt->pos_ = new_pos;
      updated_point_ids_.insert(pt_id);
    }
    f->resetKeyPoints();
    f->setKeyPoints();
  }

  const double stable_rate =
      (1.0 * stable_point_ids_.size() / updated_point_ids_.size());
  if (stable_rate > options_.inc_version_stable_ratio)
  {
    Point::global_map_value_version_++;
  }
  out_ << "Updated " << updated_point_ids_.size() << " with "
       << stable_point_ids_.size() << " points stable (" << stable_rate * 100
       << ")." << std::endl;
}

void GlobalMap::getOverlapKeyframesMaxN(const Frame& query_f, const int max,
                                        std::vector<FramePtr>* overlap_kfs)
{
  CHECK_NOTNULL(overlap_kfs);
  overlap_kfs->clear();
  last_queried_kf_ids_.clear();

  const Eigen::Vector3d cam_pos = query_f.pos();
  std::vector<std::pair<int, double>> frame_id_to_dist_vec;

  for (auto& p : frame_ids_to_query_from_)
  {
    const FramePtr& kf = frame_id_to_kf_in_graph_map_[p];
    if (kf->bundleId() > last_updated_bundle_id_)
    {
      break;
    }
    frame_id_to_dist_vec.emplace_back(
        std::make_pair(kf->id(), (kf->pos() - cam_pos).norm()));
  }
  std::sort(
      frame_id_to_dist_vec.begin(), frame_id_to_dist_vec.end(),
      [](const std::pair<int, double>& p1, const std::pair<int, double>& p2) {
        return p1.second < p2.second;
      });

  size_t max_kfs =
      max > 0 ? static_cast<size_t>(max) : frame_id_to_dist_vec.size();
  size_t n_found = 0;
  for (const auto& p : frame_id_to_dist_vec)
  {
    const FramePtr& kf = frame_id_to_kf_in_graph_map_[p.first];
    for (const auto& keypoint : kf->key_pts_)
    {
      if (keypoint.first == -1)
      {
        continue;
      }

      if (query_f.isVisible(keypoint.second))
      {
        overlap_kfs->emplace_back(kf);
        last_queried_kf_ids_.insert(kf->id());
        n_found++;
        break;
      }
    }
    if (n_found >= max_kfs)
    {
      break;
    }
  }
  // sanity check
  //  for (const FramePtr& f : *overlap_kfs)
  //  {
  //    checkFrameFromGlobalMap(f);
  //  }
  //  overlap_kfs->clear();
}

void GlobalMap::getAllKFPositions(rpg::PositionVec* kf_pos,
                                  rpg::PositionVec* kf_pos_with_priors) const
{
  CHECK_NOTNULL(kf_pos);
  kf_pos->clear();
  CHECK_NOTNULL(kf_pos_with_priors);
  kf_pos_with_priors->clear();
  for (const auto& v : frame_id_to_kf_in_graph_map_)
  {
    if (v.second->bundleId() > last_updated_bundle_id_)
    {
      break;
    }
    if (fixed_relative_frame_id_pairs_.find(v.first) !=
            fixed_relative_frame_id_pairs_.end() ||
        fixed_frame_ids_other_than_intial_.find(v.first) !=
            fixed_frame_ids_other_than_intial_.end())
    {
      kf_pos_with_priors->emplace_back(v.second->imuPos());
    }
    else
    {
      kf_pos->emplace_back(v.second->imuPos());
    }
  }
}

void GlobalMap::getQueryKFPositions(rpg::PositionVec* query_kf_pos,
                                    rpg::PositionVec* last_queried_kf_pos) const
{
  CHECK_NOTNULL(query_kf_pos);
  query_kf_pos->clear();
  CHECK_NOTNULL(last_queried_kf_pos);
  last_queried_kf_pos->clear();
  for (const auto& v : frame_ids_to_query_from_)
  {
    if (v > last_updated_bundle_id_)
    {
      break;
    }
    const FramePtr& f = frame_id_to_kf_in_graph_map_.at(v);
    if (last_queried_kf_ids_.find(v) != last_queried_kf_ids_.end())
    {
      last_queried_kf_pos->emplace_back(f->imuPos());
    }
    else
    {
      query_kf_pos->emplace_back(f->imuPos());
    }
  }
}

void GlobalMap::getBundleIDIMUPosesMap(BundleIdToIMUPose* pose_map) const
{
  CHECK_NOTNULL(pose_map);
  pose_map->clear();
  std::lock_guard<std::mutex> estimate_lock(optimizer_->estimate_mut_);
  const gtsam::Values& estimate = optimizer_->estimate_;
  for (const auto& v : frame_id_to_kf_in_graph_map_)
  {
    gtsam::Key pose_key =
        gtsam::Symbol('x', static_cast<size_t>(v.second->bundleId()));
    if (!estimate.exists(pose_key))
    {
      continue;
    }
    const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(pose_key);
    Transformation T_w_b(pose.matrix());
    pose_map->insert(std::make_pair(v.second->bundleId(), T_w_b));
  }
}

void GlobalMap::getAllPoints(rpg::PositionVec* points,
                             rpg::PositionVec* points_with_priors,
                             rpg::PositionVec* unstable_points) const
{
  CHECK_NOTNULL(points);
  points->clear();
  CHECK_NOTNULL(points_with_priors);
  points_with_priors->clear();
  CHECK_NOTNULL(unstable_points);
  unstable_points->clear();
  for (const auto& v : pt_id_to_lm_in_graph_map_)
  {
    if (!isPointOptimized(v.first))
    {
      continue;
    }

    if (fixed_pts_all_.find(v.first) != fixed_pts_all_.end())
    {
      points_with_priors->emplace_back(v.second->pos());
    }
    else if (!isPointStable(v.first))
    {
      unstable_points->emplace_back(v.second->pos());
    }
    else
    {
      points->emplace_back(v.second->pos());
    }
  }
}

bool GlobalMap::getPointFromGraphByIdUnsafe(const int pt_id,
                                            const gtsam::Values& estimate,
                                            Eigen::Vector3d* pt) const
{
  CHECK_NOTNULL(pt);
  if (options_.use_smart_factor && graph_->isPointIDSmartLandmark(pt_id))
  {
    const SmartFactorPtr factor = graph_->getSmartFacotrConst(pt_id);
    if (factor && factor->point().is_initialized() && factor->isValid())
    {
      //      (*pt) = factor->point(estimate).get().vector();
      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    gtsam::Key pt_key = gtsam::Symbol('l', static_cast<size_t>(pt_id));
    if (estimate.exists(pt_key))
    {
      *pt = estimate.at<gtsam::Point3>(pt_key).vector();
      return true;
    }
    else
    {
      return false;
    }
  }
}

void GlobalMap::getVizLinks(VisibilityInfoVec* vis_info, const int last_n) const
{
  CHECK_NOTNULL(vis_info);
  vis_info->clear();

  int cnt = 0;
  for (auto rit = frame_id_to_kf_in_graph_map_.rbegin();
       rit != frame_id_to_kf_in_graph_map_.rend(); rit++)
  {
    if (last_n > 0 && cnt >= last_n)
    {
      break;
    }

    std::map<int, UniquePointIds>::const_iterator vis_cit =
        frame_id_to_vis_pt_ids_in_graph_map_.find(rit->second->id());
    CHECK(vis_cit != frame_id_to_vis_pt_ids_in_graph_map_.cend());
    for (const int pt_id : vis_cit->second)
    {
      vis_info->emplace_back(VisibilityInfo());
      getKfToPointLink(rit->second, pt_id, &(vis_info->back()));
    }

    cnt++;
  }
}

void GlobalMap::getKeypointsVisLinks(VisibilityInfoVec* vis_info) const
{
  CHECK_NOTNULL(vis_info);
  vis_info->clear();
  for (const auto& e : frame_id_to_kf_in_graph_map_)
  {
    if (e.first > last_updated_bundle_id_)
    {
      break;
    }

    const FramePtr& f = e.second;
    const Eigen::Vector3f f_pos = f->T_world_imu().getPosition().cast<float>();
    for (const auto& pt : f->key_pts_)
    {
      if (pt.first == -1)
      {
        continue;
      }
      const Eigen::Vector3f pt_pos = pt.second.cast<float>();
      vis_info->emplace_back((VisibilityInfo() << f_pos.x(), f_pos.y(),
                              f_pos.z(), pt_pos.x(), pt_pos.y(), pt_pos.z())
                                 .finished());
    }
  }
}

void GlobalMap::getPointPositionAndIds(rpg::PositionVec* points,
                                       std::vector<int>* ids) const
{
  CHECK_NOTNULL(points);
  points->clear();
  CHECK_NOTNULL(ids);
  ids->clear();

  points->reserve(pt_id_to_lm_in_graph_map_.size());
  ids->reserve(pt_id_to_lm_in_graph_map_.size());
  for (const auto& pt : pt_id_to_lm_in_graph_map_)
  {
    points->emplace_back(pt.second->pos());
    ids->emplace_back(pt.first);
  }
}

void GlobalMap::framePointsToVisLink(const std::map<int, UniquePointIds>& d,
                                     VisibilityInfoVec* vis_info) const
{
  for (const auto& kf_pt : d)
  {
    const FramePtr& f = frame_id_to_kf_in_graph_map_.at(kf_pt.first);
    for (const int pt_id : kf_pt.second)
    {
      vis_info->emplace_back(VisibilityInfo());
      getKfToPointLink(f, pt_id, &(vis_info->back()));
    }
  }
}

bool GlobalMap::traceStampedPoses(const std::string& trace_fn)
{
  std::ofstream trace;
  trace.open(trace_fn);
  trace.precision(15);
  if (!trace)
  {
    return false;
  }
  else
  {
    for (const auto& p : frame_id_to_kf_in_graph_map_)
    {
      if (p.second->bundleId() > last_updated_bundle_id_)
      {
        continue;
      }
      Transformation pose_imu = p.second->T_world_imu();
      trace << p.second->getTimestampSec() << " "
            << pose_imu.getPosition()(0, 0) << " "
            << pose_imu.getPosition()(1, 0) << " "
            << pose_imu.getPosition()(2, 0) << " " << pose_imu.getRotation().x()
            << " " << pose_imu.getRotation().y() << " "
            << pose_imu.getRotation().z() << " " << pose_imu.getRotation().w()
            << std::endl;
    }
  }
  trace.close();
  return true;
}

bool GlobalMap::traceExtrinsics(const std::string& trace_fn)
{
  std::ofstream trace;
  trace.open(trace_fn);
  trace.precision(15);
  if (!trace)
  {
    return false;
  }
  else
  {
    for (auto& v : cam_id_to_Tbc_in_graph_map_)
    {
      trace << v.first << std::endl;
      trace << v.second.getTransformationMatrix() << std::endl;
    }
  }
  trace.close();
  return true;
}
}
