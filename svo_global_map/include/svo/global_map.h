#pragma once

#include <map>
#include <set>
#include <atomic>

#include <boost/shared_ptr.hpp>

#include <rpg_common/pose.h>
#include <svo/common/imu_calibration.h>
#include <svo/common/frame.h>
#include <svo/common/occupancy_grid_2d.h>

namespace gtsam
{
namespace noiseModel
{
class Base;
}
using SharedNoiseModel = boost::shared_ptr<noiseModel::Base>;
class Values;
}

namespace svo
{
class GTSAMOptimizer;
class GTSAMOptimizerOptions;
class GraphManager;
class GraphManagerOptions;
class ImuCalibration;
class ImuInitialization;

using VisibilityInfo = Eigen::Matrix<float, 1, 6>;
using VisibilityInfoVec = rpg::Aligned<std::vector, VisibilityInfo>;
using UnitBearingVec = rpg::Aligned<std::vector, Eigen::Vector3d>;

using UniquePointIds = std::set<int>;
using UniqueFrameIds = std::set<int>;
using UniqueCamIds = std::set<size_t>;

using PtIdToFtrIdx = std::map<int, size_t>;
using PtIdToPos = rpg::aligned::Map<int, Eigen::Vector3d>;

using FrameIdPairs = std::map<int, int>;
using FrameIdToIMUMeas = std::map<int, ImuMeasurements>;
using PointIdPairs = std::map<int, int>;
using PointIdPairsVec = std::vector<std::pair<int, int>>;
using BundleIdToIMUPose = std::map<int, Transformation>;

struct GlobalMapOptions
{
  // general options
  // Use smart factors in GTSAM
  bool use_smart_factor = false;

  // Add IMU measurements to the graph
  bool use_imu = true;

  // Whether or not to use thread
  bool use_thread = true;

  // Only add 1 keyframe every N to the graph
  int only_add_every_n = -1;

  // Only start the optimization every N keyframes added to the graph
  int optimize_every_n = -1;

  // do we also optimize the extrinsics?
  bool refine_extrinsics = false;

  // When adding the features from a new frame, do we ignore the features
  // that fall close to the fixed landmarks matched in this frame.
  bool ignore_redundant_landmarks = false;
  // If the above option is true, what is the grid size used for filtering
  // out the feathres.
  int ignore_grid_size = 30;

  // reprojected points
  bool add_reprojected_fixed_landmark = false;
  size_t max_reprojected_fixed_landmark_each_frame = 20u;
  int fix_lm_reobs_threshold = -1;

  // Initialization: we wait until there are 'init_min_frame_before_opt' frames
  // and at least one of them has 'init_min_lm_obs_frame' well-observed
  // landmarks to start the initial bundle adjustment.
  size_t init_min_frame_before_opt = 10;
  size_t init_min_lm_obs_frame = 5u;

  // Cache point observations: a point is only added to the graph when
  // - has at least some observations
  size_t min_num_obs = 2u;
  // - the conditional number of the information matrix cannot be too large
  double pt_cond_thresh = 30.0;
  // - the estimated covariance cannot be too large
  double pt_sigma_thresh = 0.1;
  // - has to have some observations that have enough disparity
  double min_parallax_thresh_rad = 5.0 / 180 * M_PI;

  // If a point is added to the graph, whether to
  bool add_range_prior_to_good_pt = true;

  // Do we use edgelet?
  bool use_edgelet = false;
  // Do we use map point? If so, what are the minimum number of observations
  // for map points (should be higher than the above thershold).
  bool use_map_point = false;
  size_t mappoint_min_obs = 4u;
  // If a point still do not get enough observations after some frames added
  // we don't add this to the optimization problem and simply discard it
  int max_point_age = 20;

  // Relative constraint: useful when there are few common landmarks between
  // frames.
  // Whether to use the relative constraint
  bool use_relative_pose_fixation = true;
  // Add relative constraint when the number of the common landmarks is
  // less than a certain threshold
  size_t relative_min_common = 8u;
  // what is the strength of the relative pose constraint
  double relative_pos_sigma_meter = 0.1;
  double relative_rot_sigma_rad = 5.0 / 180.0 * M_PI;
  // If we have IMU, what is the minimum time needed for adding relative
  // constraint (IMU should already give good constraint)?
  double relative_prior_with_imu_min_time_sec = -1.0;
  // If after some time, there are enough common landmarks (from the cached
  // points)
  // do we remove the relative constraint? (may cause jittering in the
  // optimization problem)
  bool frame_fixation_release_check = true;
  // We only remove the relative constraint when there are enough common
  // landmarks
  size_t relative_release_min_common = 20u;

  // A keyframe is only set as a frame to query from, if it has more than
  // certain number of new 3D points compared to the previous added frame.
  // This is to avoid having too many frames with largely overlapped visible
  // landmarks.
  size_t query_frame_min_3d_points = 20u;

  // EXPERIMENTAL: stability (maybe slow)
  // If the position of a point changes over the below threshold after one
  // optimiation,
  // it is considered unstable.
  double point_stable_thresh_meter = -1.0;
  // Only incrementa the point value version when a certain ratio of points
  // are stable.
  double inc_version_stable_ratio = -1.0;

  // debug
  // Fix all poses as the initial values
  bool debug_fix_all_poses = false;
  // whether to print to stdcout
  bool print_stdcout = true;
  // very detailed output
  bool verbose = false;
};

// This struct keeps track of added observations, and determines
// whether a point is good enough to be added to the graph
struct CachedPointObservations
{
  int age_ = 0;
  PointPtr point_ = nullptr;
  std::vector<FramePtr> ob_frames_;
  std::vector<size_t> ftr_indices_;
  UnitBearingVec obs_unit_bearing_from_pt_;
  Eigen::Matrix3d fim_;
  FeatureType type_;

  double cur_cond_num_ = -1;
  double cur_mean_det_ = -1;
  bool is_parallax_good_ = false;

  // options
  static double min_parallax_rad;
  static size_t min_num_obs;
  static size_t min_map_point_num_obs;
  static double cond_thresh;
  static double fim_det_cbr_minimum;

  CachedPointObservations(const CachedPointObservations&) = delete;
  CachedPointObservations()
  {
    fim_.setZero();
  }

  void addObservation(const FramePtr& f, const size_t ftr_idx);
  void setAllObservationsToOutlier();
  inline bool hasEnoughObs() const
  {
    return isMapPoint(type_) ? ob_frames_.size() >= min_map_point_num_obs :
                               ob_frames_.size() >= min_num_obs;
  }
  inline bool isPointGood() const
  {
    return hasEnoughObs() && is_parallax_good_ && point_->nRefs() == 0 &&
           (cond_thresh < 0 || cur_cond_num_ < cond_thresh) &&
           cur_mean_det_ > fim_det_cbr_minimum;
  }
};

// A global map constructed from past keyframes:
//   - maintain consistency
//   - allow query of overlap keyframes/visible landmarks for reprojection
class GlobalMap
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GlobalMap() = delete;
  GlobalMap(const GlobalMapOptions& options,
            const GraphManagerOptions& graph_manager_options,
            const GTSAMOptimizerOptions& gtsam_opt_options,
            const CameraBundlePtr& ncams);
  ~GlobalMap()
  {
    finishAndQuit();
  }
  GlobalMapOptions options_;

  void initializeIMUParams(const ImuCalibration& imu_calib,
                           const ImuInitialization& imu_init);

  inline size_t numFramesInMap() const
  {
    return frame_id_to_kf_in_graph_map_.size();
  }

  void addKeyframe(const FramePtr& new_frame);

  void reset()
  {
    frame_id_to_kf_in_graph_map_.clear();
    frame_id_to_unoptimized_pose_map_.clear();
    pt_id_to_lm_in_graph_map_.clear();
    cam_id_to_Tbc_in_graph_map_.clear();

    frame_id_to_vis_pt_ids_in_graph_map_.clear();
    frame_id_to_vis_ftr_idx_in_graph_map_.clear();
    frame_ids_to_query_from_.clear();

    pt_id_to_cached_pt_obs_map_.clear();
    bundle_id_to_frame_id_map_.clear();

    fixed_frame_ids_other_than_intial_.clear();
    fixed_pts_all_.clear();

    fixed_relative_frame_id_pairs_.clear();

    from_id_to_imu_meas_map_.clear();
    imu_meas_since_cur_start_kf_.clear();
    imu_cur_start_kf_id_ = -1;
    imu_constraints_frame_id_pairs_.clear();
    frame_id_with_inertial_states_.clear();

    n_matched_points_in_graph_ = 0u;
    frame_id_to_matched_pts_id_.clear();
    n_reobserved_landmarks_in_graph_ = 0u;
    frame_id_to_reobs_pts_ids_.clear();
    n_reobserved_landmarks_in_graph_frontend_ = 0u;
    frame_id_to_reobs_pts_ids_frontend_.clear();
    matched_new_frame_ids_.clear();
    lm_id_to_num_reobs_map_.clear();
    fixed_landmarks_in_graph_.clear();

    prev_added_frame_id_ = -1;
    newly_added_frame_id_ = -1;

    last_updated_bundle_id_ = -1;
    last_published_bundle_id_ = -1;
    updated_point_ids_.clear();
    stable_point_ids_.clear();

    n_since_last_optimization_ = 0;
    has_unoptimized_point_matches_ = false;
  }

  // Get keyframe positions: all the keyframes and the keyframes used for query
  void getAllKFPositions(rpg::PositionVec* kf_pos,
                         rpg::PositionVec* kf_pos_with_priors) const;
  void getQueryKFPositions(rpg::PositionVec* query_kf_pos,
                           rpg::PositionVec* last_queried_kf_pos) const;
  // Get point positions
  void getAllPoints(rpg::PositionVec* points,
                    rpg::PositionVec* points_with_priors,
                    rpg::PositionVec* unstable_points) const;
  void getPointPositionAndIds(rpg::PositionVec* points,
                              std::vector<int>* ids) const;

  // Get lines representing the visibility constraint
  // All keyframes or last N keyframes
  void getVizLinks(VisibilityInfoVec* vis_info, const int last_n = -1) const;
  // The visiblity lines for keypoints (used for determining overlapping views)
  void getKeypointsVisLinks(VisibilityInfoVec* vis_info) const;
  // Matched points: where the two points have different IDs, but the same
  // position
  inline void getMatchedPointsVisLinks(VisibilityInfoVec* vis_info) const
  {
    this->framePointsToVisLink(frame_id_to_matched_pts_id_, vis_info);
  }
  // Matched points for reobserved landmarks in the bundle adjustment
  inline void getReobservedPointsVisLinks(VisibilityInfoVec* vis_info) const
  {
    this->framePointsToVisLink(frame_id_to_reobs_pts_ids_, vis_info);
  }

  // Reobserved landmarks: points in the global map that are successfully
  // projected into the new frame
  inline void
  getReobservedPointsVisLinksFrontend(VisibilityInfoVec* vis_info) const
  {
    this->framePointsToVisLink(frame_id_to_reobs_pts_ids_frontend_, vis_info);
  }

  // Functions to gather IMU measurements
  // Accumulate IMU measurements to be added for the next keyframe
  inline void accumulateIMUMeasurements(const ImuMeasurements& imu_meas)
  {
    if (imu_cur_start_kf_id_ == -1)
    {
      return;
    }
    if (imu_meas_since_cur_start_kf_.front().timestamp_ > imu_meas.back().timestamp_)
    {
      return;
    }
    //CHECK_LT(imu_meas_since_cur_start_kf_.front().timestamp_,
             //imu_meas.back().timestamp_);
    imu_meas_since_cur_start_kf_.insert(imu_meas_since_cur_start_kf_.begin(),
                                        imu_meas.begin(), imu_meas.end());
  }
  // Start new accumulation: called when a new keyframe is added
  inline void startNewAccumulation(const int new_start_kf_id)
  {
    if (imu_cur_start_kf_id_ != -1)
    {
      from_id_to_imu_meas_map_[imu_cur_start_kf_id_] =
          imu_meas_since_cur_start_kf_;
    }
    imu_cur_start_kf_id_ = new_start_kf_id;
    imu_meas_since_cur_start_kf_.clear();
  }

  // Add matching points, which will be added to the graph next time
  inline void addMatchingPointInfo(const MatchedPointsInfo& match_info)
  {
    if (match_info.lc_kf_id_ > 0 && match_info.cur_kf_id_ > 0)
    {
      match_points_info_to_add_.push_front(match_info);
      matched_new_frame_ids_.insert(match_info.cur_kf_id_);
    }

    constexpr size_t kMaxN = 10u;
    if (match_points_info_to_add_.size() > kMaxN)
    {
      match_points_info_to_add_.erase(match_points_info_to_add_.begin() + kMaxN,
                                      match_points_info_to_add_.end());
    }
  }

  // Utility functions to check the status of the graph
  inline bool pointInGraph(const int id)
  {
    return pt_id_to_lm_in_graph_map_.find(id) !=
           pt_id_to_lm_in_graph_map_.end();
  }

  inline bool frameInGraph(const int frame_id)
  {
    return frame_id_to_kf_in_graph_map_.find(frame_id) !=
           frame_id_to_kf_in_graph_map_.end();
  }

  inline bool hasNewEstimate() const
  {
    return last_published_bundle_id_ < last_updated_bundle_id_;
  }

  inline void setResultsPublished() const
  {
    last_published_bundle_id_ = last_updated_bundle_id_.load();
  }

  inline bool needToOptimize() const
  {
    if (options_.optimize_every_n < 0 ||
        n_since_last_optimization_ >=
            static_cast<size_t>(options_.optimize_every_n) ||
        has_unoptimized_point_matches_)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  inline bool isPointOptimized(const int id) const
  {
    return updated_point_ids_.find(id) != updated_point_ids_.end();
  }

  inline bool isPointStable(const int id) const
  {
    return stable_point_ids_.find(id) != stable_point_ids_.end();
  }

  inline bool isAllOptimizationDone() const
  {
    if (newly_added_frame_id_ == -1)
    {
      return true;
    }
    auto it = frame_id_to_kf_in_graph_map_.find(newly_added_frame_id_);
    if (it == frame_id_to_kf_in_graph_map_.end())
    {
      return false;
    }
    else
    {
      return last_updated_bundle_id_ ==
             frame_id_to_kf_in_graph_map_.at(newly_added_frame_id_)->bundleId();
    }
  }

  inline bool hasInitialBA() const
  {
    return initial_ba_fixed_;
  }

  // Get the bundle id -> pose map
  void getBundleIDIMUPosesMap(BundleIdToIMUPose* pose_map) const;


  // Get maximum N overlapped keyframes
  void getOverlapKeyframesMaxN(const Frame& query_f, const int max,
                               std::vector<FramePtr>* overlap_kfs);


  // Finish all optimization
  void quitOptimizerThread();
  void finishAndQuit()
  {
    while (!this->isAllOptimizationDone())
    {
      this->optimize();
      this->updateStates();
    }
    this->quitOptimizerThread();
  }

  // Logging
  bool traceStampedPoses(const std::string& trace_fn);
  bool traceExtrinsics(const std::string& trace_fn);

  // Toggle whether to add reobserved landamrks (fixed landmarks in graph)
  // E.g., should be disabled when tracking is lost for a long time without
  // successful reinitialization
  inline void setEnableReobserved(const bool enabled)
  {
    options_.add_reprojected_fixed_landmark = enabled;
  }

  // Misc
  inline size_t numMatchesToAdd()
  {
    size_t n = 0;
    for (const auto& v : match_points_info_to_add_)
    {
      n += v.pt_id_matches_.size();
    }

    return n;
  }


private:
  // actual optimization
  void optimize();
  // Update the states in the global map from GTSAM
  void updateStates();
  void updateStatesWithEstimate(const gtsam::Values& estimate,
                                const BundleId last_optimized_state_index);

  bool getPointFromGraphByIdUnsafe(const int pt_id,
                                   const gtsam::Values& estimate,
                                   Eigen::Vector3d* pt) const;
  void getIMUMeasurementsBetween(const int start_f_id, const int end_f_id,
                                 const bool remove,
                                 ImuMeasurements* imu_meas_from_start);

  // Functions for adding visual observations
  void addVisualMeasurementsToGraph(const FramePtr& new_frame);
  void addCachedPointAndObservations(const CachedPointObservations& cached);
  void addLandmark(svo::Point& pt, const Frame& f);
  void addObservationToLandmark(const FramePtr& f, const size_t ftr_idx);
  bool addObservationToLandmarkByPointId(const FramePtr& f, const int point_id);
  void addReobservedLandmarkToGraph(const FramePtr& new_frame);
  void updateFixedLandmarks();
  bool isFeatureGoodToAdd(const Frame& f, const size_t ftr_idx);

  // Adding matching points
  void addMatchingPointsToGraph();

  // Add extrinsics
  void addTbc(svo::Transformation& Tbc, const size_t cam_id);
  inline bool isTbcAdded(const size_t cam_id)
  {
    return cam_id_to_Tbc_in_graph_map_.find(cam_id) !=
           cam_id_to_Tbc_in_graph_map_.end();
  }

  // Check whether we are good for the initial BA
  bool checkAndFixInitialBA();

  // Update different priors (see below) based on the current status
  void updatePriors();

  // Adding priors
  // - Pose fixation
  void fixFramePoseUnsafe(
      const FramePtr& f, const bool bookkeeping,
      const gtsam::SharedNoiseModel& noise = gtsam::SharedNoiseModel());
  // - Relative Pose
  bool checkAndFixRelativePose(const int prev_id, const int cur_id);
  void fixRelativePoseUnsafe(
      const FramePtr& f1, const FramePtr& f2,
      const gtsam::SharedNoiseModel& noise = gtsam::SharedNoiseModel());
  void checkAndRemoveRelativePosePriors();
  // - Range priors
  void addRangePriorUnsafe(const Point& pt, const Frame& f,
                           const double ratio_dist);

  // Bookkeeping functions
  // - visible points to frame
  inline void registerPointVisibleToFrame(const int frame_id, const int pt_id,
                                          const size_t ftr_idx)
  {
    auto f_to_p_it = frame_id_to_vis_pt_ids_in_graph_map_.find(frame_id);
    CHECK(f_to_p_it != frame_id_to_vis_pt_ids_in_graph_map_.end());
    UniquePointIds& cur_vis_points = f_to_p_it->second;
    CHECK(cur_vis_points.find(pt_id) == cur_vis_points.end());
    cur_vis_points.insert(pt_id);

    auto p_to_f_it = point_id_to_vis_frame_ids_in_graph_map_.find(pt_id);
    CHECK(p_to_f_it != point_id_to_vis_frame_ids_in_graph_map_.end());
    UniqueFrameIds& cur_vis_frames = p_to_f_it->second;
    CHECK(cur_vis_frames.find(frame_id) == cur_vis_frames.end());
    cur_vis_frames.insert(frame_id);

    auto f_to_ftr_idx_it = frame_id_to_vis_ftr_idx_in_graph_map_.find(frame_id);
    CHECK(f_to_ftr_idx_it != frame_id_to_vis_ftr_idx_in_graph_map_.end());
    PtIdToFtrIdx& cur_ftr_indices = f_to_ftr_idx_it->second;
    CHECK(cur_ftr_indices.find(pt_id) == cur_ftr_indices.end());
    cur_ftr_indices[pt_id] = (ftr_idx);
  }
  // - reobserved landmarks
  inline void registerReobservedLandmarks(const int pt_id)
  {
    if (fixed_landmarks_in_graph_.find(pt_id) !=
        fixed_landmarks_in_graph_.end())
    {
      return;
    }

    auto it = lm_id_to_num_reobs_map_.find(pt_id);
    if (it != lm_id_to_num_reobs_map_.end())
    {
      it->second++;
    }
    else
    {
      lm_id_to_num_reobs_map_.emplace(std::make_pair(pt_id, 0u));
    }

    return;
  }
  // - point fixation prior
  inline void registerPointFixed(const int pt_id)
  {
    CHECK(fixed_pts_all_.find(pt_id) == fixed_pts_all_.end());
    fixed_pts_all_.insert(pt_id);
  }
  // - general function to bookkeeping points ids that are related to a frame
  inline void registerFrameRelatedPoints(const int frame_id, const int pt_id,
                                         std::map<int, UniquePointIds>* d)
  {
    auto it = d->find(frame_id);
    if (it == d->end())
    {
      (*d)[frame_id] = UniquePointIds{ pt_id };
    }
    else
    {
      it->second.insert(pt_id);
    }
  }

  // utility functions
  void framePointsToVisLink(const std::map<int, UniquePointIds>& d,
                            VisibilityInfoVec* vis) const;
  inline void getKfToPointLink(const FramePtr& f, const int pt_id,
                               VisibilityInfo* v) const
  {
    const Eigen::Vector3d& kf_pos = f->T_world_imu().getPosition();
    std::map<int, UniquePointIds>::const_iterator vis_cit =
        frame_id_to_vis_pt_ids_in_graph_map_.find(f->id());
    std::map<int, PointPtr>::const_iterator pt_cit =
        pt_id_to_lm_in_graph_map_.find(pt_id);
    CHECK(pt_cit != pt_id_to_lm_in_graph_map_.cend());
    (*v)(0, 0) = static_cast<float>(kf_pos.x());
    (*v)(0, 1) = static_cast<float>(kf_pos.y());
    (*v)(0, 2) = static_cast<float>(kf_pos.z());
    (*v)(0, 3) = static_cast<float>(pt_cit->second->pos().x());
    (*v)(0, 4) = static_cast<float>(pt_cit->second->pos().y());
    (*v)(0, 5) = static_cast<float>(pt_cit->second->pos().z());
  }

  inline int ftrIdx(const int pt_id, const int f_id) const
  {
    auto fit = frame_id_to_vis_ftr_idx_in_graph_map_.find(f_id);
    CHECK(fit != frame_id_to_vis_ftr_idx_in_graph_map_.end());
    auto vit = fit->second.find(pt_id);
    CHECK(vit != fit->second.end());
    return vit->second;
  }

  inline void checkFrameFromGlobalMap(const FramePtr& f)
  {
    for (size_t ftr_idx = 0; ftr_idx < f->numFeatures(); ftr_idx++)
    {
      const FeatureType& type = f->type_vec_[ftr_idx];
      CHECK(isFixedLandmark(type) || type == FeatureType::kOutlier)
          << str(type) << std::endl;
      if (isFixedLandmark(type))
      {
        CHECK(f->landmark_vec_[ftr_idx]);
      }
    }
  }

  inline size_t numCommonPoints(const UniquePointIds& set1,
                                const UniquePointIds& set2)
  {
    size_t n_common = 0;
    for (const int id : set1)
    {
      if (set2.find(id) != set2.end())
      {
        n_common++;
      }
    }
    return n_common;
  }

  inline bool pointVisibleInFrameInGraph(const int pt_id, const int frame_id)
  {
    auto f_it = frame_id_to_kf_in_graph_map_.find(frame_id);
    if (f_it == frame_id_to_kf_in_graph_map_.end())
    {
      return false;
    }
    auto f_vis_it = frame_id_to_vis_pt_ids_in_graph_map_.find(frame_id);
    CHECK(f_vis_it != frame_id_to_vis_pt_ids_in_graph_map_.end());

    return f_vis_it->second.find(pt_id) != f_vis_it->second.end();
  }

  inline bool hasFutureObservations(const Frame& f, const size_t ftr_idx) const
  {
    const PointPtr& pt = f.landmark_vec_[ftr_idx];
    CHECK(pt);

    int n_future_obs = std::count_if(
        pt->obs_.begin(), pt->obs_.end(),
        [&](const KeypointIdentifier& v) { return v.frame_id > f.id(); });
    return (n_future_obs > 0);
  }

  // output
  std::ostream& out_;

  // stuff that is already added to the graph
  // The values should be updated every time we finish an optimization
  // The keyframe states
  std::map<int, FramePtr> frame_id_to_kf_in_graph_map_;
  // The initial pose when a frame is added to the graph, useful to add
  // relative constraints using the original values
  std::map<int, Transformation> frame_id_to_unoptimized_pose_map_;
  // Point (with sufficient observations) that are added to the grpah
  std::map<int, PointPtr> pt_id_to_lm_in_graph_map_;
  // Extrinsics
  std::map<size_t, Transformation> cam_id_to_Tbc_in_graph_map_;

  // Buffer of points and pose before adding to the graph
  std::map<int, CachedPointObservations> pt_id_to_cached_pt_obs_map_;

  // visibility map
  // frame -> visible point ids
  std::map<int, UniquePointIds> frame_id_to_vis_pt_ids_in_graph_map_;
  // frame -> visible point feature indices in the frame
  std::map<int, PtIdToFtrIdx> frame_id_to_vis_ftr_idx_in_graph_map_;
  // point -> visible frame ids
  std::map<int, UniqueFrameIds> point_id_to_vis_frame_ids_in_graph_map_;

  // IMU related
  // Variables used to accumulate IMU measurements
  FrameIdToIMUMeas from_id_to_imu_meas_map_;
  ImuMeasurements imu_meas_since_cur_start_kf_;
  int imu_cur_start_kf_id_ = -1;
  // bookkeeping frames with IMU states and constraints
  FrameIdPairs imu_constraints_frame_id_pairs_;
  UniqueFrameIds frame_id_with_inertial_states_;

  // Frames to query from
  UniqueFrameIds frame_ids_to_query_from_;

  // Matched points and observed landmarks
  // Matched points that need to be added next
  std::deque<MatchedPointsInfo> match_points_info_to_add_;
  // From loop closing:
  // - normal feature in the frontend
  size_t n_matched_points_in_graph_ = 0u;
  std::map<int, UniquePointIds> frame_id_to_matched_pts_id_;
  // - from the global map
  size_t n_reobserved_landmarks_in_graph_ = 0u;
  std::map<int, UniquePointIds> frame_id_to_reobs_pts_ids_;
  // - matched frames from loop closing
  UniqueFrameIds matched_new_frame_ids_;
  // From the frontend: by reprojecting the landmarks
  size_t n_reobserved_landmarks_in_graph_frontend_ = 0u;
  std::map<int, UniquePointIds> frame_id_to_reobs_pts_ids_frontend_;

  // Bookkeeping fixed landmarks:
  // Point ids in
  UniquePointIds fixed_landmarks_in_graph_;
  // Grid used for filtering out redundant points
  OccGrid2DPtr reprojected_landmark_grid_ = nullptr;
  // Number of observations
  std::map<int, size_t> lm_id_to_num_reobs_map_;

  // book keeping on a frame basis: change every frame
  bool skip_fixed_lm_from_frontend_ = false;
  int prev_added_frame_id_ = -1;
  int newly_added_frame_id_ = -1;

  // book keeping for update states from estimate and publish
  // useful to avoid redundant update/publishing
  mutable std::atomic_int last_updated_bundle_id_{ -1 };
  mutable std::atomic_int last_published_bundle_id_{ -1 };
  size_t n_since_last_optimization_ = 0;
  bool has_unoptimized_point_matches_ = false;

  // Last update: what are the points that are updated/statble
  UniquePointIds updated_point_ids_;
  UniquePointIds stable_point_ids_;

  // queried keyframes last time
  UniqueFrameIds last_queried_kf_ids_;

  // needed to get relative constraints (extrinsics) for multiple cameras
  // Not used for now
  CameraBundlePtr ncams_;
  std::map<BundleId, UniqueFrameIds> bundle_id_to_frame_id_map_;

  // graph and optimizer
  std::shared_ptr<GTSAMOptimizer> optimizer_;
  std::shared_ptr<GraphManager> graph_;
  gtsam::SharedNoiseModel relative_pose_prior_;

  // Fixation bookkeeping
  // - have initial BA
  bool initial_ba_fixed_ = false;
  // - have other frames fixed other than the initial BA
  UniqueFrameIds fixed_frame_ids_other_than_intial_;
  // - have points that have fixations
  UniquePointIds fixed_pts_all_;
  // - frames with relative constraint
  FrameIdPairs fixed_relative_frame_id_pairs_;
};
using GlobalMapPtr = std::shared_ptr<GlobalMap>;
}
