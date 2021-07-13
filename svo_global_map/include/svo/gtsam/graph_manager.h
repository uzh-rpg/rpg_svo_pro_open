#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <svo/gtsam/smart_factors_fwd.h>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/navigation/CombinedImuFactor.h>

namespace svo
{
// fwd
struct ImuCalibration;
struct ImuInitialization;
struct ImuMeasurement;
typedef std::deque<ImuMeasurement, Eigen::aligned_allocator<ImuMeasurement>>
    ImuMeasurements;
class Point;
class Frame;
using GTSAMSlotIndices = gtsam::FastVector<size_t>;

struct GraphManagerOptions
{
  // noise of visual measurements in pixel
  double reproj_error_ns_px = 1.0;
  double smart_reproj_outlier_thresh_px = 3.0;
  bool use_robust_px_noise = true;

  // this avoids chierality check
  bool use_bearing_factor = true;

  // prior for visual-inertial case
  double init_pos_sigma_meter = 0.001;
  double init_roll_pitch_sigma_rad = 45.0 / 180.0 * M_PI;
  double init_yaw_sigma_rad = 1.0 / 180.0 * M_PI;

  // default general pose prior
  double position_prior_sigma_meter = 0.001;
  double rotation_prior_sigma_rad = 1.0 / 180 * M_PI;
  double point_prior_sigma_meter = 0.001;
};

struct SmartFactorStatistics
{
  size_t n_valid = 0;
  size_t n_degenerate = 0;
  size_t n_behind_cam = 0;
  size_t n_outlier = 0;
  size_t n_farpoint = 0;
  void print() const;
};

class GraphManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using NFramesId = int;
  using PointId = int;
  using Ptr = std::shared_ptr<GraphManager>;
  using PreintegratedImuMeasurements = gtsam::PreintegratedCombinedMeasurements;
  using CombinedPreintegratedMeasurementPtr =
      std::shared_ptr<PreintegratedImuMeasurements>;
  using PointIdSmartFactorMap =
      std::unordered_map<int, boost::shared_ptr<SmartFactor>>;
  using IdToFactorIdxMap = std::map<int, std::vector<size_t>>;
  using IdToSlotIdMap = std::map<int, std::vector<int>>;

  GraphManager(const GraphManagerOptions& options);
  ~GraphManager();

  GraphManagerOptions options_;
  gtsam::Cal3_S2::shared_ptr cam_calib_;
  boost::shared_ptr<PreintegratedImuMeasurements::Params>
      preintegration_params_;
  std::vector<gtsam::SharedNoiseModel> uplane_noise_pyr_;
  gtsam::SharedNoiseModel smart_noise_;
  gtsam::SharedNoiseModel imu_bias_prior_noise_;
  gtsam::SharedNoiseModel velocity_prior_noise_;
  gtsam::SharedNoiseModel zero_velocity_prior_noise_;
  gtsam::SharedNoiseModel point_match_noise_;
  Eigen::Matrix<double, 6, 6> init_pose_prior_visual_inertial_;
  std::unique_ptr<gtsam::SmartProjectionParams> smart_factor_params_;

  // general prior noise
  gtsam::SharedNoiseModel default_pose_prior_noise_;
  gtsam::SharedNoiseModel default_point_prior_noise_;

  /// Updates to be considered for next optimization iteration. MUTEX PROTECTED
  /// @{

  std::mutex graph_mut_;
  gtsam::NonlinearFactorGraph new_factors_;  // new factors to be added
  gtsam::Values new_values_;                 // new states to be added

  // smart factors
  PointIdSmartFactorMap point_id_to_new_smart_factor_map_;  // pointId ->
                                                            // {SmartFactorPtr}
  SmartFactorInfoMap point_id_to_smart_factor_info_map_;    // pointId ->
                                                            // {SmartFactorPtr,
                                                            // SlotIndex}
  SmartFactorStatistics smart_factor_stats_;

  // general book keeping
  BundleId last_added_state_index_ = -1;
  GTSAMSlotIndices factor_slots_to_delete_;

  /// @}

  // bookkeeping
  // for prior factors
  // absolute pose prior
  IdToFactorIdxMap bundle_id_to_pose_prior_factor_id_map_;
  IdToSlotIdMap bundle_id_to_prior_slot_id_map_;
  // absolute point prior
  IdToFactorIdxMap point_id_to_point_prior_factor_id_map_;
  IdToSlotIdMap point_id_to_point_prior_slot_id_map_;
  // range point prior
  IdToFactorIdxMap point_id_to_range_prior_factor_id_map_;
  IdToSlotIdMap point_id_to_range_prior_slot_id_map_;
  // relative pose prior
  IdToFactorIdxMap first_bundle_id_to_relative_prior_factor_id_map_;
  IdToSlotIdMap first_bundle_id_to_prior_slot_id_map_;
  std::map<int, int> first_bundle_id_to_second_bundle_id_map_;

  void initialize(const double uplane_to_px_multiplier);

  void initializeImuNoiseModels(const ImuCalibration& imu_calib,
                                const ImuInitialization& imu_init);

  void reset();

  // Landmarks and observations
  void addLandmark(const svo::Point& point);

  void addExtrinsics(const svo::Transformation& T_b_c,
                     const size_t cam_id);

  void addObservationToLandmark(const svo::Frame& frame,
                                const size_t keypoint_index,
                                const bool refine_extrinsics);

  // smart factors
  void addSmartLandmark(const svo::Point& point,
                        const Transformation& T_imu_cam);

  void addObservationToSmartLandmark(const svo::Frame& frame,
                                     const size_t keypoint_index);

  void addCombinedImuFactor(const NFramesId& from_id, const NFramesId& to_id,
                            const Eigen::Vector3d& acc_bias,
                            const Eigen::Vector3d& gyr_bias,
                            const double to_timestamp_sec,
                            const ImuMeasurements& imu_measurements);

  // Pose priors
  void addInitalPosePriorVI(const NFramesId& state_index,
                            const Transformation& T_W_B);

  void addPosePriorFactor(const NFramesId& state_index,
                          const Transformation& T_W_B, const double rot_sig_rad,
                          const double pos_sig_meter);

  void addPosePriorFactor(const NFramesId& state_index,
                          const Transformation& T_W_B,
                          const gtsam::SharedNoiseModel& noise_model);

  void removePosePriorFactor(const NFramesId& state_index);

  // Point priors
  void addPointPriorFactor(const PointId& state_index,
                           const Eigen::Vector3d& pw, const double pos_sig);

  void addPointPriorFactor(const PointId& state_index,
                           const Eigen::Vector3d& pw,
                           const gtsam::SharedNoiseModel& noise_model);

  void removePointPriorFactor(const PointId& point_id);

  // range priors
  void addRangePriorFactor(const NFramesId& state_index,
                           const PointId& point_id, const double dist,
                           const Transformation& T_b_c,
                           const gtsam::SharedNoiseModel& noise);
  void removeRangePriorFactor(const PointId& point_id);

  // Relative pose priors
  void addRelativePosePriorFactor(const NFramesId& bundle_id_1,
                                  const NFramesId& bundle_id_2,
                                  const Transformation& T_p1_p2,
                                  const gtsam::SharedNoiseModel& noise);
  void removeRelativePosePriorFactor(const NFramesId& bundle_id_1,
                                     const NFramesId& bundle_id_2);

  // Inertial related
  void addBiasPriorFactor(const NFramesId& state_index,
                          const Eigen::Vector3d& acc_bias,
                          const Eigen::Vector3d& gyro_bias);

  void addVelocityPriorFactor(const NFramesId& state_index,
                              const Eigen::Vector3d& W_v);

  void addZeroVelocityPriorFactor(const NFramesId& state_index);

  // Augment State
  void augmentStateWithPose(const NFramesId& state_index,
                            const Transformation& T_W_B);

  void augmentStateWithVelocityAndBias(const NFramesId& state_index,
                                       const Eigen::Vector3d& W_v,
                                       const Eigen::Vector3d& acc_bias,
                                       const Eigen::Vector3d& gyr_bias);
  // Point matches
  void addPointMatchFactor(const PointId first, const PointId second);

  // register factor index to get slot information later
  void registerNewFactor(const int id, const size_t factor_idx,
                         IdToFactorIdxMap* id_to_factor_idx_map,
                         IdToSlotIdMap* id_to_slot_id_map);

  void deleteRegisteredFactor(const int id,
                              IdToFactorIdxMap* id_to_factor_idx_map,
                              IdToSlotIdMap*);

  // Retrieval (Always lock mut_ when using)
  void getUpdatesCopy(gtsam::NonlinearFactorGraph* graph_updates,
                      gtsam::Values* value_updates,
                      GTSAMSlotIndices* delete_indices,
                      std::vector<int>* smart_factor_point_ids);

  // Note: this function relies on the fact that the smart factors are passed
  // first
  void updateSlotInfo(const gtsam::FastVector<size_t>& new_slots,
                      const std::vector<int>& smart_factor_point_ids);
  void updateSlotInfoSingle(const gtsam::FastVector<size_t>& new_slots,
                            const IdToFactorIdxMap& id_to_factor_idx_map,
                            IdToSlotIdMap* id_to_slot_id_map);

  inline const SmartFactorPtr getSmartFacotrConst(const PointId pt_id)
  {
    auto it = point_id_to_smart_factor_info_map_.find(pt_id);
    CHECK(it != point_id_to_smart_factor_info_map_.end());
    if (it->second.slot_in_graph_ == -1)
    {
      return SmartFactorPtr();
    }
    else
    {
      return it->second.factor_;
    }
  }

  inline bool isPointIDSmartLandmark(const PointId pid)
  {
    return point_id_to_smart_factor_info_map_.find(pid) !=
           point_id_to_smart_factor_info_map_.end();
  }

  // misc
  void updateSmartFactorStatistics();
};

}  // namespace svo
