#include "svo/gtsam/graph_manager.h"
#include "svo/gtsam/camera_bearing_factor.h"
#include "svo/gtsam/camera_bearing_extrinsics_factor.h"

#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/imu_calibration.h>
#include <vikit/math_utils.h>

// boost
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

// gtsam
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/sam/RangeFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam_unstable/slam/PoseBetweenFactor.h>

namespace svo
{
void SmartFactorStatistics::print() const
{
  std::cout << "Smart factor statistics:\n"
            << "- n_valid: " << n_valid << std::endl
            << "- n_degenerate: " << n_degenerate << std::endl
            << "- n_behind_cam: " << n_behind_cam << std::endl
            << "- n_farpoint: " << n_farpoint << std::endl
            << "- n_outlier: " << n_outlier << std::endl;
}

GraphManager::GraphManager(const GraphManagerOptions& options)
  : options_(options)
{
}

GraphManager::~GraphManager()
{
}

void GraphManager::initialize(const double uplane_to_px_multiplier)
{
  // unit plane noise properties
  double reproj_error_noise_uplane_ =
      options_.reproj_error_ns_px / uplane_to_px_multiplier;
  double smart_reproj_outlier_thresh_uplane =
      options_.smart_reproj_outlier_thresh_px / uplane_to_px_multiplier;

  // camera
  cam_calib_.reset(new gtsam::Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0));

  // reprojection error noise models
  const size_t kMaxPyrLevels = 6;
  const double kRobustKurtosis = 3.0;
  uplane_noise_pyr_.resize(kMaxPyrLevels);
  const size_t noise_dim = options_.use_bearing_factor ? 3 : 2;
  for (size_t level = 0u; level < kMaxPyrLevels; ++level)
  {
    const double cur_noise = reproj_error_noise_uplane_ * (1 << level);
    if (options_.use_robust_px_noise)
    {
      uplane_noise_pyr_[level] = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(kRobustKurtosis),
          gtsam::noiseModel::Isotropic::Sigma(noise_dim, cur_noise));
    }
    else
    {
      uplane_noise_pyr_[level] =
          gtsam::noiseModel::Isotropic::Sigma(noise_dim, cur_noise);
    }
  }

  smart_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2, reproj_error_noise_uplane_);
  smart_factor_params_.reset(new gtsam::SmartProjectionParams(
      gtsam::HESSIAN,
      gtsam::IGNORE_DEGENERACY,  // creates rotation-only constraint
      true, true));

  // TODO: retriangulation threshold is a const variable and cant be set in
  // constructor.
  smart_factor_params_->setRankTolerance(1e-3);
  //  smart_factor_params_->setEnableEPI(false); // nonlinear refinement after
  //  triangulation
  //  smart_factor_params_->setLandmarkDistanceThreshold(100);
  smart_factor_params_->setDynamicOutlierRejectionThreshold(
      smart_reproj_outlier_thresh_uplane);

  // initial pose priors
  // rotation
  init_pose_prior_visual_inertial_.setZero();
  init_pose_prior_visual_inertial_.diagonal()[0] =
      options_.init_roll_pitch_sigma_rad * options_.init_roll_pitch_sigma_rad;
  init_pose_prior_visual_inertial_.diagonal()[1] =
      init_pose_prior_visual_inertial_.diagonal()[0];
  init_pose_prior_visual_inertial_.diagonal()[2] =
      options_.init_yaw_sigma_rad * options_.init_yaw_sigma_rad;
  // position
  init_pose_prior_visual_inertial_.diagonal()[3] =
      options_.init_pos_sigma_meter * options_.init_pos_sigma_meter;
  init_pose_prior_visual_inertial_.diagonal()[4] =
      init_pose_prior_visual_inertial_.diagonal()[3];
  init_pose_prior_visual_inertial_.diagonal()[5] =
      init_pose_prior_visual_inertial_.diagonal()[3];

  // general priors
  default_point_prior_noise_ =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(3)
           << gtsam::Vector3::Constant(options_.point_prior_sigma_meter))
              .finished());
  default_pose_prior_noise_ =
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6)
               << gtsam::Vector3::Constant(options_.rotation_prior_sigma_rad),
           gtsam::Vector3::Constant(options_.position_prior_sigma_meter))
              .finished());

  // point matches
  point_match_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e-6);

  std::cout << "\nGraphManager: Initialized noise for default fixation:\n";
  std::cout << "Point:\n";
  default_point_prior_noise_->print();
  std::cout << "Pose:\n";
  default_pose_prior_noise_->print();
  std::cout << "Visual Inertial inital pose:\n";
  std::cout << init_pose_prior_visual_inertial_ << std::endl;
}

void GraphManager::initializeImuNoiseModels(const ImuCalibration& imu_calib,
                                            const ImuInitialization& imu_init)
{
  // prior bias
  Eigen::Matrix<double, 6, 1> prior_bias_sigmas;
  prior_bias_sigmas.head<3>().setConstant(imu_init.acc_bias_sigma);
  prior_bias_sigmas.tail<3>().setConstant(imu_init.omega_bias_sigma);
  imu_bias_prior_noise_ =
      gtsam::noiseModel::Diagonal::Sigmas(prior_bias_sigmas);

  // velocity prior
  velocity_prior_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(3, imu_init.velocity_sigma);

  // Create the constant zero velocity noise model.
  static const double kZeroVelocitySigma = 1.0e-3;  // [m/s]
  zero_velocity_prior_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(3, kZeroVelocitySigma);

  // Calculate the measurement and integration noise covariances.
  // Note that the covariances are specified in continous-time.
  preintegration_params_ = PreintegratedImuMeasurements::Params::MakeSharedD(
      -imu_calib.gravity_magnitude);
  preintegration_params_->gyroscopeCovariance =
      std::pow(imu_calib.gyro_noise_density, 2.0) * Eigen::Matrix3d::Identity();
  preintegration_params_->accelerometerCovariance =
      std::pow(imu_calib.acc_noise_density, 2.0) * Eigen::Matrix3d::Identity();
  preintegration_params_->integrationCovariance =
      std::pow(imu_calib.imu_integration_sigma, 2.0) *
      Eigen::Matrix3d::Identity();
  preintegration_params_->biasAccCovariance =
      std::pow(imu_calib.acc_bias_random_walk_sigma, 2.0) *
      Eigen::Matrix3d::Identity();
  preintegration_params_->biasOmegaCovariance =
      std::pow(imu_calib.gyro_bias_random_walk_sigma, 2.0) *
      Eigen::Matrix3d::Identity();
  preintegration_params_->biasAccOmegaInt.setZero();
  preintegration_params_->omegaCoriolis = imu_calib.omega_coriolis;
  preintegration_params_->use2ndOrderCoriolis = false;
}

void GraphManager::reset()
{
  VLOG(3) << "Backend: Graph Manager reset";
  new_factors_.resize(0);
  new_values_.clear();
  point_id_to_new_smart_factor_map_.clear();
  point_id_to_smart_factor_info_map_.clear();
  last_added_state_index_ = -1;
  smart_factor_stats_ = SmartFactorStatistics();

  bundle_id_to_pose_prior_factor_id_map_.clear();
  bundle_id_to_prior_slot_id_map_.clear();
  point_id_to_point_prior_factor_id_map_.clear();
  point_id_to_point_prior_slot_id_map_.clear();
  point_id_to_range_prior_factor_id_map_.clear();
  point_id_to_range_prior_factor_id_map_.clear();

  first_bundle_id_to_relative_prior_factor_id_map_.clear();
  first_bundle_id_to_prior_slot_id_map_.clear();
  first_bundle_id_to_second_bundle_id_map_.clear();
}

// -----------------------------------------------------------------------------

void GraphManager::addLandmark(const svo::Point& point)
{
  // Add the initial landmark position expressed in the world frame.
  new_values_.insert(gtsam::Symbol('l', point.id()),
                     gtsam::Point3(point.pos()));
}

void GraphManager::addExtrinsics(const svo::Transformation& T_b_c,
                                 const size_t cam_id)
{
  new_values_.insert(gtsam::Symbol('C', cam_id),
                     gtsam::Pose3(T_b_c.getTransformationMatrix()));
}

void GraphManager::addObservationToLandmark(const svo::Frame& frame,
                                            const size_t keypoint_index,
                                            const bool refine_extrinsics)
{
  //  std::cout << "Ftr index: " << keypoint_index << std::endl;
  //  std::cout << "Frames: \n"
  //            << "- n lm: " << frame.landmark_vec_.size() << std::endl
  //            << "- n f_vec: " << frame.f_vec_.size() << std::endl
  //            << "- n level_vec: " << frame.level_vec_.size() << std::endl;
  //  CHECK(!frame.in_ba_graph_vec_[keypoint_index]);
  const int idx = static_cast<int>(keypoint_index);
  const size_t pose_id = static_cast<size_t>(frame.bundleId());
  const svo::Point& point =
      *CHECK_NOTNULL(frame.landmark_vec_.at(keypoint_index).get());
  const size_t point_id = static_cast<size_t>(point.id());
  const size_t level = static_cast<size_t>(frame.level_vec_(idx));

  if (options_.use_bearing_factor)
  {
    const Eigen::Vector3d f = frame.f_vec_.col(static_cast<int>(idx));
    if (refine_extrinsics)
    {
      new_factors_.push_back(boost::make_shared<CameraBearingTbcFactor>(
          f, uplane_noise_pyr_.at(level), gtsam::Symbol('x', pose_id),
          gtsam::Symbol('C', frame.nframe_index_),
          gtsam::Symbol('l', point_id)));
    }
    else
    {
      new_factors_.push_back(boost::make_shared<CameraBearingFactor3D>(
          f, uplane_noise_pyr_.at(level), gtsam::Symbol('x', pose_id),
          gtsam::Symbol('l', point_id),
          gtsam::Pose3(frame.T_imu_cam().getTransformationMatrix())));
    }
  }
  else
  {
    CHECK(!refine_extrinsics) << "Tbc refinement not implemented for 2D "
                                 "projection yet.";
    const Eigen::Vector2d uv = vk::project2(frame.f_vec_.col(idx));
    new_factors_.push_back(boost::make_shared<ProjectionFactor>(
        uv, uplane_noise_pyr_.at(level), gtsam::Symbol('x', pose_id),
        gtsam::Symbol('l', point_id), cam_calib_,
        gtsam::Pose3(frame.T_imu_cam().getTransformationMatrix())));
  }

  //  frame.in_ba_graph_vec_[keypoint_index] = true;

  VLOG(40) << "Added observation to point " << point.id();
}

// -----------------------------------------------------------------------------
void GraphManager::addSmartLandmark(const svo::Point& point,
                                    const Transformation& T_imu_cam)
{
  CHECK(point_id_to_smart_factor_info_map_.find(point.id()) ==
        point_id_to_smart_factor_info_map_.end());

  gtsam::Pose3 T_body_cam(T_imu_cam.getTransformationMatrix());
  SmartFactor::shared_ptr new_factor(new SmartFactor(
      smart_noise_, cam_calib_, T_body_cam, *smart_factor_params_));

  // book keeping
  point_id_to_new_smart_factor_map_.insert(
      std::make_pair(point.id(), new_factor));
  point_id_to_smart_factor_info_map_.insert(
      std::make_pair(point.id(), SmartFactorInfo(new_factor, -1)));
}

void GraphManager::addObservationToSmartLandmark(const svo::Frame& frame,
                                                 const size_t keypoint_index)
{
  const svo::Point& point =
      *CHECK_NOTNULL(frame.landmark_vec_.at(keypoint_index).get());

  // Update existing smart-factor.
  auto smart_factors_map_iter =
      point_id_to_smart_factor_info_map_.find(point.id());
  CHECK(smart_factors_map_iter != point_id_to_smart_factor_info_map_.end())
      << "Tried to add a new observation to an existing landmark that is not "
         "in graph";

  const Eigen::Vector2d uv =
      vk::project2(frame.f_vec_.col(static_cast<int>(keypoint_index)));

  // if the factor is already added,  we make a new one
  // and the old one will be deleted before next optimization;
  // otherwise we directly update the shared factor (to be added next time)
  // This is probably due to the fact that isam2 only update
  // affected variables from the new factors
  // https://bitbucket.org/gtborg/gtsam/issues/420/problem-with-isam2-stereo-smart-factors-no
  if (smart_factors_map_iter->second.slot_in_graph_ != -1)
  {
    SmartFactor::shared_ptr old_factor = smart_factors_map_iter->second.factor_;
    SmartFactor::shared_ptr new_factor =
        boost::make_shared<SmartFactor>(*old_factor);
    new_factor->add(uv,
                    gtsam::Symbol('x', static_cast<size_t>(frame.bundleId())));
    point_id_to_new_smart_factor_map_.insert(
        std::make_pair(point.id(), new_factor));
    smart_factors_map_iter->second.factor_ = new_factor;
  }
  else
  {
    smart_factors_map_iter->second.factor_->add(
        uv, gtsam::Symbol('x', static_cast<size_t>(frame.bundleId())));
  }

  VLOG(40) << "Added smart observation to point " << point.id();
}

// -----------------------------------------------------------------------------
void GraphManager::addCombinedImuFactor(const NFramesId& from_id,
                                        const NFramesId& to_id,
                                        const Eigen::Vector3d& acc_bias,
                                        const Eigen::Vector3d& gyr_bias,
                                        const double to_timestamp_sec,
                                        const ImuMeasurements& imu_measurements)
{
  // Create the preintegrated measurement.
  PreintegratedImuMeasurements pim(
      preintegration_params_, gtsam::imuBias::ConstantBias(acc_bias, gyr_bias));

  auto it = imu_measurements.rbegin();
  auto it_plus = imu_measurements.rbegin();
  for (++it_plus; it != imu_measurements.rend(); ++it, ++it_plus)
  {
    double dt = 0.0;
    if (it_plus == imu_measurements.rend())  // only for newest measurement
      dt = to_timestamp_sec - it->timestamp_;
    else
      dt = it_plus->timestamp_ - it->timestamp_;
    pim.integrateMeasurement(it->linear_acceleration_, it->angular_velocity_,
                             dt);
  }

  new_factors_.push_back(boost::make_shared<gtsam::CombinedImuFactor>(
      gtsam::Symbol('x', from_id), gtsam::Symbol('v', from_id),
      gtsam::Symbol('x', to_id), gtsam::Symbol('v', to_id),
      gtsam::Symbol('b', from_id), gtsam::Symbol('b', to_id), pim));

  VLOG(40) << "Integrated " << imu_measurements.size() << " measurements"
           << " between frame " << from_id << " and " << to_id;
}

void GraphManager::addInitalPosePriorVI(const NFramesId& state_index,
                                        const Transformation& T_W_B)
{
  // Rotate initial pose uncertainty in world coordinates to body coordinates.
  Eigen::Matrix3d R_W_B = T_W_B.getRotationMatrix();
  Eigen::Matrix3d R_B_W = R_W_B.transpose();
  init_pose_prior_visual_inertial_.topLeftCorner(3, 3) =
      R_B_W * init_pose_prior_visual_inertial_.topLeftCorner(3, 3) * R_W_B;

  // Add pose prior.
  gtsam::SharedNoiseModel noise_init_pose =
      gtsam::noiseModel::Gaussian::Covariance(init_pose_prior_visual_inertial_);
  new_factors_.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gtsam::Symbol('x', state_index),
      gtsam::Pose3(T_W_B.getTransformationMatrix()), noise_init_pose));

  VLOG(40) << "Priors - Rotation Cov: \n"
           << init_pose_prior_visual_inertial_.topLeftCorner(3, 3);
  VLOG(40) << "Priors - Position Cov: \n"
           << init_pose_prior_visual_inertial_.bottomRightCorner(3, 3);
}

void GraphManager::addPosePriorFactor(const NFramesId& state_index,
                                      const Transformation& T_W_B,
                                      const double rot_sig_rad,
                                      const double pos_sig_meter)
{
  gtsam::SharedNoiseModel noise;
  if (pos_sig_meter < 0 || rot_sig_rad < 0)
  {
    CHECK_NOTNULL(default_pose_prior_noise_.get());
    noise = default_pose_prior_noise_;
  }
  else
  {
    noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << gtsam::Vector3::Constant(rot_sig_rad),
         gtsam::Vector3::Constant(pos_sig_meter))
            .finished());
  }
  addPosePriorFactor(state_index, T_W_B, noise);
}

void GraphManager::addPosePriorFactor(const NFramesId& state_index,
                                      const Transformation& T_W_B,
                                      const gtsam::SharedNoiseModel& noise)
{
  new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gtsam::Symbol('x', state_index),
      gtsam::Pose3(T_W_B.getTransformationMatrix()), noise);
  registerNewFactor(state_index, new_factors_.size() - 1,
                    &bundle_id_to_pose_prior_factor_id_map_,
                    &bundle_id_to_prior_slot_id_map_);
}

void GraphManager::removePosePriorFactor(const NFramesId& state_index)
{
  deleteRegisteredFactor(state_index, &bundle_id_to_pose_prior_factor_id_map_,
                         &bundle_id_to_prior_slot_id_map_);
}

void GraphManager::addPointPriorFactor(const PointId& point_id,
                                       const Eigen::Vector3d& pw,
                                       const double pos_sig)
{
  gtsam::SharedNoiseModel noise;
  if (pos_sig < 0)
  {
    CHECK_NOTNULL(default_point_prior_noise_.get());
    noise = default_point_prior_noise_;
  }
  else
  {
    noise = gtsam::noiseModel::Isotropic::Sigma(3, pos_sig);
  }
  addPointPriorFactor(point_id, pw, noise);
}

void GraphManager::addPointPriorFactor(const PointId& point_id,
                                       const Eigen::Vector3d& pw,
                                       const gtsam::SharedNoiseModel& noise)
{
  new_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Point3>>(
      gtsam::Symbol('l', point_id), gtsam::Point3(pw), noise);
  registerNewFactor(point_id, new_factors_.size() - 1,
                    &point_id_to_point_prior_factor_id_map_,
                    &point_id_to_point_prior_slot_id_map_);
}

void GraphManager::removePointPriorFactor(const PointId& point_id)
{
  deleteRegisteredFactor(point_id, &point_id_to_point_prior_factor_id_map_,
                         &point_id_to_point_prior_slot_id_map_);
}

void GraphManager::addRangePriorFactor(const NFramesId& state_index,
                                       const PointId& point_id,
                                       const double dist,
                                       const Transformation& T_b_c,
                                       const gtsam::SharedNoiseModel& noise)
{
  CHECK_NOTNULL(noise.get());
  new_factors_.emplace_shared<CamPointDistFactor>(
      gtsam::Symbol('x', state_index), gtsam::Symbol('l', point_id), dist,
      noise, gtsam::Pose3(T_b_c.getTransformationMatrix()));
  registerNewFactor(point_id, new_factors_.size() - 1,
                    &point_id_to_range_prior_factor_id_map_,
                    &point_id_to_range_prior_slot_id_map_);
}

void GraphManager::removeRangePriorFactor(const PointId& point_id)
{
  deleteRegisteredFactor(point_id, &point_id_to_range_prior_factor_id_map_,
                         &point_id_to_range_prior_slot_id_map_);
}

void GraphManager::addRelativePosePriorFactor(
    const NFramesId& bundle_id_1, const NFramesId& bundle_id_2,
    const Transformation& T_p1_p2, const gtsam::SharedNoiseModel& noise)
{
  CHECK_NOTNULL(noise.get());
  new_factors_.emplace_shared<RelativePoseFactor>(
      gtsam::Symbol('x', bundle_id_1), gtsam::Symbol('x', bundle_id_2),
      gtsam::Pose3(T_p1_p2.getTransformationMatrix()), noise);
  registerNewFactor(bundle_id_1, new_factors_.size() - 1,
                    &first_bundle_id_to_relative_prior_factor_id_map_,
                    &first_bundle_id_to_prior_slot_id_map_);
  CHECK(first_bundle_id_to_second_bundle_id_map_.find(bundle_id_1) ==
        first_bundle_id_to_second_bundle_id_map_.end());
  first_bundle_id_to_second_bundle_id_map_[bundle_id_1] = bundle_id_2;
}

void GraphManager::removeRelativePosePriorFactor(const NFramesId& bundle_id_1,
                                                 const NFramesId& bundle_id_2)
{
  auto sec_it = first_bundle_id_to_second_bundle_id_map_.find(bundle_id_1);
  CHECK(sec_it != first_bundle_id_to_second_bundle_id_map_.end());
  CHECK_EQ(sec_it->second, bundle_id_2);

  deleteRegisteredFactor(bundle_id_1,
                         &first_bundle_id_to_relative_prior_factor_id_map_,
                         &first_bundle_id_to_prior_slot_id_map_);
  first_bundle_id_to_second_bundle_id_map_.erase(sec_it);
}

void GraphManager::addBiasPriorFactor(const NFramesId& state_index,
                                      const Eigen::Vector3d& acc_bias,
                                      const Eigen::Vector3d& gyr_bias)
{
  new_factors_.push_back(
      boost::make_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          gtsam::Symbol('b', state_index),
          gtsam::imuBias::ConstantBias(acc_bias, gyr_bias),
          imu_bias_prior_noise_));

  VLOG(40) << "Priors - Bias Mean: Acc (" << acc_bias.transpose()
           << "), Gyro ( " << gyr_bias.transpose() << ")";
  VLOG(40) << "Priors - Bias Sigmas: "
           << imu_bias_prior_noise_->sigmas().transpose();
}

void GraphManager::addVelocityPriorFactor(const NFramesId& state_index,
                                          const Eigen::Vector3d& W_v)
{
  new_factors_.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
      gtsam::Symbol('v', state_index), W_v, velocity_prior_noise_));

  VLOG(40) << "Priors - Velocity";
}

void GraphManager::addZeroVelocityPriorFactor(const NFramesId& state_index)
{
  new_factors_.push_back(boost::make_shared<gtsam::PriorFactor<gtsam::Vector3>>(
      gtsam::Symbol('v', state_index), Eigen::Vector3d::Zero(),
      zero_velocity_prior_noise_));
}

void GraphManager::augmentStateWithPose(const NFramesId& state_index,
                                        const Transformation& T_W_B)
{
  VLOG(4) << "Backend: Add new pose: " << state_index;
  new_values_.insert(gtsam::Symbol('x', state_index),
                     gtsam::Pose3(T_W_B.getTransformationMatrix()));
  last_added_state_index_ = state_index;
}

void GraphManager::augmentStateWithVelocityAndBias(
    const NFramesId& state_index, const Eigen::Vector3d& W_v,
    const Eigen::Vector3d& acc_bias, const Eigen::Vector3d& gyr_bias)
{
  new_values_.insert(gtsam::Symbol('v', state_index), W_v);
  new_values_.insert(gtsam::Symbol('b', state_index),
                     gtsam::imuBias::ConstantBias(acc_bias, gyr_bias));
}

void GraphManager::addPointMatchFactor(const PointId first,
                                       const PointId second)
{
  new_factors_.push_back(boost::make_shared<PointMatchFactor>(
      gtsam::Symbol('l', first), gtsam::Symbol('l', second),
      gtsam::Point3(0.0, 0.0, 0.0), point_match_noise_));
}

void GraphManager::registerNewFactor(const int id, const size_t factor_idx,
                                     IdToFactorIdxMap* id_to_factor_idx_map,
                                     IdToSlotIdMap* id_to_slot_id_map)
{
  auto it = id_to_factor_idx_map->find(id);
  if (it == id_to_factor_idx_map->end())
  {
    (*id_to_factor_idx_map)[id] = std::vector<size_t>{ factor_idx };
    CHECK(id_to_slot_id_map->find(id) == id_to_slot_id_map->end());
    (*id_to_slot_id_map)[id] = std::vector<int>{};
  }
  else
  {
    CHECK((*id_to_factor_idx_map)[id].size() != 0);
    (*id_to_factor_idx_map)[id].push_back(factor_idx);
    auto s_it = id_to_slot_id_map->find(id);
    CHECK(s_it != id_to_slot_id_map->end());
    CHECK_EQ(s_it->second.size(), 0u);
  }
}

void GraphManager::deleteRegisteredFactor(
    const int id, IdToFactorIdxMap* id_to_factor_idx_map,
    IdToSlotIdMap* id_to_slot_id_map)
{
  auto slot_map_it = id_to_slot_id_map->find(id);
  CHECK(slot_map_it != id_to_slot_id_map->end());
  if (slot_map_it->second.size() > 0u)
  {
    for (const auto slot_id : slot_map_it->second)
    {
      factor_slots_to_delete_.push_back(static_cast<size_t>(slot_id));
    }
  }
  else
  {
    CHECK(id_to_factor_idx_map->find(id) != id_to_factor_idx_map->end());
  }
  id_to_slot_id_map->erase(slot_map_it);

  auto factor_idx_it = id_to_factor_idx_map->find(id);
  if (factor_idx_it != id_to_factor_idx_map->end())
  {
    for (const auto factor_idx : factor_idx_it->second)
    {
      new_factors_.erase(new_factors_.begin() + factor_idx);
    }
    id_to_factor_idx_map->erase(factor_idx_it);
  }
}

void GraphManager::getUpdatesCopy(gtsam::NonlinearFactorGraph* new_factors,
                                  gtsam::Values* new_states,
                                  GTSAMSlotIndices* slots_to_delete,
                                  std::vector<int>* new_smart_factor_point_ids)
{
  CHECK_NOTNULL(new_factors);
  CHECK_NOTNULL(new_states);
  CHECK_NOTNULL(slots_to_delete);
  CHECK_NOTNULL(new_smart_factor_point_ids);

  // handle new smart factors
  for (auto& s : point_id_to_new_smart_factor_map_)
  {
    new_factors->push_back(s.second);
    new_smart_factor_point_ids->push_back(s.first);

    // get current slot, which will be deleted
    const auto& it = point_id_to_smart_factor_info_map_.find(s.first);
    if (it->second.slot_in_graph_ != -1)
    {
      CHECK_GE(it->second.slot_in_graph_, 0);
      slots_to_delete->push_back(
          static_cast<size_t>(it->second.slot_in_graph_));
    }
  }
  // ordinary factors to remove
  for (const auto& v : factor_slots_to_delete_)
  {
    slots_to_delete->push_back(v);
  }
  new_factors->push_back(new_factors_.begin(), new_factors_.end());

  // values
  (*new_states) = new_values_;

  // reset bookkeeping
  point_id_to_new_smart_factor_map_.clear();
  new_factors_.resize(0);
  new_values_.clear();
  factor_slots_to_delete_.clear();
}

void GraphManager::updateSlotInfo(
    const gtsam::FastVector<size_t>& new_slots,
    const std::vector<int>& new_smart_factor_point_ids)
{
  // update slots of new inserted indices:
  VLOG(1) << ">>> === will update slots info ===:" << std::endl;
  VLOG(1) << "- # new_slots: " << new_slots.size() << "; "
          << "- # new smart_factor_pt_ids: "
          << new_smart_factor_point_ids.size() << "; "
          << "- # new pose prior: "
          << bundle_id_to_pose_prior_factor_id_map_.size() << "; "
          << "- # new point position prior: "
          << point_id_to_point_prior_factor_id_map_.size() << "; "
          << "- # new point range prior: "
          << point_id_to_range_prior_factor_id_map_.size() << "; "
          << "- # new relative pose prior: "
          << first_bundle_id_to_relative_prior_factor_id_map_.size()
          << std::endl;
  for (size_t i = 0; i < new_smart_factor_point_ids.size(); ++i)
  {
    const auto& it = point_id_to_smart_factor_info_map_.find(
        new_smart_factor_point_ids.at(i));
    CHECK(it != point_id_to_smart_factor_info_map_.end());
    it->second.slot_in_graph_ = static_cast<int>(new_slots.at(i));
  }

  updateSlotInfoSingle(new_slots, bundle_id_to_pose_prior_factor_id_map_,
                       &bundle_id_to_prior_slot_id_map_);
  bundle_id_to_pose_prior_factor_id_map_.clear();

  updateSlotInfoSingle(new_slots, point_id_to_point_prior_factor_id_map_,
                       &point_id_to_point_prior_slot_id_map_);
  point_id_to_point_prior_factor_id_map_.clear();

  updateSlotInfoSingle(new_slots, point_id_to_range_prior_factor_id_map_,
                       &point_id_to_range_prior_slot_id_map_);
  point_id_to_range_prior_factor_id_map_.clear();

  updateSlotInfoSingle(new_slots,
                       first_bundle_id_to_relative_prior_factor_id_map_,
                       &first_bundle_id_to_prior_slot_id_map_);
  first_bundle_id_to_relative_prior_factor_id_map_.clear();
}

void GraphManager::updateSlotInfoSingle(
    const gtsam::FastVector<size_t>& new_slots,
    const svo::GraphManager::IdToFactorIdxMap& id_to_factor_idx_map,
    svo::GraphManager::IdToSlotIdMap* id_to_slot_id_map)
{
  CHECK_NOTNULL(id_to_slot_id_map);
  for (const auto& p : id_to_factor_idx_map)
  {
    auto it = id_to_slot_id_map->find(p.first);
    CHECK(it != id_to_slot_id_map->end() && it->second.empty());
    for (const size_t factor_idx : p.second)
    {
      it->second.push_back(static_cast<int>(new_slots[factor_idx]));
    }
  }
}

void GraphManager::updateSmartFactorStatistics()
{
  for (const auto& p : point_id_to_smart_factor_info_map_)
  {
    const SmartFactor& f = *(p.second.factor_);
    if (f.isValid())
    {
      smart_factor_stats_.n_valid++;
    }
    else if (f.isDegenerate())
    {
      smart_factor_stats_.n_degenerate++;
    }
    else if (f.isPointBehindCamera())
    {
      smart_factor_stats_.n_behind_cam++;
    }
    else if (f.isOutlier())
    {
      smart_factor_stats_.n_outlier++;
    }
    else if (f.isFarPoint())
    {
      smart_factor_stats_.n_farpoint++;
    }
    else
    {
      LOG(FATAL) << "Unknown factor status.";
    }
  }
}

}  // namespace svo
