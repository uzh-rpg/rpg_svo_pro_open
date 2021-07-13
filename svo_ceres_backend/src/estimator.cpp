/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2016, ETH Zurich, Wyss Zurich, Zurich Eye
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file Estimator.cpp
 * @brief Source file for the Estimator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include "svo/ceres_backend/estimator.hpp"

#include <svo/common/conversions.h>
#include <svo/common/point.h>

#include "svo/ceres_backend/ceres_iteration_callback.hpp"
#include "svo/ceres_backend/imu_error.hpp"
#include "svo/ceres_backend/marginalization_error.hpp"
#include "svo/ceres_backend/pose_error.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"
#include "svo/ceres_backend/relative_pose_error.hpp"
#include "svo/ceres_backend/reprojection_error.hpp"
#include "svo/ceres_backend/speed_and_bias_error.hpp"
#include "svo/ceres_backend/homogeneous_point_error.hpp"

namespace svo {
std::vector<std::string> MarginalizationTiming::names_ {
  "0_mag_pre_iterate", "1_mag_collection_non_pose_terms",
  "2_marg_collect_poses", "3_actual_marginalization",
  "4_marg_update_errors", "5_finish"
};

}

/// \brief ze Main namespace of this package.
namespace svo {

using ErrorType = ceres_backend::ErrorType;

// Constructor if a ceres map is already available.
Estimator::Estimator(
    std::shared_ptr<ceres_backend::Map> map_ptr)
  : map_ptr_(map_ptr),
    cauchy_loss_function_ptr_(new ceres::CauchyLoss(1)),
    huber_loss_function_ptr_(new ceres::HuberLoss(1)),
    marginalization_residual_id_(0)
{}

// The default constructor.
Estimator::Estimator()
  : Estimator(std::make_shared<ceres_backend::Map>())
{}

Estimator::~Estimator()
{}

// Add a camera to the configuration. Sensors can only be added and never removed.
void Estimator::addCameraBundle(
    const ExtrinsicsEstimationParametersVec& extrinsics_estimation_parameters,
    const CameraBundlePtr camera_rig)
{
  CHECK(camera_rig != nullptr);
  CHECK_EQ(camera_rig->getNumCameras(), extrinsics_estimation_parameters.size());
  extrinsics_estimation_parameters_ = extrinsics_estimation_parameters;
  camera_rig_ = camera_rig;
  constant_extrinsics_ids_.resize(camera_rig_->getNumCameras());

  for (size_t i = 0; i < extrinsics_estimation_parameters_.size(); ++i)
  {
    if (extrinsics_estimation_parameters_[i].sigma_c_relative_translation > 1e-12
        ||
        extrinsics_estimation_parameters_[i].sigma_c_relative_orientation > 1e-12)
    {
      //! @todo This has to be changed to a vector of bools if for certain
      //! cameras we estimate the temporal changes of the extrinsics and for
      //! others we do not.
      estimate_temporal_extrinsics_ = true;
      LOG(FATAL) << "Estimating temporal changes of extrinsics seems to cause"
                    " jumps in the estimations. Not supported at this moment.";
    }
  }
}

// Add an IMU to the configuration.
int Estimator::addImu(const ImuParameters& imu_parameters)
{
  if(imu_parameters_.size() > 1)
  {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imu_parameters_.push_back(imu_parameters);
  return imu_parameters_.size() - 1;
}

// Remove all cameras from the configuration
void Estimator::clearCameras()
{
  extrinsics_estimation_parameters_.clear();
}

// Remove all IMUs from the configuration.
void Estimator::clearImus()
{
  imu_parameters_.clear();
}

// Add a pose to the state.
bool Estimator::addStates(const FrameBundleConstPtr& frame_bundle,
                          const ImuMeasurements &imu_measurements,
                          const double &timestamp)
{
  BackendId nframe_id = createNFrameId(frame_bundle->getBundleId());
  VLOG(20) << "Adding state to estimator. Bundle ID: "
           << frame_bundle->getBundleId()
           << " with backend-id: " << std::hex << nframe_id << std::dec
           << " num IMU measurements: " << imu_measurements.size();

  double last_timestamp = 0;
  Transformation T_WS;
  SpeedAndBias speed_and_bias;

  // initialization or propagate the IMU
  if (states_.ids.empty())
  {
    if(is_reinit_)
    {
      last_timestamp = reinit_timestamp_start_;
      speed_and_bias = reinit_speed_bias_;
      T_WS = reinit_T_WS_;
      int num_used_imu_measurements =
          ceres_backend::ImuError::propagation(
            imu_measurements, imu_parameters_.at(0), T_WS, speed_and_bias,
            last_timestamp, timestamp);
      CHECK_GT(num_used_imu_measurements, 1)
          << "No imu measurements is used for reinitialization."
             " Something wrong with the IMU bookkeeping.";
      T_WS.getRotation().normalize();
      is_reinit_ = false;
    }
    else
    {
      // in case this is the first frame ever, let's initialize the pose:
      bool success0 = initPoseFromImu(imu_measurements, T_WS);
      DEBUG_CHECK(success0) << "pose could not be initialized from imu measurements.";
      if (!success0)
      {
        return false;
      }
      speed_and_bias.setZero();
      speed_and_bias.segment<3>(6) = imu_parameters_.at(0).a0;
    }
  }
  else
  {
    last_timestamp = states_.timestamps.back();
    // get the previous states
    BackendId T_WS_id = states_.ids.back();
    BackendId speed_and_bias_id = changeIdType(T_WS_id, IdType::ImuStates);
    T_WS =
        std::static_pointer_cast<ceres_backend::PoseParameterBlock>(
          map_ptr_->parameterBlockPtr(T_WS_id.asInteger()))->estimate();
    speed_and_bias =
        std::static_pointer_cast<ceres_backend::SpeedAndBiasParameterBlock>(
          map_ptr_->parameterBlockPtr(
            speed_and_bias_id.asInteger()))->estimate();
    //! @todo last_timestamp redundant because we already select imu
    //!       measurements for specific timespan
    int num_used_imu_measurements =
        ceres_backend::ImuError::propagation(
          imu_measurements, imu_parameters_.at(0), T_WS, speed_and_bias,
          last_timestamp, timestamp, nullptr, nullptr);
    T_WS.getRotation().normalize();
    //! @todo could check this sooner if we select IMU measurements as we do
//    DEBUG_CHECK(num_used_imu_measurements > 1) << "propagation failed";
    if (num_used_imu_measurements < 1)
    {
      LOG(ERROR) << "numUsedImuMeasurements=" << num_used_imu_measurements;
      return false;
    }
  }

  // check if id was used before
  DEBUG_CHECK(!states_.findSlot(nframe_id).second)
      << "pose ID" << nframe_id << " was used before!";

  // add the pose states
  std::shared_ptr<ceres_backend::PoseParameterBlock> pose_parameter_block =
      std::make_shared<ceres_backend::PoseParameterBlock>(T_WS, nframe_id.asInteger());
  if (!map_ptr_->addParameterBlock(pose_parameter_block,
                                   ceres_backend::Map::Pose6d))
  {
    return false;
  }
  states_.addState(nframe_id, false, timestamp);
  // add IMU states
  for (size_t i=0; i<imu_parameters_.size(); ++i)
  {
    BackendId id = changeIdType(nframe_id, IdType::ImuStates);
    std::shared_ptr<ceres_backend::SpeedAndBiasParameterBlock>
        speed_and_bias_parameter_block =
        std::make_shared<ceres_backend::SpeedAndBiasParameterBlock>(speed_and_bias,
                                                           id.asInteger());
    if(!map_ptr_->addParameterBlock(speed_and_bias_parameter_block))
    {
      return false;
    }
  }

  // Now we deal with error terms
  CHECK_GE(states_.ids.size(), 1u);

  // add initial prior or IMU errors
  if (states_.ids.size() == 1)
  {
    // let's add a prior
    Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
    information(5,5) = 1.0e8;
    information(0,0) = 1.0e8;
    information(1,1) = 1.0e8;
    information(2,2) = 1.0e8;
    std::shared_ptr<ceres_backend::PoseError > pose_error =
        std::make_shared<ceres_backend::PoseError>(T_WS, information);
    map_ptr_->addResidualBlock(pose_error, nullptr, pose_parameter_block);
    registerFixedFrame(pose_parameter_block->id());

    for (size_t i = 0; i < imu_parameters_.size(); ++i)
    {
      // get these from parameter file
      const double sigma_bg = imu_parameters_.at(0).sigma_bg;
      const double sigma_ba = imu_parameters_.at(0).sigma_ba;
      std::shared_ptr<ceres_backend::SpeedAndBiasError > speed_and_bias_error =
          std::make_shared<ceres_backend::SpeedAndBiasError>(
            speed_and_bias, 1.0, sigma_bg*sigma_bg, sigma_ba*sigma_ba);
      // add to map
      map_ptr_->addResidualBlock(
            speed_and_bias_error,
            nullptr,
            map_ptr_->parameterBlockPtr(
              changeIdType(nframe_id, IdType::ImuStates).asInteger()));
    }
  }
  else
  {
    const BackendId last_nframe_id = states_.ids[states_.ids.size() - 2];
    for (size_t i = 0; i < imu_parameters_.size(); ++i)
    {
      std::shared_ptr<ceres_backend::ImuError> imuError =
          std::make_shared<ceres_backend::ImuError>(imu_measurements,
                                           imu_parameters_.at(i),
                                           last_timestamp,
                                           timestamp);
      map_ptr_->addResidualBlock(
            imuError,
            nullptr,
            map_ptr_->parameterBlockPtr(last_nframe_id.asInteger()),
            map_ptr_->parameterBlockPtr(
              changeIdType(last_nframe_id, IdType::ImuStates).asInteger()),
            map_ptr_->parameterBlockPtr(nframe_id.asInteger()),
            map_ptr_->parameterBlockPtr(
              changeIdType(nframe_id, IdType::ImuStates).asInteger()));
    }
  }

  // Now deal with extrinsics
  // At the beginning or changing extrinsics, we add new parameter
  // for the extrinsics; otherwise we only keep one set of extrinsics
  std::vector<BackendId> cur_bundle_extrinsics_ids = constant_extrinsics_ids_;
  if (estimate_temporal_extrinsics_ || states_.ids.size() == 1)
  {
    for (size_t i = 0; i < extrinsics_estimation_parameters_.size(); ++i)
    {
      const Transformation T_S_C = camera_rig_->get_T_C_B(i).inverse();
      cur_bundle_extrinsics_ids[i] = changeIdType(nframe_id, IdType::Extrinsics, i);
      std::shared_ptr<ceres_backend::PoseParameterBlock> extrinsics_parameter_block =
          std::make_shared<ceres_backend::PoseParameterBlock>(
            T_S_C, cur_bundle_extrinsics_ids[i].asInteger());
      if (!map_ptr_->addParameterBlock(extrinsics_parameter_block,
                                       ceres_backend::Map::Pose6d))
      {
        return false;
      }
    }
    if (!estimate_temporal_extrinsics_)
    {
      constant_extrinsics_ids_ = cur_bundle_extrinsics_ids;
    }
  }

  if (states_.ids.size() == 1)
  {
    // sensor states
    for (size_t i = 0; i < extrinsics_estimation_parameters_.size(); ++i)
    {
      if(extrinsics_estimation_parameters_[i].isExtrinsicsFixed())
      {
        map_ptr_->setParameterBlockConstant(
              cur_bundle_extrinsics_ids[i].asInteger());
      }
      else
      {
        const Transformation T_SC = camera_rig_->get_T_C_B(i).inverse();
        std::shared_ptr<ceres_backend::PoseError > camera_pose_error =
            std::make_shared<ceres_backend::PoseError>(T_SC,
              extrinsics_estimation_parameters_[i].absoluteTranslationVar(),
              extrinsics_estimation_parameters_[i].absoluteRotationVar());
        // add to map
        map_ptr_->addResidualBlock(
              camera_pose_error,
              nullptr,
              map_ptr_->parameterBlockPtr(cur_bundle_extrinsics_ids[i].asInteger()));
      }
    }
  }
  else
  {
    const BackendId last_nframe_id = states_.ids[states_.ids.size() - 2];
    for (size_t i = 0; i < extrinsics_estimation_parameters_.size(); ++i)
    {
      if(estimate_temporal_extrinsics_)
      {
        // i.e. they are different estimated variables, so link them with a
        // temporal error term
        double dt =  timestamp - last_timestamp;
        double translation_sigma_c =
            extrinsics_estimation_parameters_[i].sigma_c_relative_translation;
        double translation_variance =
            translation_sigma_c * translation_sigma_c * dt;
        double rotation_sigma_c =
            extrinsics_estimation_parameters_[i].sigma_c_relative_orientation;
        double rotation_variance = rotation_sigma_c * rotation_sigma_c * dt;
        std::shared_ptr<ceres_backend::RelativePoseError> relative_extrinsics_error =
            std::make_shared<ceres_backend::RelativePoseError>(translation_variance,
                                                      rotation_variance);
        map_ptr_->addResidualBlock(
              relative_extrinsics_error,
              nullptr,
              map_ptr_->parameterBlockPtr(
                changeIdType(last_nframe_id, IdType::Extrinsics, i).asInteger()),
              map_ptr_->parameterBlockPtr(
                changeIdType(nframe_id, IdType::Extrinsics, i).asInteger()));
      }
    }
  }

  return true;
}

// Add a landmark.
bool Estimator::addLandmark(const PointPtr &landmark,
                            const bool set_fixed)
{
  // track id is the same as point id
  BackendId landmark_backend_id = createLandmarkId(landmark->id());

  std::shared_ptr<ceres_backend::HomogeneousPointParameterBlock>
      point_parameter_block =
      std::make_shared<ceres_backend::HomogeneousPointParameterBlock>(
        landmark->pos(), landmark_backend_id.asInteger());
  if (!map_ptr_->addParameterBlock(point_parameter_block,
                                   ceres_backend::Map::HomogeneousPoint))
  {
    return false;
  }

  // add landmark to map
  landmarks_map_.emplace_hint(
        landmarks_map_.end(),
        landmark_backend_id, MapPoint(landmark));
  DEBUG_CHECK(isLandmarkAdded(landmark_backend_id))
      << "bug: inconsistend landmarkdMap_ with mapPtr_.";
  landmark->in_ba_graph_ = true;

  // fixation
  if (set_fixed)
  {
    setLandmarkConstant(landmark_backend_id);
    registerFixedLandmark(landmark_backend_id.asInteger());
  }
  return true;
}

// Add a high prior to a landmarks position
bool Estimator::setLandmarkConstant(const BackendId &landmark_backend_id)
{
  DEBUG_CHECK(landmark_backend_id.type() == IdType::Landmark);
  map_ptr_->setParameterBlockConstant(landmark_backend_id.asInteger());
  landmarks_map_[landmark_backend_id].fixed_position = true;

  return true;
}

bool Estimator::changeSpeedAndBiasesFixation(const bool set_fixed)
{
  for (const BackendId& id : states_.ids)
  {
    BackendId sba_id = changeIdType(id, IdType::ImuStates);
    if (!map_ptr_->parameterBlockExists(sba_id.asInteger()))
    {
      continue;
    }
    std::shared_ptr<ceres_backend::SpeedAndBiasParameterBlock> block_ptr =
        std::static_pointer_cast<ceres_backend::SpeedAndBiasParameterBlock>(
          map_ptr_->parameterBlockPtr(sba_id.asInteger()));
    if (block_ptr != nullptr)
    {
      if (set_fixed && !block_ptr->fixed())
      {
        map_ptr_->setParameterBlockConstant(block_ptr);
      }
      else if (!set_fixed && block_ptr->fixed())
      {
        map_ptr_->setParameterBlockVariable(block_ptr);
      }
    }
  }

  return true;
}

void Estimator::removeAllFixedLandmarks()
{
  for (const uint64_t param_id : fixed_landmark_parameter_ids_)
  {
    removeLandmarkByBackendId(BackendId(param_id), true);
  }
  fixed_landmark_parameter_ids_.clear();
}

size_t Estimator::numValidFixedLandmarks() const
{
  LOG(FATAL) << "NOT TESTED!!";
  size_t cnt = 0;
  for (const uint64_t param_id : fixed_landmark_parameter_ids_)
  {
    ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(BackendId(param_id).asInteger());
    if (residuals.size() >= 2)
    {
      cnt++;
    }
  }
  return cnt;
}

void Estimator::setAllFixedLandmarksEnabled(const bool enabled)
{
  for (const uint64_t param_id : fixed_landmark_parameter_ids_)
  {
    ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(BackendId(param_id).asInteger());
//    if (residuals.size() == 1)
//    {
//      std::shared_ptr<ceres_backend::ReprojectionError> rep_error =
//          std::dynamic_pointer_cast<ceres_backend::ReprojectionError>(
//            residuals[0].error_interface_ptr);
//      if(rep_error)
//      {
//        rep_error->setDisabled(true);
//      }
//      continue;
//    }
    size_t n_proj = 0;
    for (size_t r = 0; r < residuals.size(); ++r)
    {
      std::shared_ptr<ceres_backend::ReprojectionError> rep_error =
          std::dynamic_pointer_cast<ceres_backend::ReprojectionError>(
            residuals[r].error_interface_ptr);

      if(rep_error)
      {
        rep_error->setDisabled(!enabled);
        n_proj ++;
        continue;
      }

      LOG(FATAL) << "Should not reach here.";
    }
    CHECK_GT(n_proj, 0u);
  }
}

void Estimator::updateFixedLandmarks()
{
  for (const uint64_t param_id : fixed_landmark_parameter_ids_)
  {
    BackendId bid(param_id);

    const Eigen::Vector3d pos = landmarks_map_[bid].point->pos();
    const Eigen::Vector4d homo_pos (pos[0], pos[1], pos[2], 1.0);
    ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(bid.asInteger());
    std::shared_ptr<ceres_backend::HomogeneousPointParameterBlock> pt_ptr =
        std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
          map_ptr_->parameterBlockPtr(param_id));
    CHECK(pt_ptr);
    pt_ptr->setEstimate(homo_pos);
  }
}

void Estimator::removeLandmarkByBackendId(
    const BackendId &bid, const bool check_fixed)
{
  ceres_backend::Map::ResidualBlockCollection residuals =
      map_ptr_->residuals(bid.asInteger());
  for (size_t r = 0; r < residuals.size(); ++r)
  {
    map_ptr_->removeResidualBlock(residuals[r].residual_block_id);
  }

  if (check_fixed)
  {
    CHECK(map_ptr_->isParameterBlockConstant(bid.asInteger()));
  }

  map_ptr_->removeParameterBlock(bid.asInteger());
  landmarks_map_[bid].point->in_ba_graph_ = false;
  landmarks_map_.erase(bid);
}

bool Estimator::addVelocityPrior(BackendId nframe_id,
                                 const Eigen::Vector3d& velocity,
                                 double sigma)
{
  DEBUG_CHECK(states_.findSlot(nframe_id).second);
  SpeedAndBias speed_and_bias;
  speed_and_bias.head<3>() = velocity;
  speed_and_bias.tail<6>().setZero();
  const double speed_information = 1.0 / (sigma * sigma);
  const double bias_information = 0.0;
  Eigen::Matrix<double, 9, 9> information;
  information.setIdentity();
  information.topLeftCorner<3, 3>() *= speed_information;
  information.bottomLeftCorner<6, 6>() *= bias_information;
  std::shared_ptr<ceres_backend::SpeedAndBiasError> prior =
      std::make_shared<ceres_backend::SpeedAndBiasError>(speed_and_bias, information);
  ceres::ResidualBlockId id =
      map_ptr_->addResidualBlock(
        prior,
        nullptr,
        map_ptr_->parameterBlockPtr(
          changeIdType(nframe_id, IdType::ImuStates).asInteger()));
  return id != nullptr;
}

void Estimator::resetPrior()
{
  marginalization_error_ptr_.reset(
        new ceres_backend::MarginalizationError(*map_ptr_.get()));
}

// Remove an observation from a landmark.
bool Estimator::removeObservation(ceres::ResidualBlockId residual_block_id)
{
  const ceres_backend::Map::ParameterBlockCollection parameters =
      map_ptr_->parameters(residual_block_id);
  const BackendId landmarkId(parameters.at(1).first);
  DEBUG_CHECK(landmarkId.type() == IdType::Landmark);
  // remove in landmarksMap
  MapPoint& map_point = landmarks_map_.at(landmarkId);
  for(auto it = map_point.observations.begin();
      it!= map_point.observations.end();)
  {
    if(it->second == uint64_t(residual_block_id))
    {
      it = map_point.observations.erase(it);
    }
    else
    {
      ++it;
    }
  }
  // remove residual block
  map_ptr_->removeResidualBlock(residual_block_id);
  return true;
}

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template<class T>
bool vectorContains(const std::vector<T>& vector, const T & query)
{
  for (size_t i = 0; i < vector.size(); ++i)
  {
    if (vector[i] == query)
    {
      return true;
    }
  }
  return false;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool Estimator::applyMarginalizationStrategy(
    size_t num_keyframes, size_t num_imu_frames, MarginalizationTiming* timing)
{
  vk::Timer timer;
  if (timing)
  {
    timing->reset();
    timer.start();
  }

  // keep the newest numImuFrames
  std::vector<BackendId>::reverse_iterator rit_id = states_.ids.rbegin();
  std::vector<bool>::reverse_iterator rit_keyframe =
      states_.is_keyframe.rbegin();
  for (size_t k = 0; k < num_imu_frames; ++k)
  {
    ++rit_id;
    ++rit_keyframe;
    if (rit_id==states_.ids.rend())
    {
      // nothing to do.
      return true;
    }
  }
  // if not, continue looping the rest of the states

  // remove linear marginalizationError, if existing
  if (marginalization_error_ptr_ && marginalization_residual_id_)
  {
    bool success = map_ptr_->removeResidualBlock(marginalization_residual_id_);
    DEBUG_CHECK(success) << "could not remove marginalization error";
    marginalization_residual_id_ = 0;
    if (!success)
      return false;
  }

  // these will keep track of what we want to marginalize out.
  //! @todo keepParameterBlocks could be removed. Only 'false' is pushed back..
  std::vector<uint64_t> parameter_blocks_to_be_marginalized;
  std::vector<bool> keep_parameter_blocks;

  if (!hasPrior())
  {
    resetPrior();
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<BackendId> marginalize_pose_frames;
  std::vector<BackendId> marginalize_all_but_pose_frames;
  std::vector<BackendId> all_linearized_frames;
  size_t counted_keyframes = 0;
  // Note: rit is now pointing to the first frame not in the sliding window
  // => Either the first keyframe or the frame falling out of the sliding window.
  while (rit_id != states_.ids.rend())
  {
    // we marginalize in two cases
    //   * a frame outside the imu window but is not a keyframe
    //   * the oldest keyframe when we have enough keyframe
    if (!(*rit_keyframe) || counted_keyframes >= num_keyframes)
    {
      marginalize_pose_frames.push_back(*rit_id);
    }
    else
    {
      counted_keyframes++;
    }

    // for all the frames outside the IMU window, we only keep the pose
    marginalize_all_but_pose_frames.push_back(*rit_id);
    all_linearized_frames.push_back(*rit_id);
    ++rit_id;// check the next frame
    ++rit_keyframe;
  }
  if (timing)
  {
    timing->add(std::string("0_mag_pre_iterate"), timer.stop());
    timer.start();
  }

  // marginalize everything but pose:
  for (size_t k = 0; k < marginalize_all_but_pose_frames.size(); ++k)
  {
    // Add all IMU error terms.
    uint64_t speed_and_bias_id =
        changeIdType(marginalize_all_but_pose_frames[k], IdType::ImuStates).asInteger();
    if (!map_ptr_->parameterBlockExists(speed_and_bias_id))
    {
      continue; // already marginalized.
    }
    if (map_ptr_->parameterBlockPtr(speed_and_bias_id)->fixed())
    {
      continue; // Do not remove fixed blocks.
    }
    parameter_blocks_to_be_marginalized.push_back(speed_and_bias_id);
    keep_parameter_blocks.push_back(false);

    // Get all residuals connected to this state.
    ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(speed_and_bias_id);
    for (size_t r = 0; r < residuals.size(); ++r)
    {
      if (residuals[r].error_interface_ptr->typeInfo() != ErrorType::kReprojectionError)
      {
        marginalization_error_ptr_->addResidualBlock(
              residuals[r].residual_block_id);
      }
    }
  }
  if (timing)
  {
    timing->add(std::string("1_mag_collection_non_pose_terms"), timer.stop());
    timer.start();
  }

  // marginalize ONLY pose now:
  // For frames whose poses are marginalized, also deal with landmarks
  for (size_t k = 0; k < marginalize_pose_frames.size(); ++k)
  {
    // schedule removal
    parameter_blocks_to_be_marginalized.push_back(
          marginalize_pose_frames[k].asInteger());
    keep_parameter_blocks.push_back(false);

    // add remaing error terms
    ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(marginalize_pose_frames[k].asInteger());

    // pose
    for (size_t r = 0; r < residuals.size(); ++r)
    {
      ErrorType cur_t = residuals[r].error_interface_ptr->typeInfo();
      if(cur_t == ErrorType::kPoseError)
      {
        // avoids linearising initial pose error
        map_ptr_->removeResidualBlock(residuals[r].residual_block_id);
        deRegisterFixedFrame(marginalize_pose_frames[k].asInteger());
        continue;
      }

      if (cur_t != ErrorType::kReprojectionError)
      {
        // we make sure no reprojection errors are yet included.
        marginalization_error_ptr_->addResidualBlock(
              residuals[r].residual_block_id);
      }
    }

    // add remaining error terms of the sensor states.
    if (estimate_temporal_extrinsics_)
    {
      for(size_t cam_idx = 0; cam_idx < camera_rig_->getNumCameras(); ++cam_idx)
      {
        uint64_t extr_id =
            changeIdType(marginalize_pose_frames[k],
                         IdType::Extrinsics, cam_idx).asInteger();
        if (map_ptr_->parameterBlockPtr(extr_id)->fixed())
        {
          continue;  // we never eliminate fixed blocks.
        }
        parameter_blocks_to_be_marginalized.push_back(extr_id);
        keep_parameter_blocks.push_back(false);

        ceres_backend::Map::ResidualBlockCollection residuals =
            map_ptr_->residuals(extr_id);
        for (size_t r = 0; r < residuals.size(); ++r)
        {
          if (residuals[r].error_interface_ptr->typeInfo() != ErrorType::kReprojectionError)
          {
            marginalization_error_ptr_->addResidualBlock(
                  residuals[r].residual_block_id);
          }
        }
      }
    }

    // now finally we trmarginalize_pose_framesbservations.
    DEBUG_CHECK(all_linearized_frames.size()>0) << "bug";
    // this is the id of the oldest frame in the sliding window
    const BackendId current_kf_id = all_linearized_frames.at(0);
    // If the frame dropping out of the sliding window is not a keyframe, then
    // the observations are deleted. If it is, then the landmarks visible in
    // the oldest keyframe but not the newest one are marginalized.
    {
      for(PointMap::iterator pit = landmarks_map_.begin();
          pit != landmarks_map_.end();)
      {
        ceres_backend::Map::ResidualBlockCollection residuals =
            map_ptr_->residuals(pit->first.asInteger());
        CHECK(residuals.size() != 0);

        // dealing with fixed landmarks: just delete corresponding residuals
        if (pit->second.fixed_position)
        {
          for (size_t r = 0; r < residuals.size(); ++r)
          {
            if (residuals[r].error_interface_ptr->typeInfo() ==
                ErrorType::kReprojectionError)
            {
              BackendId pose_id =
                  BackendId(
                    map_ptr_->parameters(residuals[r].residual_block_id).at(0).first);
              if (vectorContains(marginalize_pose_frames, pose_id))
              {
                removeObservation(residuals[r].residual_block_id);
                residuals.erase(residuals.begin() + r);
                break;
              }
            }
          }
          if (residuals.size() == 0)
          {
            deRegisterFixedLandmark(pit->first.asInteger());
            map_ptr_->removeParameterBlock(pit->first.asInteger());
            pit->second.point->in_ba_graph_ = false;
            pit = landmarks_map_.erase(pit);
          }
          else
          {
            pit++;
          }
          continue;
        }


        // Now the ordinary landmarks
        // First loop: check if we can skip
        bool skip_landmark = true;
        bool visible_in_imu_window = false;
        bool just_delete = false;
        bool marginalize = true;
        bool error_term_added = false;
        size_t obs_count = 0;
        for (size_t r = 0; r < residuals.size(); ++r)
        {
          if (residuals[r].error_interface_ptr->typeInfo() == ErrorType::kReprojectionError)
          {
            BackendId pose_id =
                BackendId(
                  map_ptr_->parameters(residuals[r].residual_block_id).at(0).first);

            // if the landmark is visible inthe frame to marginalize
            if(vectorContains(marginalize_pose_frames, pose_id))
            {
              skip_landmark = false;
            }

            // the landmark is still visible in the IMU window, we keep it
            if(pose_id >= current_kf_id)
            {
              marginalize = false;
              visible_in_imu_window = true;
            }

            if(vectorContains(all_linearized_frames, pose_id))
            {
              obs_count++;
            }
          }
        }

        // the landmark is not affected by the marginalization
        if(skip_landmark)
        {
          pit++;
          continue;
        }

        // Second loop: actually collect residuals to marginalize
        for (size_t r = 0; r < residuals.size(); ++r)
        {
          if (residuals[r].error_interface_ptr->typeInfo() == ErrorType::kReprojectionError)
          {
            BackendId pose_id(
                  map_ptr_->parameters(residuals[r].residual_block_id).at(0).first);
            const bool is_pose_to_be_margin =
                vectorContains(marginalize_pose_frames, pose_id);
            const bool is_pose_in_sliding_window =
                vectorContains(all_linearized_frames, pose_id);

            if((is_pose_to_be_margin && visible_in_imu_window )||
               (!is_pose_in_sliding_window && !visible_in_imu_window) ||
               is_pose_to_be_margin)
            {
              // ok, let's ignore the observation.
              removeObservation(residuals[r].residual_block_id);
              residuals.erase(residuals.begin() + r);
              r--;
            }
            else if(!visible_in_imu_window && is_pose_in_sliding_window)
            {
              // TODO: consider only the sensible ones for marginalization
              if(obs_count < 2)
              {
                removeObservation(residuals[r].residual_block_id);
                residuals.erase(residuals.begin() + r);
                r--;
              }
              else
              {
                // add information to be considered in marginalization later.
                error_term_added = true;
                // the residual term is deleted from the map as well
                marginalization_error_ptr_->addResidualBlock(
                      residuals[r].residual_block_id, false);
              }
            }

            // check anything left
            if (residuals.size() == 0)
            {
              just_delete = true;
              marginalize = false;
            }
          }
        }

        // now we deal with parameter blocks
        if(just_delete)
        {
          map_ptr_->removeParameterBlock(pit->first.asInteger());
          pit->second.point->in_ba_graph_ = false;
          pit = landmarks_map_.erase(pit);
          continue;
        }

        if(marginalize && error_term_added)
        {
          parameter_blocks_to_be_marginalized.push_back(pit->first.asInteger());
          keep_parameter_blocks.push_back(false);
          pit->second.point->in_ba_graph_ = false;
          pit = landmarks_map_.erase(pit);
          continue;
        }

        pit++;
      } // loop of landmark map
    }

    // update book-keeping and go to the next frame
    //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    VLOG(20) << "Marginalizing out state with id " << marginalize_pose_frames[k];
    states_.removeState(marginalize_pose_frames[k]);
  }
  if (timing)
  {
    timing->add(std::string("2_marg_collect_poses"), timer.stop());
    timer.start();
  }

  // now apply the actual marginalization
  if(parameter_blocks_to_be_marginalized.size() > 0)
  {
    if (FLAGS_v >= 20)
    {
      std::stringstream s;
      s << "Marginalizing following parameter blocks:\n";
      for (uint64_t id : parameter_blocks_to_be_marginalized)
      {
        s << BackendId(id) << "\n";
      }
      VLOG(20) << s.str();
    }

    // clean parameter blocks --> some get lost in marginalization term during
    // loop closures
    for(std::vector<uint64_t>::iterator it =
        parameter_blocks_to_be_marginalized.begin();
        it != parameter_blocks_to_be_marginalized.end();)
    {
      if(!marginalization_error_ptr_->isInMarginalizationTerm(*it))
      {
        VLOG(20) << "removing block with id " << *it;
        it = parameter_blocks_to_be_marginalized.erase(it);
        keep_parameter_blocks.pop_back(); // this only works because all false
        // proper way:
//        keep_parameter_blocks.erase(
//              keep_parameter_blocks.begin()+static_cast<long>(
//                it-parameter_blocks_to_be_marginalized.begin()));
        continue;
      }
      ++it;
    }

    marginalization_error_ptr_->marginalizeOut(parameter_blocks_to_be_marginalized,
                                               keep_parameter_blocks);
  }
  if (timing)
  {
    timing->add(std::string("3_actual_marginalization"), timer.stop());
    timer.start();
  }

  // update error computation
  if(parameter_blocks_to_be_marginalized.size() > 0)
  {
    marginalization_error_ptr_->updateErrorComputation();
  }
  if (timing)
  {
    timing->add(std::string("4_marg_update_errors"), timer.stop());
    timer.start();
  }

  // add the marginalization term again
  if(marginalization_error_ptr_->num_residuals()==0)
  {
    marginalization_error_ptr_.reset();
  }
  if (marginalization_error_ptr_)
  {
    std::vector<std::shared_ptr<ceres_backend::ParameterBlock> > parameter_block_ptrs;
    marginalization_error_ptr_->getParameterBlockPtrs(parameter_block_ptrs);
    marginalization_residual_id_ = map_ptr_->addResidualBlock(
          marginalization_error_ptr_, nullptr, parameter_block_ptrs);
    DEBUG_CHECK(marginalization_residual_id_)
        << "could not add marginalization error";
    if (!marginalization_residual_id_)
    {
      return false;
    }
  }

  if(needPoseFixation() && !hasFixedPose())
  {
    // finally fix the first pose properly
    setOldestFrameFixed();
  }
  if (timing)
  {
    timing->add(std::string("5_finish"), timer.stop());
  }

  return true;
}


// Prints state information to buffer.
void Estimator::printStates(BackendId pose_id, std::ostream& buffer) const
{
  auto slot = states_.findSlot(pose_id);
  if (!slot.second)
  {
    buffer << "Tried to print info on pose with ID " << pose_id
           << " which is not part of the estimator." << std::endl;
    return;
  }
  buffer << "Pose. ID: " << pose_id
         << " - Keyframe: " << (states_.is_keyframe[slot.first] ? "yes" : "no")
      << " - Timestamp: " << states_.timestamps[slot.first]
      << ":\n";
  buffer << getPoseEstimate(pose_id).first << "\n";

  BackendId speed_and_bias_id = changeIdType(pose_id, IdType::ImuStates);
  auto sab = getSpeedAndBiasEstimate(speed_and_bias_id);
  if (sab.second)
  {
    buffer << "Speed and Bias. ID: " << speed_and_bias_id << ":\n";
    buffer << sab.first.transpose() << "\n";
  }
  std::vector<BackendId> extrinsics_id = constant_extrinsics_ids_;
  if (estimate_temporal_extrinsics_)
  {
    for (size_t i = 0; i < extrinsics_id.size(); ++i)
    {
      extrinsics_id[i] = changeIdType(pose_id, IdType::Extrinsics, i);
    }
  }
  for (size_t i = 0; i < extrinsics_id.size(); ++i)
  {
    auto extrinsics = getPoseEstimate(extrinsics_id[i]);
    if (extrinsics.second)
    {
      buffer << "Extrinsics. ID: " << extrinsics_id[i] << ":\n";
      buffer << extrinsics.first << "\n";
    }
  }
  buffer << "-------------------------------------------" << std::endl;
}

// Initialise pose from IMU measurements. For convenience as static.
bool Estimator::initPoseFromImu(
    const ImuMeasurements &imu_measurements,
    Transformation& T_WS)
{
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  const int n_measurements = imu_measurements.size();

  if (n_measurements == 0)
  {
    return false;
  }

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (int i = 0; i < n_measurements; ++i)
  {
    acc_B += imu_measurements[i].linear_acceleration_;
  }
  acc_B /= static_cast<double>(n_measurements);
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> pose_increment;
  pose_increment.head<3>() = Eigen::Vector3d::Zero();
  //! @todo this gives a bad result if ez_W.cross(e_acc) norm is
  //! close to zero, deal with it!
  pose_increment.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  pose_increment.tail<3>() *= angle;
  T_WS = Transformation::exp(-pose_increment) * T_WS;
  T_WS.getRotation().normalize();
  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void Estimator::optimize(size_t num_iter, size_t num_threads, bool verbose)
#else
void Estimator::optimize(size_t num_iter, size_t /*num_threads*/,
                         bool verbose) // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif

{  
  //DEBUG
  //  LOG(ERROR) << "printing all parameters";
  //  const std::unordered_map<uint64_t,
  //        std::shared_ptr<ceres_backend::ParameterBlock> > &idmap =
  //        map_ptr_->idToParameterBlockMap();
  //  for(auto &it : idmap)
  //  {
  //    if(it.second->typeInfo()!="HomogeneousPointParameterBlock")
  //    {
  //      map_ptr_->printParameterBlockInfo(it.first);
  //      const double* params = it.second->parameters();
  //      for(size_t i = 0; i<it.second->dimension(); ++i)
  //      {
  //        LOG(ERROR) << params[i];
  //      }
  //    }
  //  }
  // assemble options
//  map_ptr_->options.linear_solver_type = ceres::SPARSE_SCHUR;
  map_ptr_->options.linear_solver_type = ceres::DENSE_SCHUR;
  //map_ptr_->options.initial_trust_region_radius = 1.0e4;
  //map_ptr_->options.initial_trust_region_radius = 2.0e6;
  //map_ptr_->options.preconditioner_type = ceres::IDENTITY;
  map_ptr_->options.trust_region_strategy_type = ceres::DOGLEG;
  //map_ptr_->options.use_nonmonotonic_steps = true;
  //map_ptr_->options.max_consecutive_nonmonotonic_steps = 10;
  //map_ptr_->options.function_tolerance = 1e-12;
  //map_ptr_->options.gradient_tolerance = 1e-12;
  //map_ptr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
  map_ptr_->options.num_threads = num_threads;
#endif
  map_ptr_->options.max_num_iterations = num_iter;

  if (verbose)
  {
    map_ptr_->options.minimizer_progress_to_stdout = true;
  }
  else
  {
    map_ptr_->options.logging_type = ceres::LoggingType::SILENT;
    map_ptr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  map_ptr_->solve();

  // update landmarks
  {
    for(auto &id_and_map_point : landmarks_map_)
    {
      // update coordinates
      id_and_map_point.second.hom_coordinates =
          std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
            map_ptr_->parameterBlockPtr(
              id_and_map_point.first.asInteger()))->estimate();
    }
  }


  //  //DEBUG
  //  LOG(ERROR) << "after optimization"
  //  for(auto &it : idmap)
  //  {
  //    if(it.second->typeInfo()!="HomogeneousPointParameterBlock")
  //    {
  //      map_ptr_->printParameterBlockInfo(it.first);
  //      const double* params = it.second->parameters();
  //      for(size_t i = 0; i<it.second->dimension(); ++i)
  //      {
  //        LOG(ERROR) << params[i];
  //      }
  //    }
  //  }
  // summary output
  if (verbose)
  {
    LOG(INFO) << map_ptr_->summary.FullReport();
    std::stringstream s;
    for (const auto& id : states_.ids)
    {
      printStates(id, s);
    }
    LOG(INFO) << s.str();
  }
}

// Set a time limit for the optimization process.
bool Estimator::setOptimizationTimeLimit(double time_limit, int min_iterations)
{
  if(ceres_callback_ != nullptr)
  {
    if(time_limit < 0.0)
    {
      // no time limit => set minimum iterations to maximum iterations
      ceres_callback_->setMinimumIterations(map_ptr_->options.max_num_iterations);
      return true;
    }
    ceres_callback_->setTimeLimit(time_limit);
    ceres_callback_->setMinimumIterations(min_iterations);
    return true;
  }
  else if(time_limit >= 0.0)
  {
    ceres_callback_.reset(
          new ceres_backend::CeresIterationCallback(time_limit, min_iterations));
    map_ptr_->options.callbacks.push_back(ceres_callback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

// getters
// Get a specific landmark.
bool Estimator::getLandmark(BackendId landmark_id,
                            MapPoint& map_point) const
{
  if (landmarks_map_.find(landmark_id) == landmarks_map_.end())
  {
    DEBUG_CHECK(false)
        << "landmark with id = " << landmark_id << " does not exist.";
    return false;
  }
  map_point = landmarks_map_.at(landmark_id);
  return true;
}

void Estimator::updateAllActivePoints() const
{
  for(auto &id_and_map_point : landmarks_map_)
  {
    if (isLandmarkFixed(id_and_map_point.first.asInteger()))
    {
      continue;
    }
    // update coordinates
    id_and_map_point.second.point->pos_=
        id_and_map_point.second.hom_coordinates.head<3>();
  }
}

// Checks whether the landmark is initialized.
bool Estimator::isLandmarkInitialized(BackendId landmark_id) const
{
  DEBUG_CHECK(isLandmarkAdded(landmark_id)) << "landmark not added";
  return std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
        map_ptr_->parameterBlockPtr(landmark_id.asInteger()))->initialized();
}

// Get a copy of all the landmarks as a PointMap.
size_t Estimator::getLandmarks(PointMap& landmarks) const
{
  landmarks = landmarks_map_;
  return landmarks_map_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy support.
// Use getLandmarks(svo::PointMap&) if possible.
size_t Estimator::getLandmarks(MapPointVector& landmarks) const
{
  landmarks.clear();
  landmarks.reserve(landmarks_map_.size());
  for(PointMap::const_iterator it=landmarks_map_.begin();
      it!=landmarks_map_.end(); ++it)
  {
    landmarks.push_back(it->second);
  }
  return landmarks_map_.size();
}

// Get pose for a given backend ID.
bool Estimator::get_T_WS(BackendId id,
                         Transformation& T_WS) const
{
  DEBUG_CHECK(id.type() == IdType::NFrame) << "wrong id type: id = " << id;
  bool success;
  std::tie(T_WS, success) = getPoseEstimate(id);
  return success;
}

// Get speeds and IMU biases for a given pose ID.
bool Estimator::getSpeedAndBias(BackendId id,
                                SpeedAndBias& speed_and_bias) const
{
  bool success;
  BackendId sab_id = changeIdType(id, IdType::ImuStates);
  std::tie(speed_and_bias, success) = getSpeedAndBiasEstimate(sab_id);
  return success;
}

// Get camera states for a given pose ID.
bool Estimator::getCameraSensorStatesFromNFrameId(
    BackendId pose_id, size_t camera_idx, Transformation& T_SCi) const
{
  BackendId extrinsics_id;
  if (estimate_temporal_extrinsics_)
  {
    extrinsics_id = changeIdType(pose_id, IdType::Extrinsics, camera_idx);
  }
  else
  {
    extrinsics_id = constant_extrinsics_ids_.at(camera_idx);
  }
  bool success;
  std::tie(T_SCi, success) = getPoseEstimate(extrinsics_id);
  return success;
}

// Get the ID of the current keyframe.
BackendId Estimator::currentKeyframeId() const
{
  for (size_t i = states_.ids.size() - 1; i < states_.ids.size(); --i)
  {
    if (states_.is_keyframe[i])
    {
      return states_.ids[i];
    }
  }
  DEBUG_CHECK(false) << "no existing keyframes ...";
  return BackendId();
}

// Get the ID of the current keyframe.
BundleId Estimator::oldestKeyframeBundleId() const
{
  for (size_t i = 0; i < states_.ids.size(); ++i)
  {
    if (states_.is_keyframe[i])
    {
      return states_.ids[i].bundleId();
    }
  }
  return -1;
}

// Get the ID of an older frame.
BackendId Estimator::frameIdByAge(size_t age) const
{
  DEBUG_CHECK(age<numFrames())
      << "requested age " << age << " out of range.";
  return states_.ids[numFrames()-1-age];
}

// Get the ID of the newest frame added to the state.
BackendId Estimator::currentFrameId() const
{
  DEBUG_CHECK(states_.ids.size() > 0) << "no frames added yet.";
  return states_.ids.back();
}

BundleId Estimator::currentBundleId() const
{
  return currentFrameId().bundleId();
}

// Checks if a particular frame is still in the IMU window
bool Estimator::isInImuWindow(BackendId nframe_id) const
{
  return getSpeedAndBiasEstimate(nframe_id).second;
}

// Set pose for a given pose ID.
bool Estimator::set_T_WS(BackendId pose_id,
                         const Transformation& T_WS)
{
  DEBUG_CHECK(pose_id.type() == IdType::NFrame);
  return setPoseEstimate(pose_id, T_WS);
}

// Set position for a given landmark.
bool Estimator::setLandmarkPosition(BackendId landmark_id,
                                    const Position& landmark_pos)
{
  DEBUG_CHECK(landmark_id.type() == IdType::Landmark);
  if (!map_ptr_->parameterBlockExists(landmark_id.asInteger()))
  {
    return false;
  }
  Eigen::Vector4d hom_position;
  hom_position << landmark_pos, 1.0;
  std::shared_ptr<ceres_backend::HomogeneousPointParameterBlock> block_ptr =
      std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
        map_ptr_->parameterBlockPtr(landmark_id.asInteger()));
  block_ptr->setEstimate(hom_position);
  return true;
}

// Set the speeds and IMU biases for a given pose ID.
bool Estimator::setSpeedAndBiasFromNFrameId(BackendId pose_id,
                                            const SpeedAndBias& speed_and_bias)
{
  DEBUG_CHECK(pose_id.type() == IdType::NFrame);
  BackendId sab_id = changeIdType(pose_id, IdType::ImuStates);
  return setSpeedAndBiasEstimate(sab_id, speed_and_bias);
}

// Set the transformation from sensor to camera frame for a given pose ID.
bool Estimator::setCameraSensorStates(
    BackendId pose_id, uint8_t camera_idx,
    const Transformation & T_SCi)
{
  DEBUG_CHECK(pose_id.type() == IdType::NFrame);
  BackendId extrinsics_id = changeIdType(pose_id, IdType::Extrinsics, camera_idx);
  return setPoseEstimate(extrinsics_id, T_SCi);
}

// Set the landmark initialization state.
void Estimator::setLandmarkInitialized(BackendId landmark_id,
                                       bool initialized)
{
  DEBUG_CHECK(isLandmarkAdded(landmark_id)) << "landmark not added";
  std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
        map_ptr_->parameterBlockPtr(landmark_id.asInteger()))->setInitialized(initialized);
}

// private stuff
// getters

std::pair<Transformation, bool> Estimator::getPoseEstimate(BackendId id) const
{
  DEBUG_CHECK(id.type() == IdType::NFrame ||
              id.type() == IdType::Extrinsics);
  if (!map_ptr_->parameterBlockExists(id.asInteger()))
  {
    return std::make_pair(Transformation(), false);
  }
#ifndef NDEBUG
  std::shared_ptr<ceres_backend::ParameterBlock> base_ptr =
      map_ptr_->parameterBlockPtr(id.asInteger());
  if (base_ptr != nullptr)
  {
    std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
        std::dynamic_pointer_cast<ceres_backend::PoseParameterBlock>(base_ptr);
    CHECK(block_ptr != nullptr) << "Incorrect pointer cast detected!";
    return std::make_pair(block_ptr->estimate(), true);
  }
#else
  std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
      std::static_pointer_cast<ceres_backend::PoseParameterBlock>(
        map_ptr_->parameterBlockPtr(id.asInteger()));
  if (block_ptr != nullptr)
  {
    return std::make_pair(block_ptr->estimate(), true);
  }
#endif
  return std::make_pair(Transformation(), false);
}

std::pair<SpeedAndBias, bool> Estimator::getSpeedAndBiasEstimate(BackendId id) const
{
  DEBUG_CHECK(id.type() == IdType::ImuStates);
  if (!map_ptr_->parameterBlockExists(id.asInteger()))
  {
    return std::make_pair(SpeedAndBias(), false);
  }
#ifndef NDEBUG
  std::shared_ptr<ceres_backend::ParameterBlock> base_ptr =
      map_ptr_->parameterBlockPtr(id.asInteger());
  if (base_ptr != nullptr)
  {
    std::shared_ptr<ceres_backend::SpeedAndBiasParameterBlock> block_ptr =
        std::dynamic_pointer_cast<ceres_backend::SpeedAndBiasParameterBlock>(base_ptr);
    CHECK(block_ptr != nullptr) << "Incorrect pointer cast detected!";
    return std::make_pair(block_ptr->estimate(), true);
  }
#else
  std::shared_ptr<ceres_backend::SpeedAndBiasParameterBlock> block_ptr =
      std::static_pointer_cast<ceres_backend::SpeedAndBiasParameterBlock>(
        map_ptr_->parameterBlockPtr(id.asInteger()));
  if (block_ptr != nullptr)
  {
    return std::make_pair(block_ptr->estimate(), true);
  }
#endif
  return std::make_pair(SpeedAndBias(), false);
}

bool Estimator::setPoseEstimate(BackendId id, const Transformation& pose)
{
  if (!map_ptr_->parameterBlockExists(id.asInteger()))
  {
    return false;
  }

  DEBUG_CHECK(id.type() == IdType::NFrame ||
              id.type() == IdType::Extrinsics);
#ifndef NDEBUG
  std::shared_ptr<ceres_backend::ParameterBlock> base_ptr =
      map_ptr_->parameterBlockPtr(id.asInteger());
  if (base_ptr == nullptr)
  {
    return false;
  }
  std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
      std::dynamic_pointer_cast<ceres_backend::PoseParameterBlock>(base_ptr);
  CHECK(base_ptr != nullptr) << "Incorrect pointer cast detected!";
  block_ptr->setEstimate(pose);
#else
  std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
      std::static_pointer_cast<ceres_backend::PoseParameterBlock>(
        map_ptr_->parameterBlockPtr(id.asInteger()));
  block_ptr->setEstimate(pose);
#endif
  return true;
}

bool Estimator::setSpeedAndBiasEstimate(BackendId id, const SpeedAndBias& sab)
{
  DEBUG_CHECK(id.type() == IdType::ImuStates);
  if (!map_ptr_->parameterBlockExists(id.asInteger()))
  {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<ceres_backend::ParameterBlock> base_ptr =
      map_ptr_->parameterBlockPtr(id.asInteger());
  if (base_ptr == nullptr)
  {
    return false;
  }
  std::shared_ptr<ceres_backend::SpeedAndBiasParameterBlock> block_ptr =
      std::dynamic_pointer_cast<ceres_backend::SpeedAndBiasParameterBlock>(base_ptr);
  CHECK(base_ptr != nullptr) << "Incorrect pointer cast detected!";
  block_ptr->setEstimate(sab);
#else
  std::shared_ptr<ceres_backend::SpeedAndBiasParameterBlock> block_ptr =
      std::static_pointer_cast<ceres_backend::SpeedAndBiasParameterBlock>(
        map_ptr_->parameterBlockPtr(id.asInteger()));
  block_ptr->setEstimate(sab);
#endif
  return true;
}

bool Estimator::removePointsByPointIds(std::vector<int> &track_ids)
{
  for(int &id : track_ids)
  {
    BackendId landmark_id = createLandmarkId(id);
    if (!isLandmarkInEstimator(landmark_id))
    {
      continue;
    }
    if (isLandmarkFixed(landmark_id.asInteger()))
    {
      removeLandmarkByBackendId(landmark_id, true);
      deRegisterFixedLandmark(landmark_id.asInteger());
    }
    else
    {
      removeLandmarkByBackendId(landmark_id, false);
    }
  }
  return true;
}

bool Estimator::setFrameFixed(const BundleId &fixed_frame_bundle_id,
                              const Transformation &T_WS_new)
{
  LOG(FATAL) << "This function should not be used currently.";
  BackendId new_fixed_frame = createNFrameId(fixed_frame_bundle_id);
  LOG(ERROR) << "Setting frame fixed";

//   compute the relative transformation
  Transformation T_WS_old;
  if(!get_T_WS(new_fixed_frame, T_WS_old))
  {
    return false;
  }

  bool success = false;
  const Transformation T_old_new = T_WS_old.inverse()*T_WS_new;
  for (BackendId &nframe_id : states_.ids)
  {

    std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
        std::static_pointer_cast<ceres_backend::PoseParameterBlock>(
          map_ptr_->parameterBlockPtr(nframe_id.asInteger()));
    DEBUG_CHECK(block_ptr) << "Incorrect pointer cast";
    T_WS_old = block_ptr->estimate();
    block_ptr->setEstimate(T_WS_old*T_old_new);

    //! @todo shift all corresponding points
    if(!success)
    {
      // add remaing error terms
      ceres_backend::Map::ResidualBlockCollection residuals =
          map_ptr_->residuals(nframe_id.asInteger());

      for (ceres_backend::Map::ResidualBlockSpec &residual : residuals)
      {
        // find the previously fixed block
        if (residual.error_interface_ptr->typeInfo() == ErrorType::kPoseError)
        {
          LOG(ERROR) << "Removed old fixation";
          // remove the old fixation
          map_ptr_->removeResidualBlock(residual.residual_block_id);
          deRegisterFixedFrame(nframe_id.asInteger());

          // fix the given frame
          Eigen::Matrix<double, 6, 6> information =
              Eigen::Matrix<double,6,6>::Zero();
          information(0, 0) = 1.0e35;
          information(1, 1) = 1.0e35;
          information(2, 2) = 1.0e35;
          information(5, 5) = 1.0e35;
          std::shared_ptr<ceres_backend::PoseError> pose_error =
              std::make_shared<ceres_backend::PoseError>(T_WS_new, information);
          std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
              std::static_pointer_cast<ceres_backend::PoseParameterBlock>(
                map_ptr_->parameterBlockPtr(new_fixed_frame.asInteger()));
          registerFixedFrame(nframe_id.asInteger());
          DEBUG_CHECK(block_ptr) << "Incorrect pointer cast";
          LOG(ERROR) << "setting pose to " << std::endl << T_WS_new;
          block_ptr->setEstimate(T_WS_new);
          map_ptr_->addResidualBlock(
                pose_error, nullptr,block_ptr);
          success =  true;
        }
      }
    }
  }

     DEBUG_CHECK(success) << "No frame was fixed";
     return success;
}

void Estimator::setOldestFrameFixed()
{
  Transformation T_WS_0;
  BackendId oldest_id = states_.ids[0];
  get_T_WS(oldest_id, T_WS_0);
  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double,6,6>::Zero();
  information(0, 0) = 1.0e14;
  information(1, 1) = 1.0e14;
  information(2, 2) = 1.0e14;
  information(5, 5) = 1.0e14;
  std::shared_ptr<ceres_backend::PoseError> pose_error =
      std::make_shared<ceres_backend::PoseError>(T_WS_0, information);
  map_ptr_->addResidualBlock(
        pose_error, nullptr,
        map_ptr_->parameterBlockPtr(oldest_id.asInteger()));
  registerFixedFrame(oldest_id.asInteger());
}

bool Estimator::removeAllPoseFixation()
{
  bool success = false;
  for (BackendId &nframe_id : states_.ids)
  {
    // add remaing error terms
    ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(nframe_id.asInteger());

    for (ceres_backend::Map::ResidualBlockSpec &residual : residuals)
    {
      // find the previously fixed block
      if (residual.error_interface_ptr->typeInfo() == ErrorType::kPoseError)
      {
        // remove the old fixation
        bool success_cur =
            map_ptr_->removeResidualBlock(residual.residual_block_id);
        DEBUG_CHECK(success_cur) << "Fixed frame could not be removed";
        deRegisterFixedFrame(nframe_id.asInteger());
        if (success_cur)
        {
          success = success_cur;
        }
      }
    }
  }
  return success;
}

bool Estimator::uniteLandmarks(const BackendId &old_id, const BackendId &new_id)
{
  DEBUG_CHECK(old_id.type() == IdType::Landmark) << "Old id is not landmark";
  DEBUG_CHECK(new_id.type() == IdType::Landmark) << "New id is not landmark";

  // get new parameterblock of landmarks
  std::shared_ptr<ceres_backend::HomogeneousPointParameterBlock> new_block =
      std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
        map_ptr_->parameterBlockPtr(new_id.asInteger()));
  DEBUG_CHECK(new_block) << "New landmark is not in backend";

  ceres_backend::Map::ResidualBlockCollection old_residuals =
      map_ptr_->residuals(old_id.asInteger());
  for(ceres_backend::Map::ResidualBlockSpec &residual : old_residuals)
  {
    std::shared_ptr<ceres_backend::ReprojectionError> error =
        std::static_pointer_cast<ceres_backend::ReprojectionError>(
          residual.error_interface_ptr);
    // remove the residual from the old block
    map_ptr_->removeResidualBlock(residual.residual_block_id);
    // add it to the new block
    map_ptr_->addResidualBlock(error,nullptr,new_block);
  }
  // remove the old block
  map_ptr_->removeParameterBlock(old_id.asInteger());

  // transfer observations in landmarks_map
  MapPoint& old_map_point = landmarks_map_.at(old_id);
  landmarks_map_.at(new_id).observations.insert(old_map_point.observations.begin(),
                                    old_map_point.observations.end());
  // remove old point from landmarks_map
  old_map_point.point->in_ba_graph_ = false;
  landmarks_map_.erase(old_id);

  return true;
}

// Transform the entire map with respect to the world frame
// fixation of all frames must be released before this!!!!
void Estimator::transformMap(const Transformation &w_T,
                             bool remove_marginalization_term,
                             bool recalculate_imu_terms)
{
  for(BackendId &nframe_id : states_.ids)
  {
    //set pose of nframe
    std::shared_ptr<ceres_backend::PoseParameterBlock> block_ptr =
        std::static_pointer_cast<ceres_backend::PoseParameterBlock>(
          map_ptr_->parameterBlockPtr(nframe_id.asInteger()));
    DEBUG_CHECK(block_ptr) << "Incorrect pointer cast";
    block_ptr->setEstimate(w_T*block_ptr->estimate());
  }

  for(auto &id_and_map_point : landmarks_map_)
  {
    // make sure not to change fixed points
    if(id_and_map_point.second.fixed_position)
    {
      continue;
    }

    // get Parameter block
    std::shared_ptr<ceres_backend::HomogeneousPointParameterBlock> block_ptr =
        std::static_pointer_cast<ceres_backend::HomogeneousPointParameterBlock>(
          map_ptr_->parameterBlockPtr(id_and_map_point.first.asInteger()));
    DEBUG_CHECK(block_ptr) << "Incorrect pointer cast";

    // update coordinates
    Eigen::Vector4d new_position = w_T.transform4(block_ptr->estimate());
    block_ptr->setEstimate(new_position);
  }

  // remove the marginalization error at the associated frames
  if(marginalization_error_ptr_ && remove_marginalization_term)
  {
    bool success = map_ptr_->removeResidualBlock(marginalization_residual_id_);
    DEBUG_CHECK(success) << "could not remove marginalization error";
    // remove pointer and ID
    marginalization_error_ptr_.reset();
    marginalization_residual_id_ = 0;
  }

  if (recalculate_imu_terms)
  {
    int cnt = 0;
    for (size_t sti = 0; sti < states_.ids.size(); sti++)
    {
      BackendId id = states_.ids[sti];
      ceres_backend::Map::ResidualBlockCollection residuals =
        map_ptr_->residuals(id.asInteger());
      for (size_t r = 0; r < residuals.size(); r++)
      {
        std::shared_ptr<ceres_backend::ImuError> imu_err =
            std::dynamic_pointer_cast<ceres_backend::ImuError>(
              residuals[r].error_interface_ptr);
        if (imu_err)
        {
          imu_err->setRedo(true);
          cnt++;
        }
      }
    }
    std::cout << "Will recompute " << cnt << " IMU terms." << std::endl;
  }
}

}  // namespace svo
