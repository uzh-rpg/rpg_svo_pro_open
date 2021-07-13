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
 * @file svo/Estimator.hpp
 * @brief Header file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#pragma once

#include <array>
#include <memory>
#include <mutex>

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <ceres/ceres.h>
#pragma diagnostic pop

#include <svo/common/types.h>
#include <svo/common/camera.h>
#include <svo/common/frame.h>
#include <svo/common/imu_calibration.h>

#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/estimator_types.hpp"

namespace svo {

// fwd:
namespace ceres_backend {
class MarginalizationError;
class CeresIterationCallback;
}

typedef std::shared_ptr<const FrameBundle> FrameBundleConstPtr;

struct States
{
  // ordered from oldest to newest.
  std::vector<BackendId> ids;
  std::vector<bool> is_keyframe;
  std::vector<double> timestamps;

  States() = default;

  void addState(BackendId id, bool keyframe, double timestamp)
  {
    DEBUG_CHECK(id.type() == IdType::NFrame);
    ids.push_back(id);
    is_keyframe.push_back(keyframe);
    timestamps.push_back(timestamp);
  }

  bool removeState(BackendId id)
  {
    auto slot = findSlot(id);
    if (slot.second)
    {
      ids.erase(ids.begin() + slot.first);
      is_keyframe.erase(is_keyframe.begin() + slot.first);
      timestamps.erase(timestamps.begin() + slot.first);
      return true;
    }
    return false;
  }

  std::pair<size_t, bool> findSlot(BackendId id) const
  {
    for (size_t i = 0; i < ids.size(); ++i)
    {
      if (ids[i] == id)
      {
        return std::make_pair(i, true);
      }
    }
    return std::make_pair(0, false);
  }
};

struct MarginalizationTiming
{
  static std::vector<std::string> names_;
  std::map<std::string, double> named_timing_;

  MarginalizationTiming()
  {
    for (const auto k : names_)
    {
      named_timing_.emplace(std::make_pair(k, 0.0));
    }
  }

  inline void reset()
  {
    for (const auto k : names_)
    {
      named_timing_[k] = 0.0;
    }
  }

  inline double get(const std::string& name) const
  {
    return named_timing_.at(name);
  }

  inline void add(const std::string& name, const double sec)
  {
    named_timing_[name] = sec;
  }
};


//! The estimator class
/*!
 The estimator class. This does all the backend work.
 Frames:
 W: World
 B: Body
 C: Camera
 S: Sensor (IMU)
 */
class Estimator
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Estimator();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param map_ptr Shared pointer to ceres map.
   */
  explicit Estimator(std::shared_ptr<ceres_backend::Map> map_ptr);
  ~Estimator();

  /// @name Sensor configuration related
  ///@{
  /**
   * @brief Add a camera rig to the configuration. Sensors can only be added and
   *        never removed.
   * @param extrinsics_estimation_parameters The parameters that tell how to
   *        estimate extrinsics.
   * @param camera_rig Shared pointer to the camera rig.
   */
  void addCameraBundle(
      const ExtrinsicsEstimationParametersVec& extrinsics_estimation_parameters,
      const CameraBundlePtr camera_rig);

  /**
   * @brief Add an IMU to the configuration.
   * @warning Currently there is only one IMU supported.
   * @param imu_parameters The IMU parameters.
   * @return index of IMU.
   */
  int addImu(const svo::ImuParameters& imu_parameters);

  /**
   * @brief Remove all cameras from the configuration
   */
  void clearCameras();

  /**
   * @brief Remove all IMUs from the configuration.
   */
  void clearImus();

  /// @}

  /**
   * @brief Add a frame to the state.
   * @param frame_bundle New nframe.
   * @param imu_measurements IMU measurements from last state to new one
   * @return True if successful.
   */
  bool addStates(const FrameBundleConstPtr &frame_bundle,
                 const ImuMeasurements& imu_measurements,
                 const double &timestamp);

  /**
   * @brief Prints state information to buffer.
   * @param pose_id The pose Id for which to print.
   * @param buffer The puffer to print into.
   */
  void printStates(BackendId pose_id, std::ostream& buffer) const;

  /**
   * @brief Add a landmark.
   * @param track_id Track ID of the new landmark.
   * @param landmark Homogeneous coordinates of landmark in W-frame.
   * @return True if successful.
   */
  bool addLandmark(const PointPtr &landmark, const bool set_fixed=false);

  /**
   * @brief Add a prior to the velocity at a specific nframe.
   * @param nframe_id Bundle ID of the new landmark.
   * @param velocity at the nframe in m/s.
   * @param sigma standard deviation of the prior
   * @return True if successful.
   */
  bool addVelocityPrior(BackendId nframe_id,
                        const Eigen::Vector3d& velocity,
                        double sigma);

  /**
   * @brief Add an observation to a landmark.
   * @param track_id Track ID of the landmark.
   * @param nframe_id ID of the nframe.
   * @param measurement Keypoint measurement.
   * @param level Pyramid level of the observation.
   * @param cam_idx ID of camera frame where the landmark was observed.
   * @param keypoint_idx ID of keypoint corresponding to the landmark.
   * @return Residual block ID for that observation.
   */
  ceres::ResidualBlockId addObservation(const FramePtr& frame,
                                        const size_t keypoint_idx);

  /**
   * @brief Applies the dropping/marginalization strategy according to the
   *        RSS'13/IJRR'14 paper. The new number of frames in the window will be
   *        numKeyframes+numImuFrames.
   * @param num_keyframes Number of keyframes.
   * @param num_imu_frames Number of frames in IMU window.
   * @return True if successful.
   */
  bool applyMarginalizationStrategy(size_t num_keyframes, size_t num_imu_frames,
                                    MarginalizationTiming* timing=nullptr);

  /**
   * @brief Initialise pose from IMU measurements. For convenience as static.
   * @param[in] imu_stamps Timestamps of the IMU measurements.
   * @param[in] imu_acc_gyr IMU measurements.
   * @param[out] T_WS initialised pose.
   * @return True if successful.
   */
  static bool initPoseFromImu(const ImuMeasurements &imu_measurements,
      Transformation& T_WS);

  /**
   * @brief Start ceres optimization.
   * @param[in] num_iter Maximum number of iterations.
   * @param[in] num_threads Number of threads.
   * @param[in] verbose Print out optimization progress and result, if true.
   */
  void optimize(size_t num_iter, size_t num_threads = 1, bool verbose = false);

  /**
   * @brief Set a time limit for the optimization process.
   * @param[in] time_limit Time limit in seconds. If timeLimit < 0 the time
   *            limit is removed.
   * @param[in] min_iterations minimum iterations the optimization process
   *            should do disregarding the time limit.
   * @return True if successful.
   */
  bool setOptimizationTimeLimit(double time_limit, int min_iterations);

  /**
   * @brief Checks whether the landmark is added to the estimator.
   * @param landmark_id The ID.
   * @return True if added.
   */
  bool isLandmarkAdded(BackendId landmark_id) const
  {
    bool isAdded = landmarks_map_.find(landmark_id) != landmarks_map_.end();
    DEBUG_CHECK(isAdded == map_ptr_->parameterBlockExists(landmark_id.asInteger()))
        << "id="<<landmark_id<<" inconsistent. isAdded = " << isAdded;
    return isAdded;
  }

  /**
   * @brief Checks whether the landmark is initialized.
   * @param landmark_id The ID.
   * @return True if initialised.
   */
  bool isLandmarkInitialized(BackendId landmark_id) const;

  /**
   * @brief Update position of specific svo::Point according to backend estimate.
   * @param[in] landmark_id referencing to specific landmark
   * @return True if successful.
   */
  bool updatePointPosition(const BackendId landmark_id) const;

  /**
   * @brief Update the position of all svo::Point according to backend estimate.
   */
  void updateAllActivePoints() const;

  /// @brief Check if the state at a slot in states is a keyframe
  /// @param[in] slot index of frame in states_ vector
  /// \return true if the frame at slot is a keyframe
  bool isStateKeyframeAtSlot(size_t slot) const
  {
    return states_.is_keyframe[slot];
  }

  /**
   * @brief Checks if a particular frame is still in the IMU window.
   * @param[in] nframe_id ID of frame to check.
   * @return True if the frame is in IMU window.
   */
  bool isInImuWindow(BackendId nframe_id) const;

  /**
   * @brief Remove the points with the track ids contained in track_ids
   * @param[in] track_ids The track ids of the points to be removed
   * @return true if successful
   */
  bool removePointsByPointIds(std::vector<int> &track_ids);

  /// @brief Remove the fixation of oldest frame, necessary when closing loop
  bool removeAllPoseFixation();

  /**
   * @brief Unite to landmarks to same, if loop is detected but landmark was
   *        introduced twice
   * @param old_id ID of older landmark that will be removed
   * @param new_id landmark ID, where observations of old landmark will be added
   * @return
   */
  bool uniteLandmarks(const BackendId &old_id, const BackendId &new_id);

  /**
   * @brief Set a strong prior to the position of a nframe (fix it)
   * @param fixed_frame_bundle_id ID of desired nframe
   * @param T_WS_new desired position
   * @return
   */
  bool setFrameFixed(const BundleId &fixed_frame_bundle_id,
                     const Transformation &T_WS_new);

  void setOldestFrameFixed();

  /**
   * @brief Transform the map with respect to the world frame. Used to keep good
   * initial guess after loop closure
   * @param w_T Transformation with respect to world frame (left sided)
   */
  void transformMap(const Transformation &w_T, bool remove_marginalization_term,
                    bool recalculate_imu_terms=false);


  /// @name Getters
  /// @{

  /// @brief Get the number of cameras used
  size_t getNumCameras() const {
    return extrinsics_estimation_parameters_.size();
  }

  /**
   * @brief Get a specific landmark.
   * @param[in]  landmark_id ID of desired landmark.
   * @param[out] map_point Landmark information, such as coordinates, landmark
   *             reference, observations.
   * @return True if successful.
   */
  bool getLandmark(BackendId landmark_id, MapPoint& map_point) const;


  /**
   * @brief Get the ID of a state at a slot in states
   * @param[in] slot index of frame in states_ vector
   * @return BackendId of the nframe at slot in states_
   */
  BackendId backendIdStateAtSlot(size_t slot) const
  {
    return states_.ids[slot];
  }
  /**
   * @brief Get a copy of all the landmarks as a PointMap.
   * @param[out] landmarks The landmarks.
   * @return number of landmarks.
   */
  size_t getLandmarks(PointMap& landmarks) const;

  /**
   * @brief Get a copy of all the landmark in a MapPointVector. This is for
   *        legacy support. Use getLandmarks(svo::PointMap&) if possible.
   * @param[out] landmarks A vector of all landmarks.
   * @see getLandmarks().
   * @return number of landmarks.
   */
  size_t getLandmarks(svo::MapPointVector& landmarks) const;

  bool get_T_WS(BackendId id, Transformation& T_WS) const;

  /**
   * @brief Get pose for a given pose ID.
   * @param[in] bundle_id Bundle ID of the FrameBundle.
   * @param[out] T_WS Homogeneous transformation of this pose.
   * @return True if successful.
   */
  bool get_T_WS(int32_t bundle_id,
                Transformation& T_WS) const
  {
    return get_T_WS(createNFrameId(bundle_id), T_WS);
  }


  /**
   * @brief Get speeds and IMU biases for a given pose ID.
   * @param[in] id Backend ID of the FrameBundle.
   * @param[out] speed_and_bias Speed And bias requested.
   * @return True if successful.
   */
  bool getSpeedAndBias(BackendId id, SpeedAndBias& speed_and_bias) const;

  /**
   * @brief Get speeds and IMU biases for a given pose ID.
   * @param[in] bundle_id Bundle ID of the FrameBundle.
   * @param[out] speed_and_bias Speed And bias requested.
   * @return True if successful.
   */
  bool getSpeedAndBias(int32_t bundle_id, SpeedAndBias& speed_and_bias) const
  {
    return getSpeedAndBias(createNFrameId(bundle_id),speed_and_bias);
  }

  /**
   * @brief Get camera states for a given pose ID.
   * @param[in]  pose_id ID of pose to get camera state for.
   * @param[in]  camera_idx index of camera to get state for.
   * @param[out] T_SCi Homogeneous transformation from sensor (IMU) frame to
   *             camera frame.
   * @return True if successful.
   */
  bool getCameraSensorStatesFromNFrameId(BackendId pose_id, size_t camera_idx,
                                         Transformation& T_SCi) const;

  /// @brief Get the number of states/frames in the estimator.
  /// \return The number of frames.
  size_t numFrames() const
  {
    return states_.ids.size();
  }

  /// @brief Get the number of landmarks in the estimator
  /// \return The number of landmarks.
  size_t numLandmarks() const
  {
    return landmarks_map_.size();
  }

  /// @brief Get the ID of the current keyframe.
  /// \return The ID of the current keyframe.
  BackendId currentKeyframeId() const;

  /// @brief Get the ID of the oldest keyframe.
  /// \return The ID of the oldest keyframe.
  BundleId oldestKeyframeBundleId() const;

  /**
   * @brief Get the ID of an older frame.
   * @param[in] age age of desired frame. 0 would be the newest frame added to
   *            the state.
   * @return ID of the desired frame or 0 if parameter age was out of range.
   */
  BackendId frameIdByAge(size_t age) const;

  /// @brief Get the ID of the newest frame added to the state.
  /// \return The ID of the current frame.
  BackendId currentFrameId() const;

  /// @brief Get the BundleId of the newest frame added to the state.
  /// \return The BundleId of the current frame.
  BundleId currentBundleId() const;

  /**
   * @brief Get the timestamp for a particular frame.
   * @param[in] nframe_id ID of frame.
   * @return Timestamp of frame.
   */
  double timestamp(BackendId nframe_id) const
  {
    auto slot = states_.findSlot(nframe_id);
    DEBUG_CHECK(slot.second) << "Frame with ID " << nframe_id
                             << " does not exist.";
    if (slot.second)
    {
      return states_.timestamps[slot.first];
    }
    return 0;
  }


  /// @brief get ceres map
  /// return map_ptr The pointer to the ceres_backend::Map.
  std::shared_ptr<ceres_backend::Map>  getMap() const
  {
    return map_ptr_;
  }

  ///@}
  /// @name Setters
  ///@{
  /**
   * @brief Set pose for a given pose ID.
   * @param[in] pose_id ID of the pose that should be changed.
   * @param[in] T_WS new homogeneous transformation.
   * @return True if successful.
   */
  bool set_T_WS(BackendId pose_id, const Transformation& T_WS);

  /**
   * @brief Set pose for a given pose ID.
   * @param[in] landmark_id ID of the landmark that should be changed.
   * @param[in] landmark_pos new landmark position.
   * @return True if successful.
   */
  bool setLandmarkPosition(BackendId landmark_id, const Position& landmark_pos);

  /**
   * @brief Set a landmark constant, used for landmarks from loop closing
   * @param landmark_backend_id BackendId of the landmark.
   * @return True if successful.
   */
  bool setLandmarkConstant(const BackendId &landmark_backend_id);


  /**
   * @brief set all the biases and velocities fixed. Useful when vision is out.
   * @param constant whether to set constant or variable
   * @return True if successful.
   */
  bool changeSpeedAndBiasesFixation(const bool set_fixed);

  /**
   * @brief Set the speeds and IMU biases for a given pose ID.
   * @param[in] pose_id ID of the pose to change speeds and biases for.
   * @param[in] speed_and_bias new speeds and biases.
   * @return True if successful.
   */
  bool setSpeedAndBiasFromNFrameId(BackendId pose_id,
                                   const SpeedAndBias& speed_and_bias);

  /**
   * @brief Set transformation from sensor to camera frame for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in] pose_id ID of the pose to change corresponding camera states for.
   * @param[in] camera_idx Index of camera to set state for.
   * @param[in] T_SCi new homogeneous transformation from sensor (IMU) to camera
   *            frame.
   * @return True if successful.
   */
  bool setCameraSensorStates(BackendId pose_id, uint8_t camera_idx,
                              const Transformation& T_SCi);

  /// @brief Set the landmark initialization state.
  /// @param[in] landmark_id The landmark ID.
  /// @param[in] initialized Whether or not initialised.
  void setLandmarkInitialized(BackendId landmark_id, bool initialized);

  /// @brief Set whether a frame is a keyframe or not.
  /// @param[in] nframe_id The frame bundle ID.
  /// @param[in] is_keyframe Whether or not keyrame.
  void setKeyframe(BackendId nframe_id, bool is_keyframe)
  {
    auto slot = states_.findSlot(nframe_id);
    if (slot.second)
    {
      states_.is_keyframe[slot.first] = is_keyframe;
    }
  }

  /// @brief set ceres map
  /// @param[in] map_ptr The pointer to the ceres_backend::Map.
  void setMap(std::shared_ptr<ceres_backend::Map> map_ptr)
  {
    map_ptr_ = map_ptr;
  }
  ///@}
  ///

  inline bool isPointInEstimator(const int id) const
  {
    return landmarks_map_.find(createLandmarkId(id)) != landmarks_map_.end();
  }

  inline bool isLandmarkInEstimator(const BackendId& id) const
  {
    return landmarks_map_.find(id) != landmarks_map_.end();
  }

  inline bool hasPrior() const
  {
    return marginalization_error_ptr_? true : false;
  }

  void resetPrior();

  inline void checkAndAddToSet(const uint64_t id, std::set<uint64_t>* id_set)
  {
    auto it = std::find(id_set->begin(), id_set->end(), id);
    CHECK(it == id_set->end()) << id_set->size() << ", " << id;
    id_set->insert(id);
  }

  inline void checkAndDeleteFromSet(
      const uint64_t id, std::set<uint64_t>* id_set)
  {
    auto it = std::find(id_set->begin(), id_set->end(), id);
    CHECK(it != id_set->end()) << id_set->size() << ", " << id;
    id_set->erase(it);
  }

  inline void registerFixedFrame(const uint64_t param_id)
  {
    checkAndAddToSet(param_id, &fixed_frame_parameter_ids_);
  }

  inline void deRegisterFixedFrame(const uint64_t param_id)
  {
    checkAndDeleteFromSet(param_id, &fixed_frame_parameter_ids_);
  }

  inline size_t numFixedLandmarks() const
  {
    return fixed_landmark_parameter_ids_.size();
  }

  size_t numValidFixedLandmarks() const;

  inline bool hasFixedPose() const
  {
    return !fixed_frame_parameter_ids_.empty();
  }

  inline void registerFixedLandmark(const uint64_t param_id)
  {
    checkAndAddToSet(param_id, &fixed_landmark_parameter_ids_);
  }

  inline void deRegisterFixedLandmark(const uint64_t param_id)
  {
    checkAndDeleteFromSet(param_id, &fixed_landmark_parameter_ids_);
  }

  inline bool isLandmarkFixed(const uint64_t param_id) const
  {
    return !fixed_landmark_parameter_ids_.empty() &&
        (fixed_landmark_parameter_ids_.find(param_id) !=
        fixed_landmark_parameter_ids_.end());
  }

  void removeAllFixedLandmarks();
  void setAllFixedLandmarksEnabled(const bool enabled);

  void removeLandmarkByBackendId(const BackendId& bid, const bool check_fixed);

  inline void removeLandmarkById(const int lm_id, const bool check_fixed)
  {
    removeLandmarkByBackendId(createLandmarkId(lm_id), check_fixed);
  }

  void updateFixedLandmarks();

  inline bool needPoseFixation() const
  {
    return fixed_landmark_parameter_ids_.size()
        <= min_num_3d_points_for_fixation_;
  }

  // for reinitialization
  bool is_reinit_ = false;
  Eigen::Matrix<double, 9, 1> reinit_speed_bias_;
  Transformation reinit_T_WS_;
  double reinit_timestamp_start_;

  //
  size_t min_num_3d_points_for_fixation_ = 10u;

 private:

  /**
   * @brief Remove an observation from a landmark.
   * @param residual_block_id Residual ID for this landmark.
   * @return True if successful.
   */
  bool removeObservation(ceres::ResidualBlockId residual_block_id);

  // getters
  std::pair<Transformation, bool> getPoseEstimate(BackendId id) const;

  std::pair<SpeedAndBias, bool> getSpeedAndBiasEstimate(BackendId id) const;

  // setters
  bool setPoseEstimate(BackendId id, const Transformation& pose);

  bool setSpeedAndBiasEstimate(BackendId id, const SpeedAndBias& sab);

  CameraBundlePtr camera_rig_;
  //! If we do not estimate extrinsics, then these are the parameter block IDs
  //! for all extrinsics.
  std::vector<BackendId> constant_extrinsics_ids_;
  bool estimate_temporal_extrinsics_ {false};

  // the following keeps track of all states at different times (key=poseId)
  States states_; ///< Buffer for currently considered states.
  std::shared_ptr<ceres_backend::Map> map_ptr_; ///< The underlying svo::Map.

  // the following are updated after the optimization
  PointMap landmarks_map_;
  ///< Contains all the current landmarks (synched after optimisation).

  // parameters
  ExtrinsicsEstimationParametersVec extrinsics_estimation_parameters_;
  ///< Extrinsics parameters.
  std::vector<svo::ImuParameters, Eigen::aligned_allocator<svo::ImuParameters> >
  imu_parameters_; ///< IMU parameters.

  // loss function for reprojection errors
  std::shared_ptr< ceres::LossFunction> cauchy_loss_function_ptr_; ///< Cauchy loss.
  std::shared_ptr< ceres::LossFunction> huber_loss_function_ptr_; ///< Huber loss.

  // the marginalized error term
  std::shared_ptr<ceres_backend::MarginalizationError> marginalization_error_ptr_;
  ///< The marginalization class
  ceres::ResidualBlockId marginalization_residual_id_;
  ///< Remembers the marginalization object's Id

  // ceres iteration callback object
  std::unique_ptr<ceres_backend::CeresIterationCallback> ceres_callback_;
  ///< Maybe there was a callback registered, store it here.

  // fixation
  std::set<uint64_t> fixed_frame_parameter_ids_;
  std::set<uint64_t> fixed_landmark_parameter_ids_;
};

}  // namespace svo

#include <svo/ceres_backend/estimator_impl.hpp>
