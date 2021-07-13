// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

// svo
#include <svo/global.h>
#include <svo/vio_common/backend_types.hpp>

// forward declarations
class Transformation;

namespace svo
{
/// EXPERIMENTAL Defines interface for various bundle adjustment methods
class AbstractBundleAdjustment
{
public:
  typedef std::shared_ptr<AbstractBundleAdjustment> Ptr;

  /// Default constructor.
  AbstractBundleAdjustment()
  {
  }

  virtual ~AbstractBundleAdjustment()
  {
  }

  // no copy
  AbstractBundleAdjustment(const AbstractBundleAdjustment&) = delete;
  AbstractBundleAdjustment& operator=(const AbstractBundleAdjustment&) = delete;

  /// Invoke bundle adjustment.
  virtual void bundleAdjustment(const FrameBundlePtr& frame_bundle) = 0;

  /// Update map with results from bundle adjustment.
  virtual void loadMapFromBundleAdjustment(const FrameBundlePtr& new_frames,
                                           const FrameBundlePtr& last_frames,
                                           const MapPtr& map,
                                           bool& have_motion_prior) = 0;

  /// Reset bundle adjustment
  virtual void reset() = 0;

  /// Bundle adjustment can run completely in parallel. Start the thread to do
  /// so.
  virtual void startThread() = 0;

  /// Stop and join the bundle adjustment thread
  virtual void quitThread() = 0;

  virtual void setPerformanceMonitor(const std::string& trace_dir) = 0;
  virtual void startTimer(BundleId bundle_id) = 0;

  /**
   * @brief Get the pose and speed bias of the IMU as per the latest IMU frame
   * @param[in] bundle_id of the latest frame
   * @param[out] The speed bias and pose of the latest imu frame
   */
  virtual void getLatestSpeedBiasPose(
      Eigen::Matrix<double, 9, 1>* speed_bias,
      Transformation* T_WS, double* timestamp) const = 0;

  BundleAdjustmentType getType() const
  {
    return type_;
  }

  virtual int getNumFrames() const = 0;

  virtual void setReinitStartValues(const Eigen::Matrix<double, 9, 1>& sb,
                                    const Transformation& Tws,
                                    const double timestamp) = 0;

  virtual void setCorrectionInWorld(const Transformation& w_T_correction) = 0;

  virtual void getAllActiveKeyframes(std::vector<FramePtr>* keyframes) = 0;

  virtual bool isFixedToGlobalMap() const = 0;

  virtual BundleId lastOptimizedBundleId() const = 0;

  virtual void getLastState(ViNodeState* state) const = 0;

protected:
  BundleAdjustmentType type_ = BundleAdjustmentType::kNone;
};

}  // namespace svo
