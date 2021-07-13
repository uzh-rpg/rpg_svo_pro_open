// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <vikit/solver/mini_least_squares_solver.h>

namespace svo {

class Frame;

/// Depth estimation by minimizing photometric error.
class DepthEstimator : public vk::solver::MiniLeastSquaresSolver<1, double, DepthEstimator>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using SolverOptions = vk::solver::MiniLeastSquaresSolverOptions;
  using DepthEstimatorState = double;

  static constexpr int kPatchHalfsize = 4;
  static constexpr int kPatchSize = 2*kPatchHalfsize;
  static constexpr int kPatchArea = kPatchSize*kPatchSize;
  static constexpr int kMaxLevel = 4;
  static constexpr int kMinLevel = 0;

  DepthEstimator(const SolverOptions& solver_options);
  ~DepthEstimator() = default;

  static SolverOptions getDefaultSolverOptions();

  void run(
      const FramePtr& cur_frame,
      const FramePtr& ref_frame,
      const int ref_feature_id);

  double evaluateError(
      const State& params,
      HessianMatrix* H,
      GradientVector* g);

  void update(
      const State& param_old,
      const UpdateVector& dx,
      State& param_new);

  bool solve(
      const HessianMatrix& H,
      const GradientVector& g,
      UpdateVector& dx);

  FramePtr cur_frame_;
  FramePtr ref_frame_;
  BearingVector f_ref_;
  Keypoint px_ref_;
  Transformation T_cur_ref_;

  uint8_t ref_patch_[kPatchSize*kPatchSize] __attribute__ ((aligned (16)));
  uint8_t ref_patch_with_border_[(kPatchSize+2)*(kPatchSize+2)] __attribute__ ((aligned (16)));

  int level_ = 0;

};

} // namespace svo
