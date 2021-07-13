// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <vikit/solver/mini_least_squares_solver.h>
#include <vikit/solver/robust_cost.h>
#include <vikit/performance_monitor.h>

#include <svo/common/types.h>
#include <svo/common/frame.h>

namespace svo {

//using namespace Eigen;
using Eigen::Vector2i;
using Eigen::Vector2f;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix2f;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix;

typedef Eigen::Matrix<double, 8, 8> Matrix8d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<FloatType,2,1> Vector2ft;
typedef Eigen::Matrix<FloatType,3,1> Vector3ft;
typedef Eigen::Matrix<FloatType,8,1> Vector8ft;

/// Sparse Image Alignment Options
struct SparseImgAlignOptions
{
  int max_level = 4;
  int min_level = 1;
  bool estimate_illumination_gain = false;
  bool estimate_illumination_offset = false;
  bool use_distortion_jacobian = false;
  bool robustification = false;
  double weight_scale = 10;
};

/// State to be estimated
struct SparseImgAlignState
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Transformation T_icur_iref;     ///< Relative transformation between last and current frame
  double alpha = 0.0;  ///< Affine illumination model multiplicative parameter
  double beta = 0.0;   ///< Affine illumination model additive parameter
  // TODO: if we have multiple frames, we should have alpha&beta for every frame individually
};

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlignBase : public vk::solver::MiniLeastSquaresSolver<8, SparseImgAlignState, SparseImgAlignBase>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<SparseImgAlignBase> Ptr;
  typedef vk::solver::MiniLeastSquaresSolverOptions SolverOptions;

  SparseImgAlignBase(
      SolverOptions optimization_options,
      SparseImgAlignOptions options);

  virtual size_t run(const FrameBundle::Ptr& ref_frames,
             const FrameBundle::Ptr& cur_frames) = 0;

  static SolverOptions getDefaultSolverOptions();

  void setWeightedPrior(
      const Transformation& T_cur_ref_prior,
      const double alpha_prior,
      const double beta_prior,
      const double lambda_rot,
      const double lambda_trans,
      const double lambda_alpha,
      const double lambda_beta);

  template<class derived>
  void setPatchSize(size_t patch_size)
  {
    patch_size_ = patch_size;
    border_size_ = 1;
    patch_size_with_border_ = patch_size_ + 2*border_size_;
    patch_area_ = patch_size_*patch_size_;
    static_cast<derived*>(this)->setPatchSizeSideEffects();
  }

  /// The derived class has to specify the necessary modifications (e.g. memory allocation)
  /// in case the patchsize is changed.
  virtual void setPatchSizeSideEffects() = 0;

  /// Set number of features used for the alignment. If a number is <= 0 all features are used
  /// Specifying a number gives you the control about the performance of the alignment.
  /// Having an alignment running over 600 or 100 features doesn't make that much a difference
  inline void setMaxNumFeaturesToAlign(int num)
  {
    max_num_features_ = num;
  }

  /// Set inital value for illumination estimation multiplicative parameter
  inline void setAlphaInitialValue(double alpha_init)
  {
    alpha_init_ = alpha_init;
  }

  /// Set inital value for illumination estimation additive parameter
  inline void setBetaInitialValue(double beta_init)
  {
    beta_init_ = beta_init;
  }

  /// Set compensation
  inline void setCompensation(const bool do_compensation)
  {
    options_.estimate_illumination_gain = do_compensation;
    options_.estimate_illumination_offset = do_compensation;
  }

  /// Warp the (cur)rent image such that it aligns with the (ref)erence image
  virtual double evaluateError(
      const SparseImgAlignState &state,
      HessianMatrix* H,
      GradientVector* g) = 0;

  void update(
      const SparseImgAlignState& old_model,
      const UpdateVector& dx,
      SparseImgAlignState& new_model);

  void applyPrior(const SparseImgAlignState& current_model);

  virtual void finishIteration() {}

  int patch_size_;
  int border_size_;
  int patch_size_with_border_;
  int patch_area_;
  static constexpr int kJacobianSize = 8;
  static constexpr int kHessianTriagN = 36; // Nr elements of the upper triangular part of the Hessian

protected:
  SparseImgAlignOptions options_;
  FrameBundle::Ptr ref_frames_; //!< reference frame, has depth for gradient pixels.
  FrameBundle::Ptr cur_frames_; //!< only the image is known!
  Transformation T_iref_world_; //!< Pose of reference frame's IMU in world coordinates
  int level_; //!< current pyramid level on which the optimization runs.
  double prior_lambda_rot_;
  double prior_lambda_trans_;
  double prior_lambda_alpha_;
  double prior_lambda_beta_;
  int max_num_features_ = -1; //!< number of features used for alignment (if -1, all are used)
  double alpha_init_ = 0.0;
  double beta_init_ = 0.0;
  vk::solver::WeightFunctionPtr weight_function_;
  double weight_scale_;
};

} // namespace svo
