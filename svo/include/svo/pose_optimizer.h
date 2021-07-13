// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <fstream>
#include <iostream>
#include <svo/common/frame.h>
#include <svo/global.h>
#include <vikit/solver/mini_least_squares_solver.h>
#include <vikit/solver/robust_cost.h>

namespace svo {

class PoseOptimizer : public vk::solver::MiniLeastSquaresSolver<6, Transformation, PoseOptimizer>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using ScaleEstimator = vk::solver::MADScaleEstimator;
  using RobustWeightFunction = vk::solver::TukeyWeightFunction;
  using SolverOptions = vk::solver::MiniLeastSquaresSolverOptions;
  typedef std::shared_ptr<PoseOptimizer> Ptr;
  typedef Matrix<double,6,6> Matrix6d;
  typedef Matrix<double,2,6> Matrix26d;
  typedef Matrix<double,3,6> Matrix36d;
  typedef Matrix<double,1,6> Matrix16d;
  typedef Matrix<double,6,1> Vector6d;
  enum class ErrorType {kUnitPlane, kBearingVectorDiff, kImagePlane} ;

  struct Statistics {
    double reproj_error_after;
    double reproj_error_before;
    Statistics()
      : reproj_error_after(0.0)
      , reproj_error_before(0.0)
    {}
  } stats_;

  PoseOptimizer(SolverOptions solver_options);
  virtual ~PoseOptimizer() = default;

  static SolverOptions getDefaultSolverOptions();

  /// Optimize frame pose (frame->T_f_w_) by minimizing the reprojection
  /// error w.r.t. to its features.
  size_t run(const FrameBundle::Ptr& frame, double reproj_thresh);

  void setRotationPrior(const Quaternion& R_frame_world, double lambda);

  inline size_t iterCount() const {
    return iter_;
  }

  inline void setErrorType(ErrorType type) {
    err_type_ = type;
  }

  inline void initTracing(const std::string& trace_dir) {
    ofs_reproj_errors_.open((trace_dir+"/reproj_errors.txt").c_str());
  }

//protected:
  FrameBundle::Ptr frame_bundle_;
  double prior_lambda_;
  ScaleEstimator scale_estimator_;
  RobustWeightFunction robust_weight_;
  double measurement_sigma_ = 1.0;
  ErrorType err_type_ = ErrorType::kUnitPlane;
  double focal_length_ = 1.0; /// focal length TODO: should be different for every camera
  std::ofstream ofs_reproj_errors_;

  double evaluateError(
      const Transformation& T_imu_world,
      HessianMatrix* H,
      GradientVector* g);

  double evaluateErrorImpl(
      const Transformation& T_imu_world,
      HessianMatrix* H,
      GradientVector* g,
      std::vector<float>* unwhitened_errors);

  void removeOutliers(
      const double reproj_err_threshold,
      Frame* frame,
      std::vector<double>* reproj_errors,
      size_t* n_deleted_edges,
      size_t* n_deleted_corners);

  void update(
      const State& T_frameold_from_world,
      const UpdateVector& dx,
      State& T_framenew_from_world);

  virtual void applyPrior(const State& current_model);
};

namespace pose_optimizer_utils {

void calculateFeatureResidualUnitPlane(
    const Eigen::Ref<const BearingVector>& f,
    const Position& xyz_in_world,
    const Transformation& T_imu_world,
    const Transformation& T_cam_imu,
    double measurement_sigma,
    const PoseOptimizer::RobustWeightFunction &robust_weight,
    double* unwhitened_error,
    double* chi2_error,
    PoseOptimizer::HessianMatrix* H,
    PoseOptimizer::GradientVector* g);

void calculateFeatureResidualImagePlane(
    const Eigen::Ref<const Keypoint>& px,
    const Position& xyz_in_world,
    const Transformation& T_imu_world,
    const Transformation& T_cam_imu,
    const svo::Camera& cam,
    double measurement_sigma,
    const PoseOptimizer::RobustWeightFunction &robust_weight,
    double* unwhitened_error,
    double* chi2_error,
    PoseOptimizer::HessianMatrix* H,
    PoseOptimizer::GradientVector* g);

void calculateFeatureResidualBearingVectorDiff(
    const Eigen::Ref<const BearingVector>& f,
    const Position& xyz_in_world,
    const Transformation& T_imu_world,
    const Transformation& T_cam_imu,
    double measurement_sigma,
    const PoseOptimizer::RobustWeightFunction &robust_weight,
    double* unwhitened_error,
    double* chi2_error,
    PoseOptimizer::HessianMatrix* H,
    PoseOptimizer::GradientVector* g);

void calculateEdgeletResidualUnitPlane(
    const Eigen::Ref<const BearingVector>& f,
    const Position& xyz_in_world,
    const Eigen::Ref<const GradientVector>& grad,
    const Transformation& T_imu_world,
    const Transformation& T_cam_imu,
    double measurement_sigma,
    const PoseOptimizer::RobustWeightFunction& robust_weight,
    double* unwhitened_error,
    double* chi2_error,
    PoseOptimizer::HessianMatrix* H,
    PoseOptimizer::GradientVector* g);

void calculateEdgeletResidualImagePlane(
    const Eigen::Ref<const Keypoint>& px,
    const Position& xyz_in_world,
    const Eigen::Ref<const GradientVector>& grad,
    const Transformation& T_imu_world,
    const Transformation& T_cam_imu,
    const svo::Camera& cam,
    double measurement_sigma,
    const PoseOptimizer::RobustWeightFunction& robust_weight,
    double* unwhitened_error,
    double* chi2_error,
    PoseOptimizer::HessianMatrix* H,
    PoseOptimizer::GradientVector* g);

void calculateEdgeletResidualBearingVectorDiff(
    const Eigen::Ref<const Keypoint>& px,
    const Eigen::Ref<const BearingVector>& f,
    const Position& xyz_in_world,
    const Eigen::Ref<const GradientVector>& grad,
    const Transformation& T_imu_world,
    const Transformation& T_cam_imu,
    const svo::Camera& cam,
    double measurement_sigma,
    const PoseOptimizer::RobustWeightFunction& robust_weight,
    double* unwhitened_error,
    double* chi2_error,
    PoseOptimizer::HessianMatrix* H,
    PoseOptimizer::GradientVector* g);


} // namespace pose_optimizer_utils

} // namespace svo
