// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/depth_estimation.h>
#include <svo/direct/patch_warp.h>
#include <svo/direct/patch_utils.h>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <opencv2/highgui/highgui.hpp>

namespace svo {

DepthEstimator::DepthEstimator(const SolverOptions& solver_options)
  : vk::solver::MiniLeastSquaresSolver<1, double, DepthEstimator>(solver_options)
{ ; }

DepthEstimator::SolverOptions DepthEstimator::getDefaultSolverOptions()
{
  SolverOptions options;
  options.strategy = vk::solver::Strategy::GaussNewton;
  options.max_iter = 10;
  options.eps = 0.00001;
  return options;
}

void DepthEstimator::run(
    const FramePtr& cur_frame,
    const FramePtr& ref_frame,
    const int ref_feature_id)
{
  CHECK_GT(ref_frame->invmu_sigma2_a_b_vec_.cols(), ref_feature_id);
  CHECK_GT(ref_frame->f_vec_.cols(), ref_feature_id);
  CHECK_GT(ref_frame->img_pyr_.size(), static_cast<size_t>(kMaxLevel));
  CHECK_GT(cur_frame->img_pyr_.size(), static_cast<size_t>(kMaxLevel));

  cur_frame_ = cur_frame;
  ref_frame_ = ref_frame;
  f_ref_ = ref_frame_->f_vec_.col(ref_feature_id);
  px_ref_ = ref_frame_->px_vec_.col(ref_feature_id);
  T_cur_ref_ = cur_frame_->T_cam_world() * ref_frame_->T_world_cam();

  double state = ref_frame_->invmu_sigma2_a_b_vec_(0, ref_feature_id);
  for(level_ = kMaxLevel; level_ >= kMinLevel; --level_)
  {
    VLOG(100) << "=== Pyramid Level " << level_ << " ===";
    mu_ = 0.1;
    optimize(state);
  }

  ref_frame_->invmu_sigma2_a_b_vec_(0, ref_feature_id) = state;
}

double DepthEstimator::evaluateError(
    const DepthEstimatorState& inverse_depth,
    HessianMatrix* H,
    GradientVector* g)
{

  const BearingVector f_cur =
      T_cur_ref_.getRotation().rotate(f_ref_) + T_cur_ref_.getPosition()*inverse_depth;
  Keypoint px_cur;
  Eigen::Matrix<double, 2, 3> projection_jacobian;
  cur_frame_->cam_->project3(f_cur, &px_cur, &projection_jacobian);
  if(!cur_frame_->cam_->isKeypointVisibleWithMargin(px_cur, (kPatchHalfsize+3)*level_))
  {
    VLOG(200) << "Depth Estimation: Cur-Patch out of image."
              << " px_cur_pyr = ("  << (px_cur / (1 << level_)).transpose()
              << "), img_size = (" << cur_frame_->img_pyr_[level_].cols
              << " x " << cur_frame_->img_pyr_[level_].rows << ")";
    return 0.0;
  }

  // Compute warped patch.
  warp::AffineTransformation2 A_cur_ref;
  warp::getWarpMatrixAffine(
      ref_frame_->cam(), cur_frame_->cam(), px_ref_, f_ref_,
      1.0/std::max(0.000001, inverse_depth), T_cur_ref_, level_, &A_cur_ref);
  if(!warp::warpAffine(A_cur_ref, ref_frame_->img_pyr_[level_], px_ref_,
                       level_, level_, kPatchHalfsize+1, ref_patch_with_border_))
  {
    VLOG(200) << "Depth Estimation: Ref-Patch out of image:"
              << " px_ref_pyr = "  << (px_ref_ / (1<<level_)).transpose()
              << "), img_size = (" << cur_frame_->img_pyr_[level_].cols
              << " x " << cur_frame_->img_pyr_[level_].rows << ")";
    return 0.0;
  }
  patch_utils::createPatchFromPatchWithBorder(
        ref_patch_with_border_, kPatchSize, ref_patch_);

  uint8_t cur_patch_with_border[(kPatchSize+2)*(kPatchSize+2)] __attribute__ ((aligned (16)));
  Keypoint px_cur_vec = px_cur / (1 << level_);
  warp::createPatchNoWarpInterpolated(
      cur_frame_->img_pyr_[level_], px_cur_vec, kPatchHalfsize+1, cur_patch_with_border);

  if(VLOG_IS_ON(200))
  {
    cv::Mat img_cur_rgb(cur_frame_->img_pyr_[level_].size(), CV_8UC3);
    cv::cvtColor(cur_frame_->img_pyr_[level_], img_cur_rgb, cv::COLOR_GRAY2RGB);
    cv::Mat img_ref_rgb(ref_frame_->img_pyr_[level_].size(), CV_8UC3);
    cv::cvtColor(ref_frame_->img_pyr_[level_], img_ref_rgb, cv::COLOR_GRAY2RGB);
    const Eigen::Vector2d px_ref_vec = px_ref_ / (1 << level_);
    cv::rectangle(img_cur_rgb, cv::Rect(px_cur_vec(0), px_cur_vec(1), kPatchSize+3, kPatchSize+3), cv::Scalar(0,255,1));
    cv::rectangle(img_ref_rgb, cv::Rect(px_ref_vec(0), px_ref_vec(1), kPatchSize+3, kPatchSize+3), cv::Scalar(0,255,1));
    cv::imshow("img_cur_rgb", img_cur_rgb);
    cv::imshow("img_ref_rgb", img_ref_rgb);
    cv::Mat img_cur(kPatchSize+2, kPatchSize+2, CV_8UC1, cur_patch_with_border);
    cv::Mat img_ref(kPatchSize+2, kPatchSize+2, CV_8UC1, ref_patch_with_border_);
    cv::imshow("patch_cur", img_cur);
    cv::imshow("patch_ref", img_ref);
    cv::waitKey(0);
  }

  double chi2 = 0.0;
  const int patch_step = kPatchSize+2;
  for(int y=0; y<kPatchSize; ++y)
  {
    uint8_t* cur_px = (uint8_t*) cur_patch_with_border  + (y+1)*patch_step + 1;
    uint8_t* ref_px = (uint8_t*) ref_patch_with_border_ + (y+1)*patch_step + 1;
    for(int x=0; x<kPatchSize; ++x, ++cur_px, ++ref_px)
    {
      Eigen::Vector2d grad(0.5 * (cur_px[1] - cur_px[-1]),
                           0.5 * (cur_px[patch_step] - cur_px[-patch_step]));
      double residual = *cur_px - *ref_px;
      double J =
          grad.dot(projection_jacobian * T_cur_ref_.getRotation().rotate( f_ref_ * (-1.0 / std::pow(inverse_depth, 2)) ));

      (*H)(0,0) += J*J;
      (*g)(0)   -= J*residual;
      chi2 += residual*residual;
    }
  }

  return chi2;
}

void DepthEstimator::update(
    const DepthEstimatorState& state_old,
    const UpdateVector& dx,
    DepthEstimatorState& state_new)
{
  state_new = state_old + dx(0);
}

bool DepthEstimator::solve(
    const HessianMatrix& H,
    const GradientVector& g,
    UpdateVector& dx)
{
  dx(0) = g(0)/H(0,0);
  return true;
}

} // namespace svo
