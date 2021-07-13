// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/img_align/sparse_img_align_base.h>

#include <algorithm>
#include <random> // std::mt19937

#include <opencv2/highgui/highgui.hpp>

#include <vikit/vision.h>
#include <vikit/math_utils.h>

#include <svo/common/logging.h>
#include <svo/common/point.h>
#include <svo/direct/depth_filter.h>


namespace svo {

SparseImgAlignBase::SparseImgAlignBase(
    SolverOptions optimization_options,
    SparseImgAlignOptions options)
  : vk::solver::MiniLeastSquaresSolver<8, SparseImgAlignState, SparseImgAlignBase>(optimization_options)
  , options_(options)
  , weight_function_(options_.robustification? (new vk::solver::TukeyWeightFunction):nullptr)
  , weight_scale_(options_.weight_scale)
{}

SparseImgAlignBase::SolverOptions SparseImgAlignBase::getDefaultSolverOptions()
{
  SolverOptions options;
  options.strategy = vk::solver::Strategy::GaussNewton;
  options.max_iter = 10;
  options.eps = 0.0005;
  return options;
}

void SparseImgAlignBase::setWeightedPrior(
    const Transformation& T_cur_ref_prior,
    const double alpha_prior,
    const double beta_prior,
    const double lambda_rot,
    const double lambda_trans,
    const double lambda_alpha,
    const double lambda_beta)
{
  prior_lambda_rot_ = lambda_rot;
  prior_lambda_trans_ = lambda_trans;
  prior_lambda_alpha_ = lambda_alpha;
  prior_lambda_beta_ = lambda_beta;
  SparseImgAlignState state;
  state.T_icur_iref = T_cur_ref_prior;
  state.alpha = alpha_prior;
  state.beta = beta_prior;
  setPrior(state, Matrix8d::Zero());
}

void SparseImgAlignBase::update(
    const SparseImgAlignState& state_old,
    const UpdateVector& dx,
    SparseImgAlignState& state_new)
{
  state_new.T_icur_iref =  state_old.T_icur_iref * Transformation::exp(-dx.head<6>());
  state_new.alpha = (state_old.alpha - dx(6))/(1.0+dx(6));
  state_new.beta  = (state_old.beta  - dx(7))/(1.0+dx(6));

  // we need to normalize from time to time otherwise rounding errors sum up
  state_new.T_icur_iref.getRotation().toImplementation().normalize();
}

void SparseImgAlignBase::applyPrior(const SparseImgAlignState& state)
{
  if(iter_ == 0)
  {
    double H_max_diag_trans = 0;
    for(size_t j=0; j<3; ++j)
      H_max_diag_trans = std::max(H_max_diag_trans, std::fabs(H_(j,j)));
    Matrix3d I_trans = Matrix3d::Identity()*prior_lambda_trans_*H_max_diag_trans;

    double H_max_diag_rot = 0;
    for(size_t j=3; j<6; ++j)
      H_max_diag_rot = std::max(H_max_diag_rot, std::fabs(H_(j,j)));
    Matrix3d I_rot = Matrix3d::Identity()*prior_lambda_rot_*H_max_diag_rot;

    const double I_alpha = prior_lambda_alpha_*H_(6,6);
    const double I_beta  = prior_lambda_beta_ *H_(7,7);

    I_prior_ = Matrix8d::Zero();
    I_prior_.block<3,3>(0,0) = I_trans;
    I_prior_.block<3,3>(3,3) = I_rot;
    I_prior_(6,6) = I_alpha;
    I_prior_(7,7) = I_beta;
  }

  H_.noalias() += I_prior_;
  g_.head<6>() += I_prior_.block<6,6>(0,0)*Transformation::log(prior_.T_icur_iref.inverse()*state.T_icur_iref);

  // TODO: this is just a placeholder. derive correct solution
  g_(6) += I_prior_(6,6)*(prior_.alpha - state.alpha);
  g_(7) += I_prior_(7,7)*(prior_.beta  - state.beta);
}

} // namespace svo
