// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/img_align/sparse_img_align_base.h>

#include <vikit/solver/robust_cost.h>
#include <vikit/performance_monitor.h>

#include <svo/common/types.h>
#include <svo/common/frame.h>

namespace svo {

typedef Eigen::Matrix<FloatType, 2, Eigen::Dynamic, Eigen::ColMajor> UvCache;
typedef Eigen::Matrix<FloatType, 3, Eigen::Dynamic, Eigen::ColMajor> XyzRefCache;
typedef Eigen::Matrix<FloatType, 6, Eigen::Dynamic, Eigen::ColMajor> JacobianProjCache;
typedef Eigen::Matrix<FloatType, 8, Eigen::Dynamic, Eigen::ColMajor> JacobianCache;
typedef Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ResidualCache;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1, Eigen::ColMajor> VisibilityMask;
typedef Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> RefPatchCache;

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlign : public SparseImgAlignBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<SparseImgAlign> Ptr;

public:
  SparseImgAlign(
      SolverOptions optimization_options,
      SparseImgAlignOptions options);

  void setPatchSizeSideEffects()
  {
    // no side effects
  }

  size_t run(const FrameBundle::Ptr& ref_frames,
             const FrameBundle::Ptr& cur_frames);

private:
  // caches:
  bool have_cache_ = false;
  std::vector< std::vector<size_t> > fts_vec_;
  UvCache uv_cache_; //<! containts feature coordinates in reference image (size 2 x #Patches)
  XyzRefCache xyz_ref_cache_; //!< contains 3d feature location in IMU frame (size 3 x #Patches)
  JacobianProjCache jacobian_proj_cache_; //!< containts 2x6 Jacobians (d proj(X))/(d xi) (size 2 x 6 x #Patches)
  JacobianCache jacobian_cache_; //<! contains 1x8 jacobians (pose and illumination model) (size 8 x AreaPatch*#Patches)
  ResidualCache residual_cache_; //<! residuals (size AreaPatch x #Patches)
  VisibilityMask visibility_mask_;//<! is Patch visible in current image? (size 1 x #Patches)
  RefPatchCache ref_patch_cache_; //<! residuals (size AreaPatch x #Patches)

protected:
  /// Warp the (cur)rent image such that it aligns with the (ref)erence image
  double evaluateError(
      const SparseImgAlignState &state,
      HessianMatrix* H,
      GradientVector* g);

  void update(
      const SparseImgAlignState& old_model,
      const UpdateVector& dx,
      SparseImgAlignState& new_model);

  void applyPrior(const SparseImgAlignState& current_model);

  virtual void finishIteration();
};

namespace sparse_img_align_utils {

void extractFeaturesSubset(
    const Frame& ref_frame,
    const int max_level,
    const int patch_size_wb,             // patch size + border (usually border = 2 for gradiant)
    std::vector<size_t>& fts);

// Fills UvCache (needed for extraction of refpatch extraction at every pyramid level),
// XyzRefCache (needed at every optimization step for reprojection) and
// JacobianProjCache (needed at every pyramid level)
void precomputeBaseCaches(
    const Frame& ref_frame,
    const std::vector<size_t>& fts,
    const bool use_pinhole_distortion,
    size_t& feature_counter,
    UvCache& uv_cache,
    XyzRefCache& xyz_ref_cache,
    JacobianProjCache& jacobian_proj_cache);

// Fills JacobianCache and RefPatchCache at every level and sets have_cache_ to true
void precomputeJacobiansAndRefPatches(
    const FramePtr& ref_frame,
    const UvCache& uv_cache,
    const JacobianProjCache& jacobian_proj_cache,
    const size_t level,
    const int patch_size,
    const size_t nr_features,
    bool estimate_alpha, bool estimate_beta,
    size_t& feature_counter,
    JacobianCache& jacobian_cache,
    RefPatchCache& ref_patch_cache);

// Fills ResidualCache and VisibilityMask
void computeResidualsOfFrame(
    const FramePtr& cur_frame,
    const size_t level,
    const int patch_size,
    const size_t nr_features,
    const Transformation& T_cur_ref,
    const float alpha,
    const float beta,
    const RefPatchCache& ref_patch_cache,
    const XyzRefCache& xyz_ref_cache,
    size_t& feature_counter,
    std::vector<Vector2d>* match_px,
    ResidualCache& residual_cache,
    VisibilityMask& visibility_mask
    );

// Compute Hessian and gradient
FloatType computeHessianAndGradient(
    const JacobianCache& jacobian_cache,
    const ResidualCache& residual_cache,
    const VisibilityMask& visibility_mask,
    const float weight_scale,
    const vk::solver::WeightFunctionPtr& weight_function,
    SparseImgAlign::HessianMatrix* H,
    SparseImgAlign::GradientVector* g);


// Experimental. Computes residuals and Hessian at the same time. No significant speedup was observed.
float computeResidualHessianGradient(
    const FramePtr& cur_frame,
    const size_t level,
    const int patch_size,
    const size_t nr_features,
    const Transformation& T_cur_ref,
    const float alpha,
    const float beta,
    const RefPatchCache& ref_patch_cache,
    const XyzRefCache& xyz_ref_cache,
    const JacobianCache& jacobian_cache,
    const float weight_scale,
    const vk::solver::WeightFunctionPtr& weight_function,
    SparseImgAlign::HessianMatrix* H,
    SparseImgAlign::GradientVector* g,
    size_t& feature_counter
    );
} // namespace sparse_img_align_utils
} // namespace svo
