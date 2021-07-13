// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/img_align/sparse_img_align.h>

#include <algorithm>
#include <random> // std::mt19937

#include <opencv2/highgui/highgui.hpp>

#include <vikit/vision.h>
#include <vikit/math_utils.h>

#include <svo/common/logging.h>
#include <svo/common/point.h>
#include <svo/common/camera.h>
#include <svo/common/seed.h>

namespace svo {

SparseImgAlign::SparseImgAlign(
    SolverOptions optimization_options,
    SparseImgAlignOptions options)
  : SparseImgAlignBase(optimization_options,options)
{
  setPatchSize<SparseImgAlign>(4);
}

size_t SparseImgAlign::run(
    const FrameBundle::Ptr& ref_frames,
    const FrameBundle::Ptr& cur_frames)
{
  CHECK(!ref_frames->empty());
  CHECK_EQ(ref_frames->size(), cur_frames->size());

  // Select all visible features and subsample if required.
  fts_vec_.clear();
  size_t n_fts_to_track = 0;
  for(auto frame : ref_frames->frames_)
  {
    std::vector<size_t> fts;
    sparse_img_align_utils::extractFeaturesSubset(
          *frame, options_.max_level, patch_size_with_border_, fts);
    n_fts_to_track += fts.size();
    fts_vec_.push_back(fts);
  }
  SVO_DEBUG_STREAM("Img Align: Tracking "<<n_fts_to_track<<" features.");
  if(n_fts_to_track == 0)
  {
    SVO_ERROR_STREAM("SparseImgAlign: no features to track!");
    return 0;
  }

  // set member variables
  ref_frames_ = ref_frames;
  cur_frames_ = cur_frames;
  T_iref_world_ = ref_frames->at(0)->T_imu_world();

  // prepare caches
  uv_cache_.resize(Eigen::NoChange, n_fts_to_track);
  xyz_ref_cache_.resize(Eigen::NoChange, n_fts_to_track);
  jacobian_proj_cache_.resize(Eigen::NoChange, n_fts_to_track*2);
  jacobian_cache_.resize(Eigen::NoChange, n_fts_to_track*patch_area_);
  residual_cache_.resize(patch_area_, n_fts_to_track);
  visibility_mask_.resize(n_fts_to_track, Eigen::NoChange);
  ref_patch_cache_.resize(patch_area_, n_fts_to_track);

  // the variable to be optimized is the imu-pose of the current frame
  Transformation T_icur_iref =
      cur_frames_->at(0)->T_imu_world() * T_iref_world_.inverse();

  SparseImgAlignState state;
  state.T_icur_iref = T_icur_iref;
  state.alpha = alpha_init_;
  state.beta = beta_init_;

  // precompute values common to all pyramid levels
  size_t feature_counter = 0;
  for(size_t i=0; i<ref_frames_->size(); ++i)
  {
    sparse_img_align_utils::precomputeBaseCaches(
          *ref_frames_->at(i), fts_vec_.at(i),
          options_.use_distortion_jacobian,
          feature_counter, uv_cache_,
          xyz_ref_cache_, jacobian_proj_cache_);
  }

  for(level_=options_.max_level; level_>=options_.min_level; --level_)
  {
    mu_ = 0.1;
    have_cache_ = false; // at every level, recompute the jacobians
    if(solver_options_.verbose)
      printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
    optimize(state);
  }

  // finished, we save the pose in the frame
  for(auto f : cur_frames->frames_)
  {
    f->T_f_w_ = f->T_cam_imu()*state.T_icur_iref*T_iref_world_;
  }

  // reset initial values of illumination estimation TODO: make reset function
  alpha_init_ = 0.0;
  beta_init_ = 0.0;

  return n_fts_to_track;
}

double SparseImgAlign::evaluateError(
    const SparseImgAlignState& state,
    HessianMatrix* H,
    GradientVector* g)
{
  if(!have_cache_) // is reset at every new level.
  {
    size_t feature_counter = 0;
    for(size_t i=0; i<ref_frames_->size(); ++i)
    {
      sparse_img_align_utils::precomputeJacobiansAndRefPatches(
            ref_frames_->at(i), uv_cache_,
            jacobian_proj_cache_, level_, patch_size_,
            fts_vec_.at(i).size(),
            options_.estimate_illumination_gain,
            options_.estimate_illumination_offset,
            feature_counter,
            jacobian_cache_, ref_patch_cache_);
    }
    have_cache_ = true;
  }

  size_t feature_counter = 0; // used to compute the cache index
  for(size_t i=0; i<ref_frames_->size(); ++i)
  {
    const Transformation T_cur_ref =
        cur_frames_->at(i)->T_cam_imu() * state.T_icur_iref
        * ref_frames_->at(i)->T_imu_cam();
    std::vector<Vector2d>* match_pxs = nullptr;
    sparse_img_align_utils::computeResidualsOfFrame(
          cur_frames_->at(i), level_,
          patch_size_, fts_vec_.at(i).size(), T_cur_ref,
          state.alpha, state.beta,
          ref_patch_cache_, xyz_ref_cache_,
          feature_counter, match_pxs, residual_cache_, visibility_mask_);
  }

  float chi2 = sparse_img_align_utils::computeHessianAndGradient(
        jacobian_cache_, residual_cache_,
        visibility_mask_, weight_scale_, weight_function_, H, g);
  return chi2;
}

void  SparseImgAlign::finishIteration()
{
  if(false)
  {
    const size_t cam_index = 0;
    cv::Mat residual_image(
          cur_frames_->at(cam_index)->img_pyr_.at(level_).size(),
          CV_32FC1, cv::Scalar(0.0));
    const size_t nr_features = fts_vec_.at(cam_index).size();
    const FloatType scale = 1.0f/(1<<level_);
    const int patch_size_wb = patch_size_ + 2*border_size_; //patch size with border
    const FloatType patch_center_wb = (patch_size_wb - 1)/2.0f;

    for(size_t i = 0; i < nr_features; ++i)
    {
      if(!visibility_mask_(i))
        continue;

      // compute top left coordinate of patch to be interpolated
      const FloatType u_tl = uv_cache_(0,i)*scale - patch_center_wb;
      const FloatType v_tl = uv_cache_(1,i)*scale - patch_center_wb;

      size_t pixel_counter = 0; // is used to compute the index of the cached residual
      for(int y = 0; y < patch_size_; ++y)
      {
        for(int x = 0; x < patch_size_; ++x, ++pixel_counter)
        {
          residual_image.at<float>(v_tl+y, u_tl+x) =
              (residual_cache_(pixel_counter, i) + 50)/255.0;
        }
      }
    }

    double minval, maxval;
    cv::minMaxLoc(residual_image, &minval, &maxval);
    residual_image = (residual_image - minval) / (maxval - minval);

    std::vector<cv::Mat> residual_image_rgb_vec({cv::Mat(residual_image.size(), CV_32FC1, cv::Scalar(1.0)),
                                                 residual_image,
                                                 cv::Mat(residual_image.size(), CV_32FC1, cv::Scalar(0.0))});
    cv::Mat residual_image_rgb;
    cv::merge(residual_image_rgb_vec, residual_image_rgb);

    cv::imshow("residual_image", residual_image_rgb);
    cv::waitKey(0);
  }
}

namespace sparse_img_align_utils
{

void extractFeaturesSubset(const Frame& ref_frame,
                           const int max_level,
                           const int patch_size_wb, // patch_size + border (usually 2 for gradient),
                           std::vector<size_t>& fts)
{
  // TODO(cfo): add seeds
//  if(fts.size() < 100)
//  {
//    std::unique_lock<decltype(ref_frame->seeds_mut_)> lock(ref_frame->seeds_mut_);
//    size_t n = 0;
//    for(const SeedPtr& seed : ref_frame->seeds_)
//    {
//      if(seed->is_converged_)
//      {
//        fts.push_back(seed->ftr_);
//        ++n;
//      }
//    }
//    SVO_DEBUG_STREAM("SparseImgAlign: add " << n << " additional seeds.");
//  }

  // ignore any feature point, which does not project fully in the image
  // checking on highest level is sufficient.
  const FloatType scale = 1.0f/(1<<max_level);
  const cv::Mat& ref_img = ref_frame.img_pyr_.at(max_level);
  const int rows_minus_two = ref_img.rows - 2;
  const int cols_minus_two = ref_img.cols - 2;
  const FloatType patch_center_wb = (patch_size_wb - 1)/2.0f;

  // check if reference with patch size is within image
  fts.reserve(ref_frame.num_features_);
  for(size_t i = 0; i < ref_frame.num_features_; ++i)
  {
    if(ref_frame.landmark_vec_[i] == nullptr &&
       ref_frame.seed_ref_vec_[i].keyframe == nullptr)
      continue;
    if(isMapPoint(ref_frame.type_vec_[i]))
    {
      continue;
    }
    const FloatType u_tl = ref_frame.px_vec_(0,i)*scale - patch_center_wb;
    const FloatType v_tl = ref_frame.px_vec_(1,i)*scale - patch_center_wb;
    const int u_tl_i = std::floor(u_tl);
    const int v_tl_i = std::floor(v_tl);
    if(!(u_tl_i < 0
         || v_tl_i < 0
         || u_tl_i + patch_size_wb >= cols_minus_two
         || v_tl_i + patch_size_wb >= rows_minus_two))
      fts.push_back(i);
  }
  SVO_DEBUG_STREAM("Img Align: Maximum Number of Features = " << ref_frame.num_features_);
}

void precomputeBaseCaches(const Frame& ref_frame,
                          const std::vector<size_t>& fts,
                          const bool use_distortion_jac,
                          size_t& feature_counter,
                          UvCache& uv_cache,
                          XyzRefCache &xyz_ref_cache,
                          JacobianProjCache& jacobian_proj_cache)
{
  const double focal_length = ref_frame.getErrorMultiplier();
  const Vector3d ref_pos = ref_frame.pos();
  const Transformation& T_imu_cam = ref_frame.T_imu_cam();
  const Transformation& T_cam_imu = ref_frame.T_cam_imu();

  for(const size_t i : fts)
  {
    uv_cache.col(feature_counter) = ref_frame.px_vec_.col(i).cast<FloatType>();

    // evaluate jacobian. cannot just take the 3d points coordinate because of
    // the reprojection errors in the reference image!!!

    FloatType depth = 0;
    if(ref_frame.landmark_vec_[i])
    {
      depth = ((ref_frame.landmark_vec_[i]->pos_ - ref_pos).norm());
    }
    else
    {
      const SeedRef& seed_ref = ref_frame.seed_ref_vec_[i];
      const Position pos = seed_ref.keyframe->T_world_cam()
          * seed_ref.keyframe->getSeedPosInFrame(seed_ref.seed_id);
      depth = (pos - ref_pos).norm();
    }

    const Vector3d xyz_ref(ref_frame.f_vec_.col(i)*depth);
    xyz_ref_cache.col(feature_counter) = xyz_ref.cast<FloatType>();
    const Vector3d xyz_in_imu( T_imu_cam * xyz_ref );

    Matrix<double,2,6> frame_jac;
    if(!use_distortion_jac &&
       ref_frame.cam()->getType() == Camera::Type::kPinhole)
    { // only allow ignoring jacobian for pinhole projection
      Frame::jacobian_xyz2uv_imu(T_cam_imu, xyz_in_imu, frame_jac);
      frame_jac *= focal_length;
    }
    else
    {
      Frame::jacobian_xyz2image_imu(*ref_frame.cam(), T_cam_imu, xyz_in_imu, frame_jac);
      frame_jac *= (-1.0);
    }

    size_t col_index = 2*feature_counter;
    jacobian_proj_cache.col(col_index) = frame_jac.row(0).cast<FloatType>();
    jacobian_proj_cache.col(col_index + 1) = frame_jac.row(1).cast<FloatType>();
    ++feature_counter;
  }
}

void precomputeJacobiansAndRefPatches(
    const FramePtr& ref_frame,
    const UvCache& uv_cache,
    const JacobianProjCache& jacobian_proj_cache,
    const size_t level,
    const int patch_size,
    const size_t nr_features,
    bool estimate_alpha,
    bool estimate_beta,
    size_t& feature_counter,
    JacobianCache& jacobian_cache,
    RefPatchCache& ref_patch_cache)
{
  const cv::Mat& ref_img = ref_frame->img_pyr_.at(level);
  const int stride = ref_img.step; // must be real stride
  const FloatType scale = 1.0f/(1<<level);
  const int patch_area = patch_size*patch_size;
  const int border = 1;
  const int patch_size_wb = patch_size + 2*border; //patch size with border
  const int patch_area_wb = patch_size_wb*patch_size_wb;
  const FloatType patch_center_wb = (patch_size_wb - 1)/2.0f;

  // interpolate patch + border (filled in row major format)
  FloatType interp_patch_array [patch_area_wb];

  for(size_t i = 0; i < nr_features; ++i, ++feature_counter)
  {
    // compute top left coordinate of patch to be interpolated
    const FloatType u_tl = uv_cache(0,feature_counter)*scale - patch_center_wb;
    const FloatType v_tl = uv_cache(1,feature_counter)*scale - patch_center_wb;

    const int u_tl_i = std::floor(u_tl);
    const int v_tl_i = std::floor(v_tl);

    // compute bilateral interpolation weights for reference image
    const FloatType subpix_u_tl = u_tl - u_tl_i;
    const FloatType subpix_v_tl = v_tl - v_tl_i;
    const FloatType wtl = (1.0 - subpix_u_tl) * (1.0 - subpix_v_tl);
    const FloatType wtr = subpix_u_tl * (1.0 - subpix_v_tl);
    const FloatType wbl = (1.0 - subpix_u_tl) * subpix_v_tl;
    const FloatType wbr = subpix_u_tl * subpix_v_tl;
    const int jacobian_proj_col = 2*feature_counter;

    // interpolate patch with border
    size_t pixel_counter = 0;
    for(int y = 0; y < patch_size_wb; ++y)
    {
      // reference image pointer (openCv stores data in row major format)
      uint8_t* r =
          static_cast<uint8_t*>(ref_img.data)
          + (v_tl_i + y)*stride
          + u_tl_i;
      for(int x = 0; x < patch_size_wb; ++x, ++r, ++pixel_counter)
      {
        // precompute interpolated reference patch color
        interp_patch_array[pixel_counter] = wtl*r[0] + wtr*r[1] + wbl*r[stride] + wbr*r[stride+1];
      }
    }

    // fill ref_patch_cache and jacobian_cache
    pixel_counter = 0;
    for(int y = 0; y < patch_size; ++y)
    {
      for(int x = 0; x < patch_size; ++x, ++pixel_counter)
      {
        int offset_center = (x + border) + patch_size_wb*(y + border);
        ref_patch_cache(pixel_counter,feature_counter) = interp_patch_array[offset_center];

        // we use the inverse compositional: thereby we can take the gradient
        // always at the same position.
        const FloatType dx = 0.5f*(interp_patch_array[offset_center + 1]
            -interp_patch_array[offset_center - 1]);
        const FloatType dy = 0.5f*(interp_patch_array[offset_center + patch_size_wb]
            -interp_patch_array[offset_center - patch_size_wb]);

        // cache the jacobian
        int jacobian_col = feature_counter*patch_area + pixel_counter;
        jacobian_cache.block<6,1>(0,jacobian_col) =
            (dx*jacobian_proj_cache.col(jacobian_proj_col) + dy*jacobian_proj_cache.col(jacobian_proj_col + 1))*scale;
        jacobian_cache(6,jacobian_col) = estimate_alpha ? -(interp_patch_array[offset_center]) : 0.0;
        jacobian_cache(7,jacobian_col) = estimate_beta  ? -1.0 : 0.0;
      }
    }
  }
}

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
    std::vector<Vector2d>* match_pxs,
    ResidualCache& residual_cache,
    VisibilityMask& visibility_mask)
{
  const cv::Mat& cur_img = cur_frame->img_pyr_.at(level);
  const int stride = cur_img.step;
  const FloatType scale = 1.0f/(1<<level);
  const int patch_area = patch_size*patch_size;
  const FloatType patch_center = (patch_size - 1)/2.0f;

  FloatType total_intensity = 0.0;

  for(size_t i = 0; i < nr_features; ++i, ++feature_counter)
  {
    Vector3ft xyz_ref = xyz_ref_cache.col(feature_counter);
    const Vector3d xyz_cur(T_cur_ref * xyz_ref.cast<double>());
    if(cur_frame->cam()->getType() ==
         vk::cameras::CameraGeometryBase::Type::kPinhole &&
       xyz_cur.z() < 0.0)
    {
      visibility_mask(feature_counter) = false;
      continue;
    }

    Eigen::Vector2d uv_cur;
    cur_frame->cam()->project3(xyz_cur, &uv_cur);
    const Vector2ft uv_cur_pyr = uv_cur.cast<FloatType>() * scale;

    // compute top left coordinate of patch to be interpolated
    const FloatType u_tl = uv_cur_pyr[0] - patch_center;
    const FloatType v_tl = uv_cur_pyr[1] - patch_center;

    // check if projection is within the image
    if(u_tl  < 0.0
       || v_tl  < 0.0
       || u_tl + patch_size + 2.0 >= cur_img.cols
       || v_tl + patch_size + 2.0 >= cur_img.rows)
    {
      visibility_mask(feature_counter) = false;
      continue;
    }
    else
    {
      visibility_mask(feature_counter) = true;
    }

    const int u_tl_i = std::floor(u_tl);
    const int v_tl_i = std::floor(v_tl);

    // compute bilateral interpolation weights for the current image
    const FloatType subpix_u_tl = u_tl-u_tl_i;
    const FloatType subpix_v_tl = v_tl-v_tl_i;
    const FloatType wtl = (1.0-subpix_u_tl) * (1.0-subpix_v_tl);
    const FloatType wtr = subpix_u_tl * (1.0-subpix_v_tl);
    const FloatType wbl = (1.0-subpix_u_tl) * subpix_v_tl;
    const FloatType wbr = subpix_u_tl * subpix_v_tl;

    size_t pixel_counter = 0; // is used to compute the index of the cached residual
    float total_res = 0.0;
    for(int y = 0; y < patch_size; ++y)
    {
      uint8_t* cur_img_ptr =
          static_cast<uint8_t*>(cur_img.data) + (v_tl_i + y) * stride + u_tl_i;

      for(int x = 0; x < patch_size; ++x, ++pixel_counter, ++cur_img_ptr)
      {
        // compute residual, with bilinear interpolation
        FloatType intensity_cur =
            wtl*cur_img_ptr[0]
            + wtr*cur_img_ptr[1]
            + wbl*cur_img_ptr[stride]
            + wbr*cur_img_ptr[stride+1];
        FloatType res = static_cast<FloatType>(intensity_cur * (1.0+alpha) + beta)
            - ref_patch_cache(pixel_counter, feature_counter);
        residual_cache(pixel_counter,feature_counter) = res;

        // for camera control:
        total_res += intensity_cur - static_cast<float>(ref_patch_cache(pixel_counter,feature_counter));
        total_intensity += intensity_cur;
      }
    }
  }
}

FloatType computeHessianAndGradient(
    const JacobianCache& jacobian_cache,
    const ResidualCache& residual_cache,
    const VisibilityMask& visibility_mask,
    const float weight_scale,
    const vk::solver::WeightFunctionPtr& weight_function,
    SparseImgAlign::HessianMatrix* H,
    SparseImgAlign::GradientVector* g)
{
  float chi2 = 0.0;
  size_t n_meas = 0;
  const size_t patch_area = residual_cache.rows();
  const size_t mask_size = visibility_mask.size();

  for(size_t i = 0; i < mask_size; ++i)
  {
    if(visibility_mask(i)==true)
    {
      size_t patch_offset = i*patch_area;
      for(size_t j = 0; j < patch_area; ++j)
      {
        FloatType res = residual_cache(j,i);

        // Robustification.
        float weight = 1.0;
        if(weight_function)
          weight = weight_function->weight(res/weight_scale);

        chi2 += res*res*weight;
        ++n_meas;

        // Compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error).
        const Vector8ft J_ft = jacobian_cache.col(patch_offset + j);
        const Vector8d J_d = J_ft.cast<double>();
        H->noalias() += J_d*J_d.transpose()*weight;
        g->noalias() -= J_d*res*weight;
      }
    }
  }

  return chi2/n_meas;
}

float computeResidualHessianGradient(const FramePtr& cur_frame,
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
    )
{
  const cv::Mat& cur_img = cur_frame->img_pyr_.at(level);
  const int stride = cur_img.step;
  const float scale = 1.0f/(1<<level);
  const int patch_area = patch_size*patch_size;
  const FloatType patch_center = (patch_size - 1)/2.0f;

  float total_intensity = 0.0;
  float chi2 = 0.0;
  size_t n_meas = 0;

  if(!(H && g))
  {
    return chi2;
  }

  for(size_t i = 0; i < nr_features; ++i, ++feature_counter)
  {
    Vector3ft xyz_ref = xyz_ref_cache.col(feature_counter);
    const Vector3d xyz_cur(T_cur_ref * xyz_ref.cast<double>());
    if(cur_frame->cam()->getType() ==
         vk::cameras::CameraGeometryBase::Type::kPinhole &&
       xyz_cur.z() < 0.0)
    {
      continue;
    }
    Eigen::Vector2d uv_cur;
    cur_frame->cam()->project3(xyz_cur, &uv_cur);
    const Vector2ft uv_cur_pyr = uv_cur.cast<FloatType>() * scale;

    // compute top left coordinate of patch to be interpolated
    const FloatType u_tl = uv_cur_pyr[0] - patch_center;
    const FloatType v_tl = uv_cur_pyr[1] - patch_center;

    // check if projection is within the image
    if(u_tl < 0.0
       || v_tl  < 0.0
       || u_tl + patch_size + 2.0 >= cur_img.cols
       || v_tl + patch_size + 2.0 >= cur_img.rows)
    {
      continue;
    }
    const int u_tl_i = std::floor(u_tl);
    const int v_tl_i = std::floor(v_tl);

    // compute bilateral interpolation weights for the current image
    const FloatType subpix_u_tl = u_tl-u_tl_i;
    const FloatType subpix_v_tl = v_tl-v_tl_i;
    const FloatType wtl = (1.0-subpix_u_tl) * (1.0-subpix_v_tl);
    const FloatType wtr = subpix_u_tl * (1.0-subpix_v_tl);
    const FloatType wbl = (1.0-subpix_u_tl) * subpix_v_tl;
    const FloatType wbr = subpix_u_tl * subpix_v_tl;

    size_t pixel_counter = 0; // is used to compute the index of the cached residual
    float total_res = 0.0;
    size_t patch_offset = feature_counter*patch_area;
    for(int y = 0; y < patch_size; ++y)
    {
      uint8_t* cur_img_ptr = (uint8_t*) cur_img.data + (v_tl_i + y) * stride + u_tl_i;

      for(int x = 0; x < patch_size; ++x, ++pixel_counter, ++cur_img_ptr)
      {
        // compute residual, with bilinear interpolation
        // TODO: check what if we just do nearest neighbour?
        FloatType intensity_cur =
            wtl*cur_img_ptr[0]
            + wtr*cur_img_ptr[1]
            + wbl*cur_img_ptr[stride]
            + wbr*cur_img_ptr[stride+1];
        FloatType res = static_cast<FloatType>(intensity_cur*(1.0+alpha) + beta)
            - ref_patch_cache(pixel_counter,feature_counter);

        // robustification
        float weight = 1.0;
        if(weight_function)
          weight = weight_function->weight(res/weight_scale);

        chi2 += res*res*weight;
        ++n_meas;

        // compute Jacobian, weighted Hessian and weighted "steepest descend images" (times error)
        const Vector8ft J_ft = jacobian_cache.col(patch_offset + pixel_counter);
        const Vector8d J_d = J_ft.cast<double>();
        H->noalias() += J_d * J_d.transpose() * weight;
        g->noalias() -= J_d * res * weight;

        // for camera control:
        total_res += intensity_cur - static_cast<float>(ref_patch_cache(pixel_counter,feature_counter));
        total_intensity += intensity_cur;
      }
    }
  }

  return chi2/n_meas;
}
} // namespace sparse_img_align_utils
} // namespace svo
