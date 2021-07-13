// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/matcher.h>

#include <cstdlib>
#include <random>
#include <chrono>

#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <kindr/minimal/angle-axis.h>

#include <svo/direct/patch_warp.h>
#include <svo/direct/patch_score.h>
#include <svo/direct/patch_utils.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/feature_wrapper.h>
#include <svo/direct/feature_alignment.h>
#include <svo/common/camera.h>
#include <svo/common/logging.h>

namespace svo {

Matcher::MatchResult Matcher::findMatchDirect(
    const Frame& ref_frame,
    const Frame& cur_frame,
    const FeatureWrapper& ref_ftr,
    const FloatType& ref_depth,
    Keypoint& px_cur)
{
  Eigen::Vector2i pxi = ref_ftr.px.cast<int>()/(1<<ref_ftr.level);
  int boundary = kHalfPatchSize+2;
  if(pxi[0] < boundary
     || pxi[1] < boundary
     || pxi[0] >= static_cast<int>(ref_frame.cam()->imageWidth()/(1<<ref_ftr.level))-boundary
     || pxi[1] >= static_cast<int>(ref_frame.cam()->imageHeight()/(1<<ref_ftr.level))-boundary)
    return MatchResult::kFailVisibility;

  // warp affine
  warp::getWarpMatrixAffine(
      ref_frame.cam_, cur_frame.cam_, ref_ftr.px, ref_ftr.f, ref_depth,
      cur_frame.T_cam_world() * ref_frame.T_world_cam(), ref_ftr.level, &A_cur_ref_);
  search_level_ = warp::getBestSearchLevel(A_cur_ref_, ref_frame.img_pyr_.size()-1);
/*
  if(pt.normal_set_)
  {
    warp::getWarpMatrixAffineHomography(
        *ref_ftr.frame->cam_,
        *cur_frame.cam_,
        ref_ftr.px,
        ref_ftr.f,
        ref_ftr.frame->T_f_w_.rotation_matrix()*pt.normal_,
        (pt.pos_-ref_ftr.frame->pos()).norm(),
        cur_frame.T_f_w_*ref_ftr.frame->T_f_w_.inverse(),
        ref_ftr.level,
        A_cur_ref_);
  }
  else
*/
  if(options_.use_affine_warp_)
  {
    if(!warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                         ref_ftr.level, search_level_, kHalfPatchSize+1, patch_with_border_))
      return MatchResult::kFailWarp;
  }
  else
  {
    // pixelwise warp:
    // TODO(zzc): currently using the search level from affine, good enough?
    if(!warp::warpPixelwise(cur_frame, ref_frame, ref_ftr,
                           ref_ftr.level, search_level_, kHalfPatchSize+1, patch_with_border_))
      return MatchResult::kFailWarp;

  }
  patch_utils::createPatchFromPatchWithBorder(
        patch_with_border_, kPatchSize, patch_);

  // px_cur should be set
  Keypoint px_scaled(px_cur/(1<<search_level_));
  Keypoint px_scaled_start(px_scaled);

  //bool success = false;
  if(isEdgelet(ref_ftr.type))
  {
    GradientVector dir_cur(A_cur_ref_ * ref_ftr.grad);
    dir_cur.normalize();
    if(feature_alignment::align1D(
         cur_frame.img_pyr_[search_level_], dir_cur, patch_with_border_,
         patch_, options_.align_max_iter,
         options_.affine_est_offset_, options_.affine_est_gain_,
         &px_scaled, &h_inv_))
    {
      if ((px_scaled - px_scaled_start).norm() > 
          options_.max_patch_diff_ratio * kPatchSize)
      {
        return MatchResult::kFailTooFar;
      }
      px_cur = px_scaled * (1<<search_level_);
      // set member variables with results (used in reprojector)
      px_cur_ = px_cur;
      cur_frame.cam()->backProject3(px_cur_, &f_cur_);
      f_cur_.normalize();
      return MatchResult::kSuccess;
    }
  }
  else
  {
    std::vector<Eigen::Vector2f>* last_fail_steps = nullptr;
    bool res = feature_alignment::align2D(
          cur_frame.img_pyr_[search_level_], patch_with_border_, patch_,
          options_.align_max_iter,
          options_.affine_est_offset_, options_.affine_est_gain_,
          px_scaled, false, last_fail_steps);
    if(res)
    {
      if ((px_scaled - px_scaled_start).norm() > 
          options_.max_patch_diff_ratio * kPatchSize)
      {
        return MatchResult::kFailTooFar;
      }
      px_cur = px_scaled * (1<<search_level_);
      // set member variables with results (used in reprojector)
      px_cur_ = px_cur;
      cur_frame.cam()->backProject3(px_cur_, &f_cur_);
      f_cur_.normalize();
      return MatchResult::kSuccess;
    }
    else
    {
      VLOG(300) << "NOT CONVERGED: search level " << search_level_;
    }
  }
  return MatchResult::kFailAlignment;
}

Matcher::MatchResult Matcher::findEpipolarMatchDirect(
    const Frame& ref_frame,
    const Frame& cur_frame,
    const FeatureWrapper& ref_ftr,
    const double d_estimate_inv,
    const double d_min_inv,
    const double d_max_inv,
    double& depth)
{
  Transformation T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse();
  return findEpipolarMatchDirect(ref_frame, cur_frame, T_cur_ref, ref_ftr,
                                 d_estimate_inv, d_min_inv, d_max_inv, depth);
}

Matcher::MatchResult Matcher::findEpipolarMatchDirect(
    const Frame& ref_frame,
    const Frame& cur_frame,
    const Transformation& T_cur_ref,
    const FeatureWrapper& ref_ftr,
    const double d_estimate_inv,
    const double d_min_inv,
    const double d_max_inv,
    double& depth)
{
  int zmssd_best = PatchScore::threshold();

  // Compute start and end of epipolar line in old_kf for match search, on image plane
  const BearingVector A = T_cur_ref.getRotation().rotate(ref_ftr.f) + T_cur_ref.getPosition()*d_min_inv;
  const BearingVector B = T_cur_ref.getRotation().rotate(ref_ftr.f) + T_cur_ref.getPosition()*d_max_inv;
  Eigen::Vector2d px_A, px_B;
  cur_frame.cam()->project3(A, &px_A);
  cur_frame.cam()->project3(B, &px_B);
  epi_image_ = px_A - px_B;

  // Compute affine warp matrix
  warp::getWarpMatrixAffine(
      ref_frame.cam_, cur_frame.cam_, ref_ftr.px, ref_ftr.f,
      1.0/std::max(0.000001, d_estimate_inv), T_cur_ref, ref_ftr.level, &A_cur_ref_);

  // feature pre-selection
  reject_ = false;
  if(isEdgelet(ref_ftr.type) && options_.epi_search_edgelet_filtering)
  {
    const Eigen::Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
    const double cosangle = fabs(grad_cur.dot(epi_image_.normalized()));
    if(cosangle < options_.epi_search_edgelet_max_angle)
    {
      reject_ = true;
      return MatchResult::kFailAngle;
    }
  }

  // prepare for match
  //    - find best search level
  //    - warp the reference patch
  search_level_ = warp::getBestSearchLevel(A_cur_ref_, ref_frame.img_pyr_.size()-1);
  // length and direction on SEARCH LEVEL
  epi_length_pyramid_ = epi_image_.norm() / (1<<search_level_);
  GradientVector epi_dir_image = epi_image_.normalized();
  if(!warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                       ref_ftr.level, search_level_, kHalfPatchSize+1, patch_with_border_))
    return MatchResult::kFailWarp;
  patch_utils::createPatchFromPatchWithBorder(
        patch_with_border_, kPatchSize, patch_);

  // Case 1: direct search locally if the epipolar line is too short
  if(epi_length_pyramid_ < 2.0)
  {
    px_cur_ = (px_A+px_B)/2.0;
    MatchResult res = findLocalMatch(cur_frame, epi_dir_image, search_level_, px_cur_);
    if(res != MatchResult::kSuccess)
      return res;
    cur_frame.cam()->backProject3(px_cur_, &f_cur_);
    f_cur_.normalize();
    return matcher_utils::depthFromTriangulation(T_cur_ref, ref_ftr.f, f_cur_, &depth);
  }

  // Case 2: search along the epipolar line for the best match
  PatchScore patch_score(patch_); // precompute for reference patch
  BearingVector C = T_cur_ref.getRotation().rotate(ref_ftr.f) + T_cur_ref.getPosition()*d_estimate_inv;
  scanEpipolarLine(cur_frame, A, B, C, patch_score, search_level_, &px_cur_, &zmssd_best);

  // check if the best match is good enough
  if(zmssd_best < PatchScore::threshold())
  {
    if(options_.subpix_refinement)
    {
      MatchResult res = findLocalMatch(cur_frame, epi_dir_image, search_level_, px_cur_);
      if(res != MatchResult::kSuccess)
        return res;
    }

    cur_frame.cam()->backProject3(px_cur_, &f_cur_);
    f_cur_.normalize();
    return matcher_utils::depthFromTriangulation(T_cur_ref, ref_ftr.f, f_cur_, &depth);
  }
  else
    return MatchResult::kFailScore;
}

std::string Matcher::getResultString(const Matcher::MatchResult& result)
{
  std::string result_str = "success";
  switch(result)
  {
    case MatchResult::kFailScore: result_str = "fail score"; break;
    case MatchResult::kFailTriangulation: result_str = "fail triangulation"; break;
    case MatchResult::kFailVisibility: result_str = "fail visibility"; break;
    case MatchResult::kFailWarp: result_str = "fail warp"; break;
    case MatchResult::kFailAlignment: result_str = "fail alignment"; break;
    case MatchResult::kFailRange: result_str = "fail range"; break;
    case MatchResult::kFailAngle: result_str = "fail angle"; break;
    case MatchResult::kFailCloseView: result_str = "fail close view"; break;
    case MatchResult::kFailLock: result_str = "fail lock"; break;
    default: result_str = "unknown";
  }
  return result_str;
}

Matcher::MatchResult Matcher::findLocalMatch(
    const Frame& frame,
    const Eigen::Ref<GradientVector>& direction,
    const int patch_level,
    Keypoint& px_cur)
{
  Keypoint px_scaled(px_cur/(1<<patch_level));
  bool res;
  if(options_.align_1d)
    res = feature_alignment::align1D(
          frame.img_pyr_[patch_level], direction,
          patch_with_border_, patch_,
          options_.align_max_iter,
          options_.affine_est_offset_, options_.affine_est_gain_,
          &px_scaled, &h_inv_);
  else
    res = feature_alignment::align2D(
        frame.img_pyr_[patch_level], patch_with_border_, patch_,
        options_.align_max_iter,
        options_.affine_est_offset_, options_.affine_est_gain_,
        px_scaled);

  if(!res)
    return MatchResult::kFailAlignment;

  px_cur = px_scaled*(1<<patch_level);
  return MatchResult::kSuccess;
}


bool Matcher::updateZMSSD(
    const Frame& frame,
    const Eigen::Vector2i& pxi,
    const int patch_level,
    const PatchScore& patch_score,
    int* zmssd_best)
{
    // TODO interpolation would probably be a good idea
    uint8_t* cur_patch_ptr = frame.img_pyr_[patch_level].data
                             + (pxi[1]-kHalfPatchSize)*frame.img_pyr_[patch_level].step
                             + (pxi[0]-kHalfPatchSize);
    int zmssd = patch_score.computeScore(cur_patch_ptr, frame.img_pyr_[patch_level].step);

    if(zmssd < *zmssd_best)
    {
      *zmssd_best = zmssd;
      return true;
    }
    else
      return false;
}

bool Matcher::isPatchWithinImage(
    const Frame& frame,
    const Eigen::Vector2i& pxi,
    const int patch_level)
{
  return !(pxi[0] < kPatchSize || pxi[1] < kPatchSize
      || pxi[0] >= (static_cast<int>(frame.cam()->imageWidth()/(1<<patch_level))-kPatchSize)
      || pxi[1] >= (static_cast<int>(frame.cam()->imageHeight()/(1<<patch_level))-kPatchSize));
}

void Matcher::scanEpipolarLine(
    const Frame& frame,
    const Eigen::Vector3d& A,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& C,
    const PatchScore& patch_score,
    const int patch_level,
    Keypoint* image_best,
    int* zmssd_best)
{
  if(options_.scan_on_unit_sphere)
    scanEpipolarUnitSphere(frame, A, B, C, patch_score, patch_level, image_best, zmssd_best);
  else
    scanEpipolarUnitPlane(frame, A, B, C, patch_score, patch_level, image_best, zmssd_best);
}

void Matcher::scanEpipolarUnitPlane(
    const Frame& frame,
    const Eigen::Vector3d& A,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& C,
    const PatchScore& patch_score,
    const int patch_level,
    Keypoint* image_best,
    int* zmssd_best)
{
  // if there're too many steps, we only search for a limited range around the center
  //    while keeping the step size small enough to check each pixel on the image plane
  size_t n_steps = epi_length_pyramid_/0.7; // one step per pixel
  Eigen::Vector2d step = (vk::project2(A) - vk::project2(B))/n_steps;
  if(n_steps > options_.max_epi_search_steps)
  {
    /* TODO
    printf("WARNING: skip epipolar search: %d evaluations, px_lenght=%f, d_min=%f, d_max=%f.\n",
           n_steps, epi_length_, d_min_inv, d_max_inv);
    */
    n_steps = options_.max_epi_search_steps;
  }
  // now we sample along the epipolar line
  Eigen::Vector2d uv_C = vk::project2(C);
  Eigen::Vector2d uv = uv_C;
  Eigen::Vector2d uv_best = uv;
  bool forward = true;
  Eigen::Vector2i last_checked_pxi(0,0);

  for(size_t i=0; i<n_steps; ++i, uv+=step)
  {
    Eigen::Vector2d px;
    frame.cam()->project3(vk::unproject2d(uv), &px);
    Eigen::Vector2i pxi(px[0]/(1<<patch_level)+0.5,
                 px[1]/(1<<patch_level)+0.5); // +0.5 to round to closest int

    if(pxi == last_checked_pxi)
      continue;
    last_checked_pxi = pxi;

    // check if the patch is full within the new frame
    if(!isPatchWithinImage(frame, pxi, patch_level))
    {
      // change search direction if pixel is out of field of view
      if(forward)
      {
        // reverse search direction
        i=n_steps*0.5;
        step = -step;
        uv = uv_C;
        forward=false;
        continue;
      }
      else
        break;
    }

    if(updateZMSSD(frame, pxi, patch_level, patch_score, zmssd_best))
      uv_best = uv;

    if(forward && i > n_steps*0.5)
    {
      // reverse search direction
      step = -step;
      uv = uv_C;
      forward=false;
    }
  }

  // convert uv_best to image coordinates
  Eigen::Vector2d projected;
  frame.cam()->project3(vk::unproject2d(uv_best), &projected);
  *image_best = projected.cast<svo::FloatType>();
}

void Matcher::scanEpipolarUnitSphere(
    const Frame& frame,
    const Eigen::Vector3d& A,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& C,
    const PatchScore& patch_score,
    const int patch_level,
    Keypoint* image_best,
    int* zmssd_best)
{
  size_t n_steps = epi_length_pyramid_ / 0.7; // TODO(zzc): better way of doing this?
  n_steps = n_steps>options_.max_epi_search_steps? options_.max_epi_search_steps:n_steps;
  size_t half_steps = n_steps / 2;

  // calculate the step in angle
  Eigen::Vector3d f_A = A.normalized();
  Eigen::Vector3d f_B = B.normalized();
  double step = std::acos(f_A.dot(f_B))/ n_steps;

  // calculate the rotation axis: positive angle -> toward A
  kindr::minimal::AngleAxis rotation_B_to_A;
  rotation_B_to_A.setAxis((f_B.cross(f_A)).normalized());

  // search around center
  Eigen::Vector3d f_C = C.normalized();
  Eigen::Vector3d f = f_C;
  Eigen::Vector3d f_best = f_C;
  Eigen::Vector2i last_checked_pxi(0, 0);
  for(size_t i=0; i<n_steps; i++)
  {
    // TODO(zzc): more compact
    // rotation angle w.r.t. f_C
    double angle = 0.0;
    if(i < half_steps) // f_A <-- f_C
      angle = i*step;
    else
      angle = (i-half_steps) * (-step); // f_C --> f_B
    rotation_B_to_A.setAngle(angle);

    // current sample on unit sphere
    f = rotation_B_to_A.rotate(f_C);

    // back project to image plane
    Eigen::Vector2d px;
    frame.cam()->project3(f, &px);
    Eigen::Vector2i pxi(px[0]/(1<<patch_level)+0.5,
                 px[1]/(1<<patch_level)+0.5); // +0.5 to round to closest int
    if(pxi == last_checked_pxi)
      continue;
    last_checked_pxi = pxi;

    // is within image?
    // TODO(zzc): FIX use visibility check in camera model
    if(!isPatchWithinImage(frame, pxi, patch_level))
    {
      if(i<half_steps) // f_A <-- f_C to f_C --> f_B
      {
        i = half_steps;
        continue;
      }
      else             // end of search
        break;
    }

    // update ZMSSD
    if(updateZMSSD(frame, pxi, patch_level, patch_score, zmssd_best))
      f_best = f;
  }

  //backproject to image plane
  Eigen::Vector2d projected;
  frame.cam()->project3(f_best, &projected);
  *image_best = projected.cast<svo::FloatType>();
}

namespace matcher_utils {

Matcher::MatchResult depthFromTriangulation(
    const Transformation& T_search_ref,
    const Eigen::Vector3d& f_ref,
    const Eigen::Vector3d& f_cur,
    double* depth)
{
  Eigen::Matrix<double,3,2> A; A << T_search_ref.getRotation().rotate(f_ref), f_cur;
  const Eigen::Matrix2d AtA = A.transpose()*A;
  if(AtA.determinant() < 0.000001)
    return Matcher::MatchResult::kFailTriangulation;
  const Eigen::Vector2d depth2 = - AtA.inverse()*A.transpose()*T_search_ref.getPosition();
  (*depth) = std::fabs(depth2[0]);
  return Matcher::MatchResult::kSuccess;
}





} // namespace matcher_utils
} // namespace svo
