// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/feature_alignment.h>

#ifdef __SSE2__
# include <emmintrin.h>
#endif
#ifdef __ARM_NEON__
# include <arm_neon.h>
#endif

#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <svo/direct/patch_utils.h>

namespace svo {
namespace feature_alignment {

#define SUBPIX_VERBOSE 0
#define SVO_DISPLAY_ALIGN_1D 0

//------------------------------------------------------------------------------
bool align1D(
    const cv::Mat& cur_img,
    const Eigen::Ref<GradientVector>& dir,                  // direction in which the patch is allowed to move
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    const bool affine_est_offset,
    const bool affine_est_gain,
    Keypoint* cur_px_estimate,
    double* h_inv)
{
  CHECK_NOTNULL(cur_px_estimate);

  constexpr int kHalfPatchSize = 4;
  constexpr int kPatchSize = 2 * kHalfPatchSize;
  constexpr int kPatchArea = kPatchSize * kPatchSize;
  bool converged = false;

  // We optimize feature position and two affine parameters.
  // Compute derivative of template and prepare inverse compositional.
  float __attribute__((__aligned__(16))) ref_patch_dv[kPatchArea];
  Eigen::Matrix3f H = Eigen::Matrix3f::Zero(3, 3);

  // Compute gradient and hessian.
  constexpr int ref_step = kPatchSize + 2;
  float* it_dv = ref_patch_dv;
  for(int y=0; y<kPatchSize; ++y)
  {
    uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
    for(int x=0; x<kPatchSize; ++x, ++it, ++it_dv)
    {
      Eigen::Vector3f J;
      const float dx = static_cast<float>(it[1]) - static_cast<float>(it[-1]);
      const float dy = static_cast<float>(it[ref_step]) - static_cast<float>(it[-ref_step]);
      J[0] = 0.5f * (dir(0) * dx + dir(1) * dy);

      // If not using the affine compensation, set the jacobian be zero.
      // In this way, all the blocks related to affine parameters will be zero.
      J[1] = affine_est_offset? 1.0f : 0.0f;
      J[2] = affine_est_gain? -1.0f*it[0] : 0.0f;

      *it_dv = J[0];
      H += J*J.transpose();
    }
  }
  // If not use affine compensation, force update to be zero by
  // * setting the affine parameter block in H to identity
  // * setting the residual block to zero (see below)
  if(!affine_est_offset)
  {
    H(1, 1) = 1.0;
  }
  if(!affine_est_gain)
  {
    H(2, 2) = 1.0;
  }

  if(h_inv)
    *h_inv = 1.0/H(0,0)*kPatchSize*kPatchSize;
  Eigen::Matrix3f Hinv = H.inverse();
  float mean_diff = 0;
  float alpha = 1.0;

  // Compute pixel location in new image:
  float u = cur_px_estimate->x();
  float v = cur_px_estimate->y();

  // termination condition
  const float min_update_squared = 0.03*0.03;
  const int cur_step = cur_img.step.p[0];
  #if SVO_DISPLAY_ALIGN_1D
  cv::Mat res_patch(kPatchSize, kPatchSize, CV_32FC1);
  cv::Mat cur_patch(kPatchSize, kPatchSize, CV_32FC1);
  #endif
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = std::floor(u);
    int v_r = std::floor(v);
    if(u_r < kHalfPatchSize
       || v_r < kHalfPatchSize
       || u_r >= cur_img.cols - kHalfPatchSize
       || v_r >= cur_img.rows - kHalfPatchSize)
      break;

    if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    uint8_t* it_ref = ref_patch;
    float* it_ref_dv = ref_patch_dv;
    float new_chi2 = 0.0;
    Eigen::Vector3f Jres = Eigen::Vector3f::Zero();
    for(int y=0; y<kPatchSize; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data +
          (v_r+y-kHalfPatchSize)*cur_step + u_r-kHalfPatchSize;
      for(int x=0; x<kPatchSize; ++x, ++it, ++it_ref, ++it_ref_dv)
      {
        float cur_intensity =
            wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = cur_intensity - alpha*(*it_ref) + mean_diff;
        Jres[0] -= res*(*it_ref_dv);

        // If affine compensation is used,
        // set Jres with respect to affine parameters.
        if(affine_est_offset)
        {
          Jres[1] -= res;
        }
        if(affine_est_gain)
        {
          Jres[2] -= (-1)*res*(*it_ref);
        }
        new_chi2 += res*res;
        #if SVO_DISPLAY_ALIGN_1D
        res_patch.at<float>(y,x) = res;
        cur_patch.at<float>(y,x) = cur_intensity;
        #endif
      }
    }
    // If not using affine compensation, force update to be zero.
    if(!affine_est_offset)
    {
      Jres[1] = 0.0;
    }
    if(!affine_est_gain)
    {
      Jres[2] = 0.0;
    }

    Eigen::Vector3f update = Hinv * Jres;
    u += update[0]*dir[0];
    v += update[0]*dir[1];
    mean_diff += update[1];
    alpha += update[2];

    VLOG(300) << "It. " << iter << ": \t"
              << "\t u=" << u << ", v=" << v
              << "\t update = "
              << update[0] << ", " << update[1] << ", " << update[2]
              << "\t new chi2 = " << new_chi2;

    #if SVO_DISPLAY_ALIGN_1D
    cv::Mat res_patch_normalized, cur_patch_normalized, ref_patch_img, ref_patch_normalized;
    patch_utils::normalizeAndUpsamplePatch(res_patch, 8, &res_patch_normalized);
    patch_utils::normalizeAndUpsamplePatch(cur_patch, 8, &cur_patch_normalized);
    patch_utils::patchToMat(ref_patch, kPatchSize, &ref_patch_img);
    patch_utils::normalizeAndUpsamplePatch(ref_patch_img, 8, &ref_patch_normalized);
    cv::Mat concatenated;
    patch_utils::concatenatePatches({ref_patch_normalized,
                                     cur_patch_normalized,
                                     res_patch_normalized}, &concatenated);
    cv::line(concatenated,
             cv::Point2f(concatenated.rows/2, concatenated.rows/2),
             cv::Point2f(concatenated.rows/2+15*dir(0), concatenated.rows/2 +15*dir(1)),
             cv::Scalar(0,0,255), 3);
    cv::imshow("concatenated", concatenated);
    cv::waitKey(0);
    #endif

    if(update[0]*update[0] < min_update_squared)
    {
      VLOG(300) << "converged.";
      converged = true;
      break;
    }
  }

  *cur_px_estimate << u, v;
  return converged;
}

//------------------------------------------------------------------------------
bool align2D(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    const bool affine_est_offset,
    const bool affine_est_gain,
    Keypoint& cur_px_estimate,
    bool no_simd,
    std::vector<Eigen::Vector2f> *each_step)
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border,
                        ref_patch, n_iter, cur_px_estimate);
#endif

  if(each_step) each_step->clear();

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // We optimize feature position and two affine parameters.
  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Eigen::Matrix4f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y)
  {
    uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Eigen::Vector4f J;
      J[0] = 0.5 * (it[1] - it[-1]);
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]);

      // If not using the affine compensation, force the jacobian to be zero.
      J[2] = affine_est_offset? 1.0 : 0.0;
      J[3] = affine_est_gain? -1.0*it[0]: 0.0;

      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose();
    }
  }
  // If not use affine compensation, force update to be zero by
  // * setting the affine parameter block in H to identity
  // * setting the residual block to zero (see below)
  if(!affine_est_offset)
  {
    H(2, 2) = 1.0;
  }
  if(!affine_est_gain)
  {
    H(3, 3) = 1.0;
  }
  Eigen::Matrix4f Hinv = H.inverse();
  float mean_diff = 0;
  float alpha = 1.0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  if(each_step) each_step->push_back(Eigen::Vector2f(u, v));

  // termination condition
  const float min_update_squared = 0.03*0.03; // TODO I suppose this depends on the size of the image (ate)
  const int cur_step = cur_img.step.p[0];
  //float chi2 = 0;
  Eigen::Vector4f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = std::floor(u);
    int v_r = std::floor(v);
    if(u_r < halfpatch_size_
       || v_r < halfpatch_size_
       || u_r >= cur_img.cols-halfpatch_size_
       || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    uint8_t* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    //float new_chi2 = 0.0;
    Eigen::Vector4f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_;
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = search_pixel - alpha*(*it_ref) + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);

        // If affine compensation is used,
        // set Jres with respect to affine parameters.
        if(affine_est_offset)
        {
          Jres[2] -= res;
        }

        if(affine_est_gain)
        {
          Jres[3] -= (-1)*res*(*it_ref);
        }
        //new_chi2 += res*res;
      }
    }
    // If not use affine compensation, force update to be zero.
    if(!affine_est_offset)
    {
      Jres[2] = 0.0;
    }
    if(!affine_est_gain)
    {
      Jres[3] = 0.0;
    }
    /*
    if(iter > 0 && new_chi2 > chi2)
    {
#if SUBPIX_VERBOSE
      cout << "error increased." << endl;
#endif
      u -= update[0];
      v -= update[1];
      break;
    }
    chi2 = new_chi2;
*/
    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];
    alpha += update[3];

    if(each_step) each_step->push_back(Eigen::Vector2f(u, v));

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  (void)no_simd;

  return converged;
}

#define  DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n)) // rounds to closest integer and descales

//------------------------------------------------------------------------------
bool align2D_SSE2(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Keypoint& cur_px_estimate)
{
  // TODO: This function should not be used as the alignment is not robust to illumination changes!
  const int halfpatch_size = 4;
  const int patch_size = 8;
  const int patch_area = 64;
  bool converged=false;
  const int W_BITS = 14;

  // compute derivative of template and prepare inverse compositional
  int16_t __attribute__((__aligned__(16))) ref_patch_dx[patch_area];
  int16_t __attribute__((__aligned__(16))) ref_patch_dy[patch_area];

  // compute gradient and hessian
  const int ref_step = patch_size+2;
  int16_t* it_dx = ref_patch_dx;
  int16_t* it_dy = ref_patch_dy;
  float A11=0, A12=0, A22=0;
  for(int y=0; y<patch_size; ++y)
  {
    uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
    for(int x=0; x<patch_size; ++x, ++it, ++it_dx, ++it_dy)
    {
      int16_t dx = static_cast<int16_t>(it[1]) - it[-1];
      int16_t dy = static_cast<int16_t>(it[ref_step]) - it[-ref_step];
      *it_dx = dx;
      *it_dy = dy;  // we are missing a factor 1/2
      A11 += static_cast<float>(dx*dx); // we are missing a factor 1/4
      A12 += static_cast<float>(dx*dy);
      A22 += static_cast<float>(dy*dy);
    }
  }

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;
#ifdef __SSE2__
  const int cur_step = cur_img.step.p[0];
#endif
  const float Dinv = 1.0f/(A11*A22 - A12*A12); // we are missing an extra factor 16
  float chi2 = 0;
  float update_u = 0, update_v = 0;

  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = std::floor(u);
    int v_r = std::floor(v);
    if(u_r < halfpatch_size || v_r < halfpatch_size || u_r >= cur_img.cols-halfpatch_size || v_r >= cur_img.rows-halfpatch_size)
      break;

    if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float b1=0, b2=0;
    float new_chi2 = 0.0;

#ifdef __SSE2__
    // compute bilinear interpolation weights
    int wTL = static_cast<int>((1.0f-subpix_x)*(1.0f-subpix_y)*(1 << W_BITS));
    int wTR = static_cast<int>(subpix_x * (1.0f-subpix_y)*(1 << W_BITS));
    int wBL = static_cast<int>((1.0f-subpix_x)*subpix_y*(1 << W_BITS));
    int wBR = (1 << W_BITS) - wTL - wTR - wBL;

    __m128i qw0 = _mm_set1_epi32(wTL + (wTR << 16)); // Sets the 4 signed 32-bit integer values to [wTL, wTR].
    __m128i qw1 = _mm_set1_epi32(wBL + (wBR << 16));
    __m128i z = _mm_setzero_si128();
    __m128 qb0 = _mm_setzero_ps(); // 4 floats
    __m128 qb1 = _mm_setzero_ps(); // 4 floats
    __m128i qdelta = _mm_set1_epi32(1 << (W_BITS-1));
    for(int y=0; y<patch_size; ++y)
    {
      const uint8_t* it  = (const uint8_t*) cur_img.data + (v_r+y-halfpatch_size)*cur_step + u_r-halfpatch_size;

      // Iptr is aligned!
      //__m128i diff0 = _mm_load_si128((const __m128i*)(ref_patch + y*8));
      __m128i diff = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(ref_patch + y*8)), z);

      // load the lower 64 bits and unpack [8u 0 8u 0..]
      __m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(it)), z);
      __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(it + 1)), z);
      __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(it + cur_step)), z);
      __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(it + cur_step + 1)), z);

      // interpolate top row and bottom row
      // _mm_unpacklo_epi16: Interleaves the lower 4 signed or unsigned 16-bit integers in a with the lower 4 signed or unsigned 16-bit integers in b.
      // _mm_madd_epi16: Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
      __m128i t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                 _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
      __m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                 _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));

      // _mm_srai_epi32: Shifts the 4 signed 32-bit integers in A right by count bits while shifting in the sign bit.
      t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS); // adding qdelta is for rounding closest int
      t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS);

      // compute 8xres:
      // _mm_packs_epi32: Packs the 8 signed 32-bit integers from a and b into signed 16-bit integers and saturates.
      diff = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff);

      // load gradient dX and dY, both are aligned!
      v00 = _mm_load_si128((const __m128i*)(ref_patch_dx + y*patch_size)); // [dx1, dx2, dx3, dx4 ...]
      v01 = _mm_load_si128((const __m128i*)(ref_patch_dy + y*patch_size)); // [dy1, dy2, dy3, dy4 ...]

      // _mm_mulhi_epi16: Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
      v10 = _mm_mullo_epi16(v00, diff); // Packs the lower 16 bits of the 8 signed 32-bit results. [15:0]
      v11 = _mm_mulhi_epi16(v00, diff); // Packs the upper 16 bits of the 8 signed 32-bit results. [31:16]

      // _mm_unpacklo_epi16: Interleaves the lower 4 signed or unsigned 16-bit integers with the lower 4 signed or unsigned 16-bit integers in b.
      v00 = _mm_unpacklo_epi16(v10, v11);
      v10 = _mm_unpackhi_epi16(v10, v11);

      // convert to float and add to dx
      qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
      qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v10));

      // same with dY
      v10 = _mm_mullo_epi16(v01, diff);
      v11 = _mm_mulhi_epi16(v01, diff);
      v00 = _mm_unpacklo_epi16(v10, v11);
      v10 = _mm_unpackhi_epi16(v10, v11);
      qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v00));
      qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
    }

    float __attribute__((__aligned__(16))) buf[4];
    _mm_store_ps(buf, qb0);
    b1 += buf[0]+buf[1]+buf[2]+buf[3];
    _mm_store_ps(buf, qb1);
    b2 += buf[0]+buf[1]+buf[2]+buf[3];
#endif

    // compute -A^-1*b
    update_u = ((A12*b2 - A22*b1) * Dinv)*2; // * 2 to compensate because above, we did not compute the derivative correctly
    update_v = ((A12*b1 - A11*b2) * Dinv)*2;
    u += update_u;
    v += update_v;

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update_u << ", " << update_v
         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update_u*update_u+update_v*update_v < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
    chi2 = new_chi2;
    (void)chi2;
    (void)subpix_x;
    (void)subpix_y;
    (void)ref_patch;
  }

  cur_px_estimate << u, v;
  (void)W_BITS;
  return converged;
}

//------------------------------------------------------------------------------
bool align2D_NEON (const cv::Mat& cur_img,
                   uint8_t* ref_patch_with_border,
                   uint8_t* ref_patch,
                   const int n_iter,
                   Keypoint& cur_px_estimate)
{
  const int halfpatch_size = 4;
  const int patch_size = 8;
  const int patch_area = 64;
  bool converged=false;
  const int W_BITS = 14;

  // compute derivative of template and prepare inverse compositional
  int16_t __attribute__((__aligned__(16))) ref_patch_dx[patch_area];
  int16_t __attribute__((__aligned__(16))) ref_patch_dy[patch_area];

  // compute gradient and hessian
  const int ref_step = patch_size+2;
  int16_t* it_dx = ref_patch_dx;
  int16_t* it_dy = ref_patch_dy;
  Eigen::Matrix3f H; H.setZero();
  for(int y=0; y<patch_size; ++y)
  {
    uint8_t* it = ref_patch_with_border + (y+1)*ref_step + 1;
    for(int x=0; x<patch_size; ++x, ++it, ++it_dx, ++it_dy)
    {
      *it_dx = static_cast<int16_t>(it[1]) - it[-1];
      *it_dy = static_cast<int16_t>(it[ref_step]) - it[-ref_step]; // divide by 2 missing
      Eigen::Vector3f J(*it_dx, *it_dy, 1.0f);
      H += J*J.transpose();
    }
  }
  Eigen::Matrix3f Hinv = H.inverse();
  float mean_diff = 0.0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;
  const int cur_step = cur_img.step.p[0];
  Eigen::Vector3f update;
  Eigen::Vector3f Jres;
  for(int iter = 0; iter<n_iter; ++iter)
  {
    const int u_r = std::floor(u);
    const int v_r = std::floor(v);
    if(u_r < halfpatch_size || v_r < halfpatch_size || u_r >= cur_img.cols-halfpatch_size || v_r >= cur_img.rows-halfpatch_size)
      break;

    if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    const float subpix_x = u-u_r;
    const float subpix_y = v-v_r;
    float b1=0, b2=0;

#ifdef __ARM_NEON__
    const int SHIFT_BITS = 7;
    const uint16_t wTL = static_cast<uint16_t>((1.0f-subpix_x)*(1.0f-subpix_y)*(1<<SHIFT_BITS));
    const uint16_t wTR = static_cast<uint16_t>(subpix_x*(1.0f-subpix_y)*(1<<SHIFT_BITS));
    const uint16_t wBL = static_cast<uint16_t>((1.0f-subpix_x)*subpix_y*(1<<SHIFT_BITS));
    const uint16_t wBR = static_cast<uint16_t>((1 << SHIFT_BITS) - wTL - wTR - wBL);

    // initialize result to zero
    int32x4_t vb1 = vdupq_n_s32(0);
    int32x4_t vb2 = vdupq_n_s32(0);
    int16x8_t vmean_diff = vdupq_n_s16( (int16_t) (mean_diff+0.5) );
    int16x8_t vres_sum = vdupq_n_s16(0);
    for(int y=0; y<patch_size; ++y)
    {
      const uint8_t* it  = (const uint8_t*) cur_img.data + (v_r+y-halfpatch_size)*cur_step + u_r-halfpatch_size;

      // load and convert from uint8 to uint16
      uint16x8_t v00 = vmovl_u8( vld1_u8( it ) );
      uint16x8_t v01 = vmovl_u8( vld1_u8( it + 1 ) );
      uint16x8_t v10 = vmovl_u8( vld1_u8( it + cur_step ) );
      uint16x8_t v11 = vmovl_u8( vld1_u8( it + cur_step + 1 ) );

      // vector multiply by scalar
      v00 = vmulq_n_u16( v00, wTL );
      v01 = vmulq_n_u16( v01, wTR );
      v10 = vmulq_n_u16( v10, wBL );
      v11 = vmulq_n_u16( v11, wBR );

      // add all results together
      v00 = vaddq_u16( v00, vaddq_u16( v01, vaddq_u16( v10, v11 ) ) );

      // descale: shift right by constant
      v00 = vrshrq_n_u16(v00, SHIFT_BITS);

      // compute difference between reference and interpolated patch,
      // use reinterpet-cast to make signed [-255,255]
      int16x8_t res = vsubq_s16(vreinterpretq_s16_u16(v00), vreinterpretq_s16_u16(vmovl_u8(vld1_u8( ref_patch + y*8 ))));

      // correct res with mean difference
      res = vaddq_s16(res, vmean_diff);

      // compute sum of the residual
      vres_sum = vaddq_s16(vres_sum, res);

      // Eigen::Vector multiply accumulate long: vmla -> Vr[i] := Va[i] + Vb[i] * Vc[i]
      // int32x4_t  vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c);    // VMLAL.S16 q0,d0,d0
      int16x8_t grad = vld1q_s16(ref_patch_dx + y*patch_size);
      vb1 = vmlal_s16(vb1, vget_low_s16(grad), vget_low_s16(res));
      vb1 = vmlal_s16(vb1, vget_high_s16(grad), vget_high_s16(res));

      grad = vld1q_s16(ref_patch_dy + y*patch_size);
      vb2 = vmlal_s16(vb2, vget_low_s16(grad), vget_low_s16(res));
      vb2 = vmlal_s16(vb2, vget_high_s16(grad), vget_high_s16(res));
    }

    // finally, sum results of vb1, vb2 and vres_sum
    int32x2_t tmp;
    tmp = vpadd_s32(vget_low_s32(vb1), vget_high_s32(vb1));
    Jres[0] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);

    tmp = vpadd_s32(vget_low_s32(vb2), vget_high_s32(vb2));
    Jres[1] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);

    int32x4_t vres_sum1 = vpaddlq_s16(vres_sum);
    tmp = vpadd_s32(vget_low_s32(vres_sum1), vget_high_s32(vres_sum1));
    Jres[2] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);
#endif

    update = Hinv * Jres * 2; // * 2 to compensate because above, we did not compute the derivative correctly
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1] << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }

    (void)subpix_x;
    (void)subpix_y;
    (void)b1;
    (void)b2;
    (void)cur_step;
  }

  cur_px_estimate << u, v;

  (void)W_BITS;
  (void)ref_patch;
  return converged;
}

//------------------------------------------------------------------------------
void alignPyr2DVec(
    const std::vector<cv::Mat>& img_pyr_ref,
    const std::vector<cv::Mat>& img_pyr_cur,
    const int max_level,
    const int min_level,
    const std::vector<int>& patch_sizes,
    const int n_iter,
    const float min_update_squared,
    const std::vector<cv::Point2f>& px_ref,
    std::vector<cv::Point2f>& px_cur,
    std::vector<uint8_t>& status)
{
  for(size_t i=0; i<px_ref.size(); ++i)
  {
    Eigen::Vector2i px_ref_level_0(px_ref[i].x, px_ref[i].y);
    Keypoint px_cur_level_0(px_cur[i].x, px_cur[i].y);
    bool res = alignPyr2D(
          img_pyr_ref, img_pyr_cur, max_level, min_level, patch_sizes,
          n_iter, min_update_squared, px_ref_level_0, px_cur_level_0);
    status[i] = res ? 1 : 0;
    px_cur[i] = cv::Point2f(px_cur_level_0[0], px_cur_level_0[1]);

    // TODO: add subpixel of rounding px_ref_level_0.
    //       typically not a problem because ref comes directly from feature
    //       extraction and is int.
  }
}

//------------------------------------------------------------------------------
bool alignPyr2D(
    const std::vector<cv::Mat>& img_pyr_ref,
    const std::vector<cv::Mat>& img_pyr_cur,
    const int max_level,
    const int min_level,
    const std::vector<int>& patch_sizes,
    const int n_iter,
    const float min_update_squared,
    const Eigen::Vector2i& px_ref_level_0,
    Keypoint& px_cur_level_0)
{
#define SVO_PYRAMIDAL_VERBOSE 0
#define SVO_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#if SVO_PYRAMIDAL_VERBOSE
  std::cout << "--------" << std::endl;
#endif

  // compute derivative of template and prepare inverse compositional
  const int max_patch_area = patch_sizes[0]*patch_sizes[0];
  uint8_t __attribute__((__aligned__(16))) ref_patch[max_patch_area];
  int16_t __attribute__((__aligned__(16))) ref_patch_dx[max_patch_area];
  int16_t __attribute__((__aligned__(16))) ref_patch_dy[max_patch_area];
  bool converged = false;

  for(int level = max_level; level >= min_level; --level)
  {
    const int patch_size = patch_sizes[level];
    CHECK(patch_size % 8 == 0)
        << ": alignPyr2D is only made for patch sizes multiples of 8!";
    const int halfpatch_size = patch_size/2;
    const int scale = (1<<level);
    const cv::Mat& img_ref = img_pyr_ref[level];
    const cv::Mat& img_cur = img_pyr_cur[level];
    const int width = img_ref.cols;
    const int height = img_ref.rows;
    const int step = img_ref.step; // should be number of bytes of each row
    const Eigen::Vector2f px_ref_flt = px_ref_level_0.cast<float>() / scale - Eigen::Vector2f(halfpatch_size, halfpatch_size);
    const Eigen::Vector2i px_ref = px_ref_flt.cast<int>();
    const Eigen::Vector2f px_ref_offset = px_ref_flt - px_ref.cast<float>();

    if(px_ref[0] < 1 || px_ref[1] < 1
       || px_ref[0] >= width-patch_size-1 || px_ref[1] >= height-patch_size-1)
    {
#if SVO_PYRAMIDAL_VERBOSE
      std::cout << "reference pixel is too close to the border" << std::endl;
#endif
      continue;
    }

    // compute gradient and hessian
    uint8_t* it_patch = ref_patch;
    int16_t* it_dx = ref_patch_dx;
    int16_t* it_dy = ref_patch_dy;
    Eigen::Matrix2f H; H.setZero();
    for(int y=0; y<patch_size; ++y)
    {
      uint8_t* it = img_ref.data + (px_ref[1]+y)*step + (px_ref[0]);
      for(int x=0; x<patch_size; ++x, ++it, ++it_patch, ++it_dx, ++it_dy)
      {
        *it_patch = *it;
        *it_dx = static_cast<int16_t>(it[1]) - it[-1];
        *it_dy = static_cast<int16_t>(it[step]) - it[-step]; // divide by 2 missing
        Eigen::Vector2f J(*it_dx, *it_dy);
        H += J*J.transpose();
      }
    }
    Eigen::Matrix2f Hinv = H.inverse();

    // Compute pixel location in new image:
    float u = px_cur_level_0[0] / scale - halfpatch_size - px_ref_offset[0];
    float v = px_cur_level_0[1] / scale - halfpatch_size - px_ref_offset[1];
    Eigen::Vector2f update; update.setZero();
    bool go_to_next_level = false;
    const int SHIFT_BITS = 7;
    converged = false; // reset at every level
    for(int iter = 0; iter<n_iter; ++iter)
    {
      if(std::isnan(u) || std::isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      {
#if SVO_PYRAMIDAL_VERBOSE
        std::cout << "update is NaN" << std::endl;
#endif
        return false;
      }

      go_to_next_level = false;
      const int u_r = std::floor(u);
      const int v_r = std::floor(v);
      if(u_r < 0 || v_r < 0 || u_r >= width-patch_size || v_r >= height-patch_size)
      {
#if SVO_PYRAMIDAL_VERBOSE
        std::cout << "current patch is too close to the border "
                  << "u,v: " << u_r << ", " << v_r
                  << "\t w,h:" << width << ", " << height
                  << "\t patch_size = " << patch_size << std::endl;
#endif
        go_to_next_level = true;
        break;
      }

      // compute interpolation weights
      const float subpix_x = u-u_r;
      const float subpix_y = v-v_r;
      const uint16_t wTL = static_cast<uint16_t>((1.0f-subpix_x)*(1.0f-subpix_y)*(1<<SHIFT_BITS));
      const uint16_t wTR = static_cast<uint16_t>(subpix_x*(1.0f-subpix_y)*(1<<SHIFT_BITS));
      const uint16_t wBL = static_cast<uint16_t>((1.0f-subpix_x)*subpix_y*(1<<SHIFT_BITS));
      const uint16_t wBR = (1 << SHIFT_BITS) - wTL - wTR - wBL;

      // loop through search_patch, interpolate
      uint8_t* it_ref = ref_patch;
      Eigen::Vector2f Jres; Jres.setZero();

#ifdef __ARM_NEON__
      // initialize result to zero
      int32x4_t vb1 = vdupq_n_s32(0);
      int32x4_t vb2 = vdupq_n_s32(0);
      for(int y=0; y<patch_size; ++y)
      {
        uint8_t* it = img_cur.data + (v_r+y)*step + (u_r);
        for(int x=0; x<patch_size; x+=8, it+=8, it_ref+=8)
        {
          // load and convert from uint8 to uint16
          uint16x8_t v00 = vmovl_u8( vld1_u8( it ) );
          uint16x8_t v01 = vmovl_u8( vld1_u8( it + 1 ) );
          uint16x8_t v10 = vmovl_u8( vld1_u8( it + step ) );
          uint16x8_t v11 = vmovl_u8( vld1_u8( it + step + 1 ) );

          // vector multiply by scalar
          v00 = vmulq_n_u16( v00, wTL );
          v01 = vmulq_n_u16( v01, wTR );
          v10 = vmulq_n_u16( v10, wBL );
          v11 = vmulq_n_u16( v11, wBR );

          // add all results together
          v00 = vaddq_u16( v00, vaddq_u16( v01, vaddq_u16( v10, v11 ) ) );

          // descale: shift right by constant
          v00 = vrshrq_n_u16(v00, SHIFT_BITS);

          // compute difference between reference and interpolated patch,
          // use reinterpet-cast to make signed [-255,255]
          int16x8_t res = vsubq_s16(vreinterpretq_s16_u16(v00),
                                    vreinterpretq_s16_u16(vmovl_u8(vld1_u8( it_ref ))));

          // Vector multiply accumulate long: vmla -> Vr[i] := Va[i] + Vb[i] * Vc[i]
          // int32x4_t  vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c);    // VMLAL.S16 q0,d0,d0
          int16x8_t grad = vld1q_s16(ref_patch_dx + y*patch_size + x);
          vb1 = vmlal_s16(vb1, vget_low_s16(grad), vget_low_s16(res));
          vb1 = vmlal_s16(vb1, vget_high_s16(grad), vget_high_s16(res));

          grad = vld1q_s16(ref_patch_dy + y*patch_size + x);
          vb2 = vmlal_s16(vb2, vget_low_s16(grad), vget_low_s16(res));
          vb2 = vmlal_s16(vb2, vget_high_s16(grad), vget_high_s16(res));
        }
      }

      // finally, sum results of vb1, vb2
      int32x2_t tmp;
      tmp = vpadd_s32(vget_low_s32(vb1), vget_high_s32(vb1));
      Jres[0] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);

      tmp = vpadd_s32(vget_low_s32(vb2), vget_high_s32(vb2));
      Jres[1] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);
#else
      int16_t* it_ref_dx = ref_patch_dx;
      int16_t* it_ref_dy = ref_patch_dy;

      for(int y=0; y<patch_size; ++y)
      {
        uint8_t* it = img_cur.data + (v_r+y)*step + (u_r);
        for(int x=0; x<patch_size; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
        {
          uint16_t cur = SVO_DESCALE(wTL*it[0] + wTR*it[1] + wBL*it[step] + wBR*it[step+1], SHIFT_BITS);
          //float cur = wTL*it[0] + wTR*it[1] + wBL*it[step] + wBR*it[step+1];
          float res = static_cast<float>(cur) - *it_ref;
          Jres[0] -= res*(*it_ref_dx);
          Jres[1] -= res*(*it_ref_dy);
        }
      }
#endif

      update = Hinv * Jres * 2.0; // * 2 to compensate because above, we did not compute the derivative correctly
      u += update[0];
      v += update[1];

#if SVO_PYRAMIDAL_VERBOSE
      std::cout << "Level: " << level
                << "\t Iter: " << iter << ":"
                << "\t u=" << u << ", v=" << v
                << "\t update = " << update[0] << ", " << update[1] << std::endl;
#endif

      if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
      {
#if SVO_PYRAMIDAL_VERBOSE
        std::cout << "level = " << level << ": converged." << std::endl;
#endif
        converged=true;
        break;
      }
    } // end iterations

    px_cur_level_0 = Keypoint((u+halfpatch_size+px_ref_offset[0])*scale,
        (v+halfpatch_size+px_ref_offset[1])*scale);

    if(!converged && !go_to_next_level)
      return false; // no need to go to next level when we didn't converge at this one.

  } // end pyramid

  return converged;
}

} // namespace feature_alignment
} // namespace svo
