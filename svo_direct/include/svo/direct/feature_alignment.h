// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#ifndef SVO_DIRECT_FEATURE_ALIGNMENT_H_
#define SVO_DIRECT_FEATURE_ALIGNMENT_H_

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <svo/common/types.h>

namespace svo {

/// Subpixel refinement of a reference feature patch with the current image.
/// Implements the inverse-compositional approach (see "Lucas-Kanade 20 Years on"
/// paper by Baker.
namespace feature_alignment {

bool align1D(
    const cv::Mat& cur_img,
    const Eigen::Ref<GradientVector>& dir,                  // direction in which the patch is allowed to move
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    const bool affine_est_offset,
    const bool affine_est_gain,
    Keypoint* cur_px_estimate,
    double* h_inv = nullptr);

bool align2D(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    const bool affine_est_offset,
    const bool affine_est_gain,
    Keypoint& cur_px_estimate,
    bool no_simd = false,
    std::vector<Eigen::Vector2f>* each_step=nullptr);

bool align2D_SSE2(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Keypoint& cur_px_estimate);

bool align2D_NEON(
    const cv::Mat& cur_img,
    uint8_t* ref_patch_with_border,
    uint8_t* ref_patch,
    const int n_iter,
    Keypoint& cur_px_estimate);

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
    std::vector<uint8_t>& status);

bool alignPyr2D(
    const std::vector<cv::Mat>& img_pyr_ref,
    const std::vector<cv::Mat>& img_pyr_cur,
    const int max_level,
    const int min_level,
    const std::vector<int>& patch_sizes,
    const int n_iter,
    const float min_update_squared,
    const Eigen::Vector2i &px_ref_level_0,
    Keypoint &px_cur_level_0);

} // namespace feature_alignment
} // namespace svo

#endif // SVO_DIRECT_FEATURE_ALIGNMENT_H_
