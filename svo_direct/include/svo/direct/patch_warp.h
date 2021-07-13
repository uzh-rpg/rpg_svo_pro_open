// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/transformation.h>

namespace svo {

// Forward declarations.
class FeatureWrapper;

/// Warp a patch from the reference view to the current view.
namespace warp {

using AffineTransformation2 = Eigen::Matrix2d;

void getWarpMatrixAffine(
    const CameraPtr& cam_ref,
    const CameraPtr& cam_cur,
    const Eigen::Ref<Keypoint>& px_ref,
    const Eigen::Ref<BearingVector>& f_ref,
    const double depth_ref,
    const Transformation& T_cur_ref,
    const int level_ref,
    AffineTransformation2* A_cur_ref);

void getWarpMatrixAffineHomography(
    const CameraPtr& cam_ref,
    const CameraPtr& cam_cur,
    const Keypoint& px_ref,
    const BearingVector& f_ref,
    const BearingVector& normal_ref,
    const double depth_ref,
    const Transformation T_cur_ref,
    const int level_ref,
    AffineTransformation2& A_cur_ref);

int getBestSearchLevel(
    const AffineTransformation2& A_cur_ref,
    const int max_level);

bool warpAffine(
    const AffineTransformation2& A_cur_ref,
    const cv::Mat& img_ref,
    const Eigen::Ref<Keypoint>& px_ref,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch);

bool warpPixelwise(
    const Frame& cur_frame,
    const Frame& ref_frame,
    const FeatureWrapper& ref_ftr,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch);

void createPatchNoWarp(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    const int halfpatch_size,
    uint8_t* patch);

void createPatchNoWarpInterpolated(
    const cv::Mat& img,
    const Eigen::Ref<Keypoint>& px,
    const int halfpatch_size,
    uint8_t* patch);


} // namespace warp
} // namespace svo
