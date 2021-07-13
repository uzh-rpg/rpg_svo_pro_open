// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <array>
#include <svo/common/types.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/transformation.h>

namespace svo {

// forward declarations
class Point;
class Frame;
class FeatureWrapper;

namespace patch_score {
template<int HALF_PATCH_SIZE> class ZMSSD;
}

/// Patch-matcher for reprojection-matching and epipolar search in triangulation.
class Matcher
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int kHalfPatchSize = 4;
  static const int kPatchSize = 8;

  typedef svo::patch_score::ZMSSD<kHalfPatchSize> PatchScore;
  typedef std::shared_ptr<Matcher> Ptr;

  struct Options
  {
    bool align_1d = false;                  //!< in epipolar search: align patch 1D along epipolar line
    int align_max_iter = 10;                //!< number of iterations for aligning the feature patches in gauss newton
    double max_epi_length_optim = 2.0;      //!< max length of epipolar line to skip epipolar search and directly go to img align
    size_t max_epi_search_steps = 100;      //!< max number of evaluations along epipolar line
    bool subpix_refinement = true;          //!< do gauss newton feature patch alignment after epipolar search
    bool epi_search_edgelet_filtering = true;
    bool scan_on_unit_sphere = true;
    double epi_search_edgelet_max_angle = 0.7;
    bool verbose = false;
    bool use_affine_warp_ = true;
    bool affine_est_offset_ = true;
    bool affine_est_gain_ = false;
    double max_patch_diff_ratio = 2.0;
  } options_;

  enum class MatchResult {
    kSuccess,
    kFailScore,
    kFailTriangulation,
    kFailVisibility,
    kFailWarp,
    kFailAlignment,
    kFailRange,
    kFailAngle,
    kFailCloseView,
    kFailLock,
    kFailTooFar
  };

  uint8_t patch_[kPatchSize*kPatchSize] __attribute__ ((aligned (16)));
  uint8_t patch_with_border_[(kPatchSize+2)*(kPatchSize+2)] __attribute__ ((aligned (16)));
  Eigen::Matrix2d A_cur_ref_;          //!< affine warp matrix
  Eigen::Vector2d epi_image_;          //!< vector from epipolar start to end on the image plane
  double epi_length_pyramid_;   //!< length of epipolar line segment in pixels on pyrimid level (only used for epipolar search)
  double h_inv_;                //!< hessian of 1d image alignment along epipolar line
  int search_level_;
  bool reject_;
  Keypoint px_cur_;
  BearingVector f_cur_;

  Matcher() = default;
  ~Matcher() = default;

  /// Find a match by directly applying subpix refinement.
  /// IMPORTANT! This function assumes that px_cur is already set to an estimate that is within ~2-3 pixel of the final result!
  MatchResult findMatchDirect(
      const Frame& ref_frame,
      const Frame& cur_frame,
      const FeatureWrapper& ref_ftr,
      const FloatType& ref_depth,
      Keypoint& px_cur);

  /// Find a match by searching along the epipolar line without using any features.
  MatchResult findEpipolarMatchDirect(
      const Frame& ref_frame,
      const Frame& cur_frame,
      const FeatureWrapper& ref_ftr,
      const double d_estimate_inv,
      const double d_min_inv,
      const double d_max_inv,
      double& depth);

  MatchResult findEpipolarMatchDirect(
      const Frame& ref_frame,
      const Frame& cur_frame,
      const Transformation &T_cur_ref,
      const FeatureWrapper& ref_ftr,
      const double d_estimate_inv,
      const double d_min_inv,
      const double d_max_inv,
      double& depth);
  /// search epipolar line between A~C~B for the best match with respect to patch score
  /// the search is done on patch_level, returns image coordinates and best ZMSSD
  void scanEpipolarLine(
      const Frame& frame,
      const Eigen::Vector3d& A,
      const Eigen::Vector3d& B,
      const Eigen::Vector3d& C,
      const PatchScore& patch_score,
      const int patch_level,
      Keypoint* image_best,
      int* zmssd_best);

  static std::string getResultString(const Matcher::MatchResult& result);

private:
  // TODO(zzc): perhaps some of these should be inline
  /// local optimization for patch_ and patch_with_border_ in *frame* around *px_cur*(image)
  MatchResult findLocalMatch(
      const Frame& frame,
      const Eigen::Ref<GradientVector>& direction,
      const int patch_level,
      Keypoint& px_cur);


  /// update best zmssd, if a better one is find, return true
  bool updateZMSSD(
      const Frame& frame,
      const Eigen::Vector2i& pxi,
      const int patch_level,
      const PatchScore& patch_score,
      int* zmssd_best);

  /// check is patch is fully within image
  bool isPatchWithinImage(
      const Frame& frame,
      const Eigen::Vector2i& pxi,
      const int patch_level);

  /// search along the epipolar line on image plane
  /// we sample on the UNIT PLANE and check corresponding patches on the image plane
  void scanEpipolarUnitPlane(const Frame& frame,
      const Eigen::Vector3d& A,
      const Eigen::Vector3d& B,
      const Eigen::Vector3d& C,
      const PatchScore& patch_score,
      const int patch_level,
      Keypoint* image_best,
      int* zmssd_best);

  /// search along the epipolar line on image plane
  /// we sample on the UNIT SPHERE and check corresponding patches on the image plane
  void scanEpipolarUnitSphere(const Frame& frame,
      const Eigen::Vector3d& A,
      const Eigen::Vector3d& B,
      const Eigen::Vector3d& C,
      const PatchScore& patch_score,
      const int patch_level,
      Keypoint* image_best,
      int* zmssd_best);
};

namespace matcher_utils {

/// calculate feature point depth
Matcher::MatchResult depthFromTriangulation(
    const Transformation& T_search_ref,
    const Eigen::Vector3d& f_ref,
    const Eigen::Vector3d& f_cur,
    double* depth);

/// returns image patch around px (used for testing)
void createPatchWithBorderNoWarp(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    const int halfpatch_size_without_border,
    uint8_t* patch_with_border);

/// returns a patch that is 2px smaller in both dimensions.
void createPatchFromPatchWithBorder(
    const uint8_t* const patch_with_border,
    const int patch_size,
    uint8_t* patch);


} // namespace matcher_utils
} // namespace svo
