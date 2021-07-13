// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/feature_detection.h>

#include <Eigen/Dense>
#include <fast/fast.h>
#include <vikit/vision.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <aslam/common/memory.h>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <svo/common/logging.h>
#include <svo/direct/feature_detection_utils.h>

namespace svo {

namespace fd_utils = feature_detection_utils;

//------------------------------------------------------------------------------
AbstractDetector::AbstractDetector(
    const DetectorOptions& options,
    const CameraPtr& cam)
  : options_(options)
  , grid_(options_.cell_size,
          std::ceil(static_cast<double>(cam->imageWidth())/options_.cell_size),
          std::ceil(static_cast<double>(cam->imageHeight())/options_.cell_size))
  , closeness_check_grid_(options_.cell_size/options_.sec_grid_fineness,
                    std::ceil(options_.sec_grid_fineness * static_cast<double>(cam->imageWidth())/options_.cell_size),
                    std::ceil(options_.sec_grid_fineness * static_cast<double>(cam->imageHeight())/options_.cell_size))
{}

//------------------------------------------------------------------------------
void AbstractDetector::detect(const FramePtr& frame)
{
  size_t max_n_features = grid_.size();
  detect(frame->img_pyr_, frame->getMask(), max_n_features, frame->px_vec_,
         frame->score_vec_, frame->level_vec_, frame->grad_vec_, frame->type_vec_);
  frame->num_features_ = frame->px_vec_.cols();
  frame->landmark_vec_.resize(frame->num_features_, nullptr);
  frame->seed_ref_vec_.resize(frame->num_features_);
  frame->invmu_sigma2_a_b_vec_.resize(Eigen::NoChange, frame->num_features_);
  frame_utils::computeNormalizedBearingVectors(frame->px_vec_, *frame->cam(), &frame->f_vec_);
}

//------------------------------------------------------------------------------
void FastDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  Corners corners(
        grid_.n_cols*grid_.n_rows,
        Corner(0, 0, options_.threshold_primary, 0, 0.0f));
  fd_utils::fastDetector(
        img_pyr, options_.threshold_primary, options_.border,
        options_.min_level, options_.max_level, corners, grid_);
  fd_utils::fillFeatures(
        corners, FeatureType::kCorner, mask, options_.threshold_primary,
        max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);

  resetGrid();
}

//------------------------------------------------------------------------------
void GradientDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  // Compute pyramid of derivative max magnitude
  ImgPyr mag_pyr(img_pyr.size());
  for(size_t i=0; i<img_pyr.size(); ++i)
  {
    fd_utils::computeDerivMaxMagnitude(img_pyr[i], mag_pyr[i]);
  }

  cv::Mat mag_ss;
  mag_pyr[0].convertTo(mag_ss, CV_32F, 1.0f/500.0f);
  //const int stride=mag_ss.cols;
#if 0
  const size_t max_level=img_pyr.size();

  if(false)
  {
    // Multiply across scale space
    float* mag_ss_ptr = (float*) mag_ss.data;
    for(int y=0; y<mag_ss.rows; ++y)
    {
      for(int x=0; x<mag_ss.cols; ++x, ++mag_ss_ptr)
      {
        for(size_t L=1; L<max_level; ++L)
        {
          const float scale = 1.0f / (1<<L);
          const int u = static_cast<int>(scale*x+0.5f);
          const int v = static_cast<int>(scale*y+0.5f);
          *mag_ss_ptr += 1.0/500.0f*mag_pyr[L].at<uint8_t>(v,u); // += ?
        }
      }
    }
  }
#endif

  // Nonmax suppression
  Corners corners;
  fd_utils::nonmax(mag_ss, options_.threshold_primary, &corners);
  fd_utils::setCornerAngles(img_pyr, &corners);

  // Create feature for every corner that has high enough corner score
  fd_utils::fillFeatures(
        corners, FeatureType::kEdgelet, mask, options_.threshold_secondary,
        max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);
}

void GradientDetectorGrid::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  Corners corners(
        grid_.n_cols * grid_.n_rows,
        Corner(0, 0, options_.threshold_secondary, 0, 0.0f));
  fd_utils::edgeletDetector_V2(
        img_pyr, options_.threshold_secondary, options_.border,
        options_.min_level, options_.max_level, corners, grid_);
  fd_utils::fillFeatures(
        corners, FeatureType::kEdgelet, mask, options_.threshold_secondary,
        max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);

  resetGrid();
}

//------------------------------------------------------------------------------
void FastGradDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  {
    // Detect fast corners.
    Corners corners(
          grid_.n_cols*grid_.n_rows,
          Corner(0, 0, options_.threshold_primary, 0, 0.0f));
    fd_utils::fastDetector(
          img_pyr, options_.threshold_primary, options_.border,
          options_.min_level, options_.max_level, corners, grid_);
    fd_utils::fillFeatures(
          corners, FeatureType::kCorner, mask, options_.threshold_primary,
          max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);
  }

  int max_features = static_cast<int>(max_n_features) - px_vec.cols();
  if(max_features > 0)
  {
    // Detect edgelets.
    Corners corners(
          grid_.n_cols * grid_.n_rows,
          Corner(0, 0, options_.threshold_secondary, 0, 0.0f));
    fd_utils::edgeletDetector_V2(
          img_pyr, options_.threshold_secondary, options_.border,
          options_.min_level, options_.max_level, corners, grid_);
    fd_utils::fillFeatures(
          corners, FeatureType::kEdgelet, mask, options_.threshold_secondary,
          max_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);
  }

  resetGrid();
}

//------------------------------------------------------------------------------
void ShiTomasiGradDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  {
    // Detect shitomasi corners.
    Corners corners(
          grid_.n_cols*grid_.n_rows,
          Corner(0, 0, options_.threshold_shitomasi, 0, 0.0f));
    fd_utils::shiTomasiDetector(
          img_pyr, options_.threshold_shitomasi, options_.border,
          options_.min_level, options_.max_level, corners, grid_, closeness_check_grid_);
    fd_utils::fillFeatures(
          corners, FeatureType::kCorner, mask, options_.threshold_shitomasi,
          max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);
  }

  int max_features = static_cast<int>(max_n_features) - px_vec.cols();
  if(max_features > 0)
  {
    // Detect fast corners.
    Corners corners_fast(
          grid_.n_cols*grid_.n_rows,
          Corner(0, 0, options_.threshold_primary, 0, 0.0f));
    fd_utils::fastDetector(
          img_pyr, options_.threshold_primary, options_.border,
          options_.min_level, options_.max_level, corners_fast, grid_);
    fd_utils::fillFeatures(
          corners_fast, FeatureType::kCorner, mask, options_.threshold_primary,
          max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);

    // Detect edgelets.
    Corners corners_grad(
          grid_.n_cols * grid_.n_rows,
          Corner(0, 0, options_.threshold_secondary, 0, 0.0f));
    fd_utils::edgeletDetector_V2(
          img_pyr, options_.threshold_secondary, options_.border,
          options_.min_level, options_.max_level, corners_grad, grid_);
    fd_utils::fillFeatures(
          corners_grad, FeatureType::kEdgelet, mask, options_.threshold_secondary,
          max_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);
  }

  resetGrid();
}

//------------------------------------------------------------------------------

void ShiTomasiDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  {
    // Detect shitomasi corners.
    Corners corners(
          grid_.n_cols*grid_.n_rows,
          Corner(0, 0, options_.threshold_shitomasi, 0, 0.0f));
    fd_utils::shiTomasiDetector(
          img_pyr, options_.threshold_shitomasi, options_.border,
          options_.min_level, options_.max_level, corners, grid_,
          closeness_check_grid_);
    fd_utils::fillFeatures(
          corners, FeatureType::kMapPoint, mask, options_.threshold_shitomasi,
          max_n_features, px_vec, score_vec, level_vec, grad_vec, types_vec, grid_);
  }
  resetGrid();
}

//------------------------------------------------------------------------------
//! @todo (MWE) kept the old version around as there might be changes still and
//! we want to have the same structure and workflow within the detectors.
#if 0
void AllPixelsDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  Corners corners;
  for(int y=1; y<=img_pyr[0].rows-1; ++y)
  {
    for(int x=1; x<=img_pyr[0].cols-1; ++x)
    {
      corners.push_back(Corner(x, y, std::numeric_limits<double>::max(), 0, 0.0f));
    }
  }
  fd_utils::setCornerAngles(img_pyr, &corners);

  fd_utils::fillFeatures(
        corners, FeatureType::kEdgelet, mask, 0.0, max_n_features, px_vec, score_vec,
        level_vec, grad_vec, types_vec, grid_);

  resetGrid();
}
#else // new
void AllPixelsDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t /*max_n_features*/,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  const int width = img_pyr.at(options_.sampling_level).cols;
  const int height = img_pyr.at(options_.sampling_level).rows;
  const int pyr_init_scale = 1 << options_.sampling_level;
  const int border = options_.border;
  const size_t num_features = (width-2*border) * (height-2*border);

  px_vec.resize(Eigen::NoChange, num_features);
  score_vec.setConstant(num_features, 1.0);
  level_vec.setConstant(num_features, options_.level);
  grad_vec.resize(Eigen::NoChange, num_features);
  types_vec.assign(num_features, svo::FeatureType::kCorner);
  size_t feature_index = 0;
  for(int y=border; y<height-border; ++y)
  {
    for(int x=border; x<width-border; ++x)
    {
      px_vec.col(feature_index++) = svo::Keypoint(x*pyr_init_scale, y*pyr_init_scale);
    }
  }
}
#endif

void CannyDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  // Compute pyramid of derivative max magnitude
  cv::Mat canny_edges;
  fd_utils::detectCannyEdges(img_pyr[options_.sampling_level], canny_edges);

  const int width = img_pyr.at(options_.sampling_level).cols;
  const int height = img_pyr.at(options_.sampling_level).rows;
  const int border = 1;
  const int pyr_init_scale = 1 << options_.sampling_level;
  const int max_num_features = cv::countNonZero(canny_edges == 255);

  px_vec.resize(Eigen::NoChange, max_num_features);
  size_t feature_index = 0;
  for(int y=border; y<height-border; ++y)
  {
    for(int x=border; x<width-border; ++x)
    {
      if (canny_edges.at<uchar>(y,x))
      {
        px_vec.col(feature_index++) = svo::Keypoint(x*pyr_init_scale, y*pyr_init_scale);
      }
    }
  }

  size_t num_features = feature_index - 1;
  px_vec.conservativeResize(Eigen::NoChange, num_features);
  score_vec.setConstant(num_features, 1.0);
  level_vec.setConstant(num_features, options_.level);
  grad_vec.resize(Eigen::NoChange, num_features);
  types_vec.assign(num_features, svo::FeatureType::kCorner);
}

void SobelDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  // Compute pyramid of derivative max magnitude
  cv::Mat sobel_edges;
  fd_utils::detectSobelEdges(img_pyr[options_.sampling_level], sobel_edges);

  const int width = img_pyr.at(options_.sampling_level).cols;
  const int height = img_pyr.at(options_.sampling_level).rows;
  const int border = 1;
  const int pyr_init_scale = 1 << options_.sampling_level;
  const int max_num_features = cv::countNonZero(sobel_edges==255);

  px_vec.resize(Eigen::NoChange, max_num_features);
  size_t feature_index = 0;
  for(int y=border; y<height-border; ++y)
  {
    for(int x=border; x<width-border; ++x)
    {
      if (sobel_edges.at<uchar>(y,x))
      {
        px_vec.col(feature_index++) = svo::Keypoint(x*pyr_init_scale, y*pyr_init_scale);
      }
    }
  }

  size_t num_features = feature_index - 1;
  px_vec.conservativeResize(Eigen::NoChange, num_features);
  score_vec.setConstant(num_features, 1.0);
  level_vec.setConstant(num_features, options_.level);
  grad_vec.resize(Eigen::NoChange, num_features);
  types_vec.assign(num_features, svo::FeatureType::kCorner);
}

//------------------------------------------------------------------------------
void GradientHuangMumfordDetector::detect(
    const ImgPyr& img_pyr,
    const cv::Mat& mask,
    const size_t max_n_features,
    Keypoints& px_vec,
    Scores& score_vec,
    Levels& level_vec,
    Gradients& grad_vec,
    FeatureTypes& types_vec)
{
  // Compute pyramid of derivative max magnitude
  ImgPyr mag_pyr_32f(img_pyr.size());
  for(size_t i=0; i<img_pyr.size(); ++i)
  {
    fd_utils::computeDerivHuangMumford(img_pyr[i], mag_pyr_32f[i]);
  }

  const int width = img_pyr.at(options_.sampling_level).cols;
  const int height = img_pyr.at(options_.sampling_level).rows;
  const int border = 1;
  const int pyr_init_scale = 1 << options_.sampling_level;

  cv::Mat mag_level_32f = mag_pyr_32f[options_.sampling_level];
  cv::Mat mag_level_thresholded;
  cv::threshold(mag_level_32f, mag_level_thresholded, options_.threshold_primary,
                1.0, cv::THRESH_BINARY_INV);
  const int max_num_features = cv::countNonZero(mag_level_thresholded==1.0);

#if 0
  cv::imshow("mag_level_32f", mag_level_32f);
  cv::imshow("mag_level_thresholded", mag_level_thresholded);
  cv::waitKey(0);
#endif
  px_vec.resize(Eigen::NoChange, max_num_features);

  size_t feature_index = 0;
  for(int y=border; y<height-border; ++y)
  {
    for(int x=border; x<width-border; ++x)
    {
      if (mag_level_thresholded.at<uchar>(y,x))
      {
        px_vec.col(feature_index++) = svo::Keypoint(x*pyr_init_scale, y*pyr_init_scale);
      }
    }
  }

  size_t num_features = feature_index - 1;

  // Resize px_vec to the actual number of features while keeping the data.
  px_vec.conservativeResize(Eigen::NoChange, num_features);
  score_vec.setConstant(num_features, 1.0);
  level_vec.setConstant(num_features, options_.level);
  grad_vec.resize(Eigen::NoChange, num_features);
  types_vec.assign(num_features, svo::FeatureType::kCorner);

//  // Nonmax suppression
//  Corners corners;
//  fd_utils::nonmax(mag_ss, options_.threshold_primary, &corners);
//  fd_utils::setCornerAngles(img_pyr, &corners);

//  // Create feature for every corner that has high enough corner score
//  fd_utils::fillFeatures(
//        corners, FeatureType::kEdgelet, mask, options_.threshold_secondary,
//        px_vec, level_vec, grad_vec, types_vec, grid_);
}

} // namespace svo

