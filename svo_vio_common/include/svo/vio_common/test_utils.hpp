// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#pragma once

#include <tuple>
#include <random>

#include <svo/common/types.h>
#include <svo/common/camera.h>

namespace svo {
namespace test_utils {

// -----------------------------------------------------------------------------
// Generate visible keypoints and landmarks.

//! Generate random visible keypoints.
Keypoints generateRandomKeypoints(
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t margin,
    const uint32_t num_keypoints);

//! Generate count random visible keypoints.
Keypoints generateUniformKeypoints(
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t margin,
    const uint32_t num_keypoints);

//! Generate random visible 3d points.
std::tuple<Keypoints, Bearings, Positions> generateRandomVisible3dPoints(
    const Camera& cam,
    const uint32_t num_points,
    const uint32_t margin = 10u,
    const FloatType min_depth = 1.0,
    const FloatType max_depth = 3.0);

// -----------------------------------------------------------------------------
// Check overlapping field of view.

//! Check if two cameras in a rig have an overlapping field of view.
//! @return Approximate percentage of overlapping field of view between cameras.
FloatType overlappingFieldOfView(
    const CameraBundle& rig,
    const uint32_t cam_a,
    const uint32_t cam_b);

// -----------------------------------------------------------------------------
// Check landmark visiblity.

//! Return if pixel u is within image boundaries.
template<typename DerivedKeyPoint>
bool isVisible(
    const typename DerivedKeyPoint::Scalar image_width,
    const typename DerivedKeyPoint::Scalar image_height,
    const Eigen::MatrixBase<DerivedKeyPoint>& px)
{
  return px[0] >= 0
      && px[1] >= 0
      && px[0] <  image_width
      && px[1] <  image_height;
}

//! Return if pixel px is within image boundaries with margin.
template<typename DerivedKeyPoint>
bool isVisibleWithMargin(
    const typename DerivedKeyPoint::Scalar image_width,
    const typename DerivedKeyPoint::Scalar image_height,
    const Eigen::MatrixBase<DerivedKeyPoint>& px,
    const typename DerivedKeyPoint::Scalar margin)
{
  return px[0] >= margin
      && px[1] >= margin
      && px[0] < (image_width - margin)
      && px[1] < (image_height - margin);
}

//! Return if pixel px is within image boundaries with margin.
inline bool isVisibleWithMargin(
    const int image_width, const int image_height,
    const int x,const int y, const int margin)
{
  return x >= margin
      && y >= margin
      && x < (image_width - margin)
      && y < (image_height - margin);
}
// -----------------------------------------------------------------------------
// Generate random matrix or data.

//! @return Sample from integer-valued distribution.
template<typename T>
T sampleUniformIntDistribution(bool deterministic = false,
                               FloatType from  = 0.0,
                               FloatType to    = 1.0)
{
  static std::mt19937 gen_nondeterministic(std::random_device{}());
  static std::mt19937 gen_deterministic(0);
  auto dist = std::uniform_int_distribution<T>(from, to);
  return deterministic ? dist(gen_deterministic) : dist(gen_nondeterministic);
}

//! @return Sample from uniform real-valued distribution.
template<typename T>
T sampleUniformRealDistribution(bool deterministic = false,
                                FloatType from  = 0.0,
                                FloatType to    = 1.0)
{
  static std::mt19937 gen_nondeterministic(std::random_device{}());
  static std::mt19937 gen_deterministic(0);
  auto dist = std::uniform_real_distribution<T>(from, to);
  return deterministic ? dist(gen_deterministic) : dist(gen_nondeterministic);
}

inline Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>
randomMatrixUniformDistributed(int rows,
                               int cols,
                               bool deterministic = false,
                               FloatType from  = 0.0,
                               FloatType to    = 1.0)
{
  assert(rows > 0);
  assert(cols > 0);
  Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> m(rows, cols);
  for (int x = 0; x < cols; ++x)
  {
    for (int y = 0; y < rows; ++y)
    {
      m(y,x) = sampleUniformRealDistribution<FloatType>(deterministic, from, to);
    }
  }
  return m;
}

template<int rows, int cols>
Eigen::Matrix<FloatType, rows, cols> randomMatrixUniformDistributed(
    bool deterministic = false,
    FloatType from  = 0.0,
    FloatType to    = 1.0)
{
  return randomMatrixUniformDistributed(rows, cols, deterministic, from, to);
}

template<int size>
Eigen::Matrix<FloatType, size, 1> randomVectorUniformDistributed(
    bool deterministic = false,
    FloatType from  = 0.0,
    FloatType to    = 1.0)
{
  return randomMatrixUniformDistributed<size, 1>(deterministic, from, to);
}

//! @return Sample from normal distribution (real-valued).
template<typename T>
T sampleNormalDistribution(bool deterministic = false,
                           FloatType mean  = 0.0,
                           FloatType sigma = 1.0)
{
  static std::mt19937 gen_nondeterministic(std::random_device{}());
  static std::mt19937 gen_deterministic(0);
  auto dist = std::normal_distribution<T>(mean, sigma);
  return deterministic ? dist(gen_deterministic) : dist(gen_nondeterministic);
}

inline Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>
randomMatrixNormalDistributed(int rows,
                              int cols,
                              bool deterministic = false,
                              FloatType mean  = 0.0,
                              FloatType sigma = 1.0)
{
  assert(rows > 0);
  assert(cols > 0);
  Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> m(rows, cols);
  for (int x = 0; x < cols; ++x)
  {
    for (int y = 0; y < rows; ++y)
    {
      m(y,x) = sampleNormalDistribution<FloatType>(deterministic, mean, sigma);
    }
  }
  return m;
}

template<int rows, int cols>
Eigen::Matrix<FloatType, rows, cols>
randomMatrixNormalDistributed(bool deterministic = false,
                              FloatType mean  = 0.0,
                              FloatType sigma = 1.0)
{
  return randomMatrixNormalDistributed(rows, cols, deterministic, mean, sigma);
}

template<int size>
Eigen::Matrix<FloatType, size, 1>
randomVectorNormalDistributed(bool deterministic = false,
                              FloatType mean  = 0.0,
                              FloatType sigma = 1.0)
{
  return randomMatrixNormalDistributed<size, 1>(deterministic, mean, sigma);
}

} // namespace test_utils
} // namespace ze
