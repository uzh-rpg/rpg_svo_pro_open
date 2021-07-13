// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#include "svo/vio_common/test_utils.hpp"

#include <random>

#include <svo/common/types.h>
#include <svo/common/transformation.h>

#include "svo/vio_common/logging.hpp"

namespace svo {
namespace test_utils {

// -----------------------------------------------------------------------------
Keypoints generateRandomKeypoints(const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t margin,
    const uint32_t num_keypoints)
{
  DEBUG_CHECK_GT(image_width, margin + 1u);
  DEBUG_CHECK_GT(image_height, margin + 1u);

  Keypoints kp(2, num_keypoints);
  for(uint32_t i = 0u; i < num_keypoints; ++i)
  {
    kp(0,i) = sampleUniformRealDistribution<FloatType>(
          false, margin, image_width - 1 - margin);
    kp(1,i) = sampleUniformRealDistribution<FloatType>(
          false, margin, image_height - 1 - margin);
  }
  return kp;
}

// -----------------------------------------------------------------------------
Keypoints generateUniformKeypoints(const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t margin,
    const uint32_t num_cols)
{
  DEBUG_CHECK_GT(image_width, margin + 1u);
  DEBUG_CHECK_GT(image_height, margin + 1u);
  const uint32_t num_rows = num_cols * image_height / image_width;

  // Compute width and height of a cell:
  FloatType w = (static_cast<FloatType>(image_width - 0.01)  - 2.0 * margin)
      / (num_cols - 1);
  FloatType h = (static_cast<FloatType>(image_height - 0.01) - 2.0 * margin)
      / (num_rows - 1);

  // Sample keypoints:
  Keypoints kp(2, num_rows * num_cols);
  for (uint32_t y = 0u; y < num_rows; ++y)
  {
    for (uint32_t x = 0u; x < num_cols; ++x)
    {
      uint32_t i = y * num_cols + x;
      kp(0,i) = margin + x * w;
      kp(1,i) = margin + y * h;
    }
  }
  return kp;
}

// -----------------------------------------------------------------------------
std::tuple<Keypoints, Bearings, Positions> generateRandomVisible3dPoints(
    const Camera& cam,
    const uint32_t num_points,
    const uint32_t margin,
    const FloatType min_depth,
    const FloatType max_depth)
{
  Keypoints px = generateRandomKeypoints(cam.imageWidth(),cam.imageHeight(),
                                         margin, num_points);
  Bearings f;
  std::vector<bool> success(px.cols(),false);
  cam.backProject3(Eigen::Ref<Keypoints>(px),&f,&success);
  Positions pos = f;
  for(uint32_t i = 0u; i < num_points; ++i)
  {
    pos.col(i) *= sampleUniformRealDistribution<FloatType>(
          false, min_depth, max_depth);
  }
  return std::make_tuple(px, f, pos);
}

// -----------------------------------------------------------------------------
FloatType overlappingFieldOfView(
    const CameraBundle& rig,
    const uint32_t cam_A,
    const uint32_t cam_B)
{
  DEBUG_CHECK_LT(cam_A, rig.getNumCameras());
  DEBUG_CHECK_LT(cam_B, rig.getNumCameras());

  // We sample uniformly keypoints in camera a and project them into camera b,
  // assuming the landmark is at infinity (i.e. only rotate).
  Keypoints px_A = generateUniformKeypoints(rig.getCamera(cam_A).imageWidth(),
                                            rig.getCamera(cam_A).imageHeight(),
                                            0u, 20u);
  Bearings f_A;
  std::vector<bool> success(px_A.cols(),false);
  rig.getCamera(cam_A).backProject3(Eigen::Ref<Keypoints>(px_A),&f_A,&success);
  Transformation T_B_A = rig.get_T_C_B(cam_B) * rig.get_T_C_B(cam_A).inverse();
  Positions p_B = T_B_A.getRotation().rotateVectorized(f_A);
  Keypoints px_B(px_A);
  uint32_t num_visible = 0u;
  for (int i = 0; i < px_B.cols(); ++i)
  {
    Keypoint p;
    rig.getCamera(cam_B).project3(Eigen::Ref<Positions>(p_B),&p);
    px_B.col(i) = p;
    //! @todo: Omnidirectional cameras: Improve check.
    if (p_B.col(i)(2) > 0 && isVisible(rig.getCamera(cam_B).imageWidth(),
                                       rig.getCamera(cam_B).imageHeight(),
                                       px_B.col(i)))
    {
      ++num_visible;
    }
  }

  return static_cast<FloatType>(num_visible) / px_B.cols();
}

} // namespace test_utils
} // namespace svo
