// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <mutex>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <vikit/math_utils.h>
#include <imp/cu_imgproc/image_pyramid.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_pinhole_camera.cuh>
#include <svo/common/frame.h>
#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/feature_wrapper.h>
#include <svo/common/seed_wrapper.h>
#include <svo/img_align/sparse_img_align_device_utils.cuh> //!< for FloatTypeGpu

namespace svo {

/**
 * @brief The FrameGpu class adds GPU data members to the frame base class.
 */
class FrameGpu : public Frame
{
public:
  typedef std::shared_ptr<FrameGpu> Ptr;

  std::vector<imp::cu::ImageGpu8uC1::Ptr> cu_img_pyramid_copy_; //!< Image Pyramid
  // TODO: USE imp::ImagePyramid
  //std::vector<imp::ImagePyramid8uC1::Ptr> cu_ref_pyramids_device_;
  imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr cu_T_imu_cam_;
  imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr cu_T_cam_imu_;
  imp::cu::PinholeCamera::Ptr cu_camera_;

  /// Default Constructor
  FrameGpu(const CameraPtr& cam,
      const cv::Mat& img,
      const uint64_t timestamp_ns,
      const size_t n_pyr_levels);

  /// Constructor without image. Just for testing!
  FrameGpu(
      const int id,
      const uint64_t timestamp_ns,
      const CameraPtr& cam,
      const Transformation& T_world_cam);

  /// Empty constructor. Just for testing!
  FrameGpu() {}

  /// Destructor
  virtual ~FrameGpu();

  // no copy
  FrameGpu(const FrameGpu&) = delete;
  FrameGpu& operator=(const FrameGpu&) = delete;

  /// Initialize new frame and create image pyramid.
  void initGpuData(imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr& cu_T_imu_cam,
                   imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr& cu_T_cam_imu,
                   imp::cu::PinholeCamera::Ptr& cu_camera);
};

} // namespace svo
