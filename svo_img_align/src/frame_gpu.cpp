// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/img_align/frame_gpu.h>

#include <algorithm>
#include <stdexcept>
#include <fast/fast.h>
#include <vikit/math_utils.h>
#include <vikit/vision.h>

/// @todo move to camera.h ?
#ifndef SVO_USE_VIKIT_CAMERA
# include <aslam/cameras/camera-pinhole.h>
# include <aslam/cameras/camera-unified-projection.h>
# include <aslam/cameras/camera-omni.h>
#endif
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

#include <svo/common/logging.h>
#include <svo/common/point.h>
#include <svo/common/camera.h>


namespace svo {


FrameGpu::FrameGpu(
    const CameraPtr& cam,
    const cv::Mat& img,
    const uint64_t timestamp_ns,
    size_t n_pyr_levels)
  : Frame(cam, img, timestamp_ns, n_pyr_levels)
{ }

FrameGpu::FrameGpu(
    const int id,
    const uint64_t timestamp_ns,
    const CameraPtr& cam,
    const Transformation& T_world_cam)
  : Frame(id, timestamp_ns, cam, T_world_cam)
{ }

FrameGpu::~FrameGpu()
{ }

void FrameGpu::initGpuData(imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr& cu_T_imu_cam,
                           imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr& cu_T_cam_imu,
                           imp::cu::PinholeCamera::Ptr& cu_camera)
{
  // Copy image pyramid to GPU
  // TODO: USE imp::ImagePyramid
  cu_img_pyramid_copy_.resize(img_pyr_.size());
  for(int i = 0; i < static_cast<int>(img_pyr_.size()); ++i)
  {
    cu_img_pyramid_copy_.at(i) = std::make_shared<imp::cu::ImageGpu8uC1>(
          imp::cu::ImageGpu8uC1(imp::ImageCv8uC1(img_pyr_.at(i))));

  }

  // Initialize transformations
  cu_T_imu_cam_ = cu_T_imu_cam;
  cu_T_cam_imu_ = cu_T_cam_imu;
  cu_camera_ = cu_camera;
}

} // namespace svo
