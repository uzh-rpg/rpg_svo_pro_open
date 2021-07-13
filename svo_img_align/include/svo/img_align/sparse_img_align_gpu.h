// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <svo/img_align/sparse_img_align_base.h>

#include <vikit/solver/robust_cost.h>
#include <vikit/performance_monitor.h>
#include <imp/cu_imgproc/image_pyramid.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_pinhole_camera.cuh>
#include <svo/common/frame.h>
#include <svo/img_align/sparse_img_align_device_utils.cuh> //for CacheMemoryHandler

namespace svo {

typedef Eigen::Matrix<FloatTypeGpu,3,1> Vector3ftGpu;

struct HostCacheHandler
{
  std::vector<Float2TypeGpu> uv_cache;
  std::vector<Float3TypeGpu> xyz_ref_cache;
  std::vector<size_t> first_ftr_index;
  std::vector<size_t> nbr_of_ftrs;
  size_t total_nbr_of_ftrs;

  void clear()
  {
    uv_cache.clear();
    xyz_ref_cache.clear();
    first_ftr_index.clear();
    nbr_of_ftrs.clear();
  }

  void reserve(size_t feature_capacity)
  {
    uv_cache.reserve(feature_capacity);
    xyz_ref_cache.reserve(feature_capacity);
  }

  inline size_t capacity() const
  {
    return uv_cache.capacity();
  }

  inline void push_uv(const float& u, const float& v)
  {
    uv_cache.push_back({static_cast<FloatTypeGpu>(u),static_cast<FloatTypeGpu>(v)});
  }

  inline void push_uv(const double& u, const double& v)
  {
    uv_cache.push_back({static_cast<FloatTypeGpu>(u),static_cast<FloatTypeGpu>(v)});
  }

  inline void push_xyz(const float& x, const float& y, const float& z)
  {
    xyz_ref_cache.push_back({static_cast<FloatTypeGpu>(x), static_cast<FloatTypeGpu>(y),
                             static_cast<FloatTypeGpu>(z)});
  }

  inline void push_xyz(const double& x, const double& y, const double& z)
  {
    xyz_ref_cache.push_back({static_cast<FloatTypeGpu>(x), static_cast<FloatTypeGpu>(y),
                             static_cast<FloatTypeGpu>(z)});
  }
};

/// Optimize the pose of the frame by minimizing the photometric error of feature patches.
class SparseImgAlignGpu : public SparseImgAlignBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<SparseImgAlignGpu> Ptr;

public:
  SparseImgAlignGpu(
      SolverOptions optimization_options,
      SparseImgAlignOptions options);

  void setPatchSizeSideEffects()
  {
    gpu_cache_.setPatchArea(patch_area_);
  }

  size_t run(const FrameBundle::Ptr& ref_frames,
             const FrameBundle::Ptr& cur_frames);

  /**
   * @brief getMedianDisparity returns the median disparty for all features between the current frame and the
   * reference frame. The value is updated when calling sparse_img_align_device_utils::computeDisparity()
   * (see .cpp file)
   */
  inline FloatTypeGpu getMedianDisparity() {return median_disparity_;}

private:
  GPUProperties gpu_props_;
  int num_blocks_reduce_;
  int num_threads_reduce_;
  bool have_cache_ = false;
  FloatTypeGpu median_disparity_;

  // data storage
  std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr> > cu_cur_imgs_pyramid_copy_;
  std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr> > cu_ref_imgs_pyramid_copy_;
  // TODO: USE imp::ImagePyramid
  //std::vector<imp::ImagePyramid8uC1::Ptr> cu_ref_pyramids_device_;
  //std::vector<imp::ImagePyramid8uC1::Ptr> cu_cur_pyramids_device_;
  HostCacheHandler host_cache_;
  GpuCacheHandler gpu_cache_;

  std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr> cu_T_imu_cam_bundle_;
  std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr> cu_T_cam_imu_bundle_;
  std::vector<imp::cu::PinholeCamera::Ptr> cu_camera_bundle_;

  std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>> cu_T_cur_ref_bundle_;

protected:
  /// Warp the (cur)rent image such that it aligns with the (ref)erence image
  double evaluateError(
      const SparseImgAlignState &state,
      HessianMatrix* H,
      GradientVector* g);
};

namespace sparse_img_align_host_utils {

void extractFeaturesSubset(const Frame& ref_frame,
                           const int max_level,
                           const int patch_size_wb, // patch size + border (usually border=2 for gradient),
                           size_t& nr_fts_extracted,
                           HostCacheHandler& host_cache);

} // namespace sparse_img_align_host_utils
} // namespace svo
