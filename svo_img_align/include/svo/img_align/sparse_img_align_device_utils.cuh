// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <cstdlib> // size_t

#include <Eigen/Dense>

#include <imp/cu_core/cu_linearmemory.cuh> // LinearMemory
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_pinhole_camera.cuh>
#include <imp/cu_core/cu_texture2d.cuh>
#include <imp/cu_imgproc/image_pyramid.hpp>
#include <imp/cu_core/cu_image_gpu.cuh>

#include <svo/img_align/gpu_types_cuda.h>
#include <svo/img_align/sparse_img_align_base.h>

namespace svo {

typedef imp::cu::LinearMemory<FloatPixelGpu> CuLinearMemoryFloat;
typedef imp::cu::LinearMemory<Float2PixelGpu> CuLinearMemoryFloat2;
typedef imp::cu::LinearMemory<Float3PixelGpu> CuLinearMemoryFloat3;
typedef imp::cu::LinearMemory<UIntPixelGpu> CuLinearMemoryUInt;
typedef imp::cu::LinearMemory<BoolPixelGpu> CuLinearMemoryBool;
typedef imp::LinearMemory<FloatPixelGpu> LinearMemoryFloat;
typedef imp::LinearMemory<Float2PixelGpu> LinearMemoryFloat2;
typedef imp::LinearMemory<Float3PixelGpu> LinearMemoryFloat3;
typedef imp::LinearMemory<UIntPixelGpu> LinearMemoryUInt;

/**
 * @brief The GpuCacheHandler class handels the dynamic memory allocation.
 * In order to reduce the number of memory reallocations, memory is only reallocated
 * if the current memory is not sufficiently large.
 */
class GpuCacheHandler
{
public:
  GpuCacheHandler();
  ~GpuCacheHandler() = default;
  GpuCacheHandler(const size_t patch_area);

  void setPatchArea(const size_t patch_area);
  void reserveFeatureCapacity(const size_t capacity);
  void reserveReductionCacheCapacity(const size_t capacity);
  inline void copyReductionCacheDeviceToHost();

  inline CuLinearMemoryFloat2& uv() const {return *uv_cache_;}
  inline CuLinearMemoryFloat3& xyzRef() const {return *xyz_ref_cache_;}
  inline CuLinearMemoryFloat& jacProj() const {return *jacobian_proj_cache_;}
  inline CuLinearMemoryFloat& jacobian() const {return *jacobian_cache_;}
  inline CuLinearMemoryFloat& residual() const {return *residual_cache_;}
  inline CuLinearMemoryBool& visibility() const {return *visibility_mask_;}
  inline CuLinearMemoryFloat& refPatch() const {return *ref_patch_cache_;}
  inline CuLinearMemoryFloat& hessianDevice() const {return *hessian_reduction_cache_;}
  inline CuLinearMemoryFloat& gradientDevice() const {return *gradient_reduction_cache_;}
  inline CuLinearMemoryFloat& chi2Device() const {return *chi2_reduction_cache_;}
  inline CuLinearMemoryUInt& nrVisibleDevice() const {return *nr_visible_cache_;}
  inline CuLinearMemoryFloat& disparity() const {return *disparity_cache_;}
  inline LinearMemoryFloat& hessianHost() const {return *hessian_reduction_cache_host_;}
  inline LinearMemoryFloat& gradientHost() const {return *gradient_reduction_cache_host_;}
  inline LinearMemoryFloat& chi2Host() const {return *chi2_reduction_cache_host_;}
  inline LinearMemoryUInt& nrVisibleHost() const {return *nr_visible_cache_host_;}


private:
  typedef CuLinearMemoryFloat2 UvCache;
  typedef CuLinearMemoryFloat3 XyzRefCache;
  typedef CuLinearMemoryFloat JacobianProjCache;
  typedef CuLinearMemoryFloat JacobianCache;
  typedef CuLinearMemoryFloat ResidualCache;
  typedef CuLinearMemoryBool VisibilityMask;
  typedef CuLinearMemoryFloat RefPatchCache;
  typedef CuLinearMemoryFloat HessianReductionCache;
  typedef CuLinearMemoryFloat GradientReductionCache;
  typedef CuLinearMemoryFloat Chi2ReductionCache;
  typedef CuLinearMemoryUInt NrVisibleCache;
  typedef CuLinearMemoryFloat DistparitiyCache;
  typedef LinearMemoryFloat HessianReductionCacheHost;
  typedef LinearMemoryFloat GradientReductionCacheHost;
  typedef LinearMemoryFloat Chi2ReductionCacheHost;
  typedef LinearMemoryUInt NrVisibleCacheHost;

  template <typename T> using UPtr  = std::unique_ptr<T>;

public:
  static constexpr size_t kJacStride = SparseImgAlignBase::kJacobianSize;
  static constexpr size_t kHessianTriagStride = SparseImgAlignBase::kHessianTriagN;
  static constexpr size_t kJacProjStride = 12;
  static constexpr size_t kMaxStorageSurplus = 1000;

private:
  size_t patch_area_;
  size_t feature_capacity_;
  size_t reduction_cache_capacity_;

  UPtr<UvCache> uv_cache_;
  UPtr<XyzRefCache> xyz_ref_cache_;
  UPtr<JacobianProjCache> jacobian_proj_cache_;
  UPtr<JacobianCache> jacobian_cache_;
  UPtr<ResidualCache> residual_cache_;
  UPtr<VisibilityMask> visibility_mask_;
  UPtr<RefPatchCache> ref_patch_cache_;
  UPtr<HessianReductionCache> hessian_reduction_cache_;
  UPtr<GradientReductionCache> gradient_reduction_cache_;
  UPtr<Chi2ReductionCache> chi2_reduction_cache_;
  UPtr<NrVisibleCache> nr_visible_cache_;
  UPtr<DistparitiyCache> disparity_cache_;
  UPtr<HessianReductionCacheHost> hessian_reduction_cache_host_;
  UPtr<GradientReductionCacheHost> gradient_reduction_cache_host_;
  UPtr<Chi2ReductionCacheHost> chi2_reduction_cache_host_;
  UPtr<NrVisibleCacheHost> nr_visible_cache_host_;
};

/**
 * @brief The GPUProperties class queries the GPU device properties upon
 * instantiation.
 */
class GPUProperties
{
public:
  GPUProperties()
  {
    int device;
    cudaError cu_err = cudaGetDevice(&device);
    if (cu_err != cudaSuccess)
      throw std::runtime_error("Unable to load GPU device");

    cu_err = cudaGetDeviceProperties(&prop_, device);
    if (cu_err != cudaSuccess)
      throw std::runtime_error("Unable to load GPU device properties");
  }

  ~GPUProperties() = default;

public:

  inline int maxGridSizeX() const {return prop_.maxGridSize[0];}
  inline int maxThreadsPerBlock() const {return prop_.maxThreadsPerBlock;}

private:
  cudaDeviceProp prop_;
};

namespace sparse_img_align_device_utils {

void precomputeBaseCaches(
    std::vector<Float2TypeGpu>& uv_cache,
    std::vector<Float3TypeGpu>&  xyz_ref_cache,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr>& cu_T_imu_cam_bundle,
    const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr>& cu_T_cam_imu_bundle,
    const std::vector<imp::cu::PinholeCamera::Ptr>& cu_camera_bundle,
    const size_t& nbr_fts_to_track,
    GpuCacheHandler& gpu_cache);

void precomputeJacobiansAndRefPatches(
    const std::vector<imp::ImagePyramid8uC1::Ptr>& ref_pyramid,
    const int level,
    const int patch_size,
    const bool estimate_alpha,
    const bool estimate_beta,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    GpuCacheHandler& gpu_cache);

void precomputeJacobiansAndRefPatches(
    const std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr> >& ref_pyramid,
    const int level,
    const int patch_size,
    const bool estimate_alpha,
    const bool estimate_beta,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    GpuCacheHandler& gpu_cache);

void computeResidualsOfFrame(
    const std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr>>& cur_pyramid,
    const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>>& cu_T_cur_ref_bundle,
    const std::vector<imp::cu::PinholeCamera::Ptr>& cu_camera_bundle,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    const int level,
    const int patch_size,
    const bool estimate_alpha,
    const bool estimate_beta,
    GpuCacheHandler& gpu_cache);

void computeNumBlocksAndThreadsReduction(
    const size_t nr_features, const size_t patch_area,
    const GPUProperties& gpu_props,
    int &num_blocks, int &num_threads);

FloatTypeGpu computeHessianAndGradient(
    SparseImgAlignBase::HessianMatrix* H,
    SparseImgAlignBase::GradientVector* g,
    const size_t nr_elements, const size_t patch_area,
    GpuCacheHandler& gpu_cache,
    const int num_blocks, const int num_threads);

/**
 * @brief computeDisparity computes the median distparity (check .cu file for more details) for
 * all the sucessfully reprojected features between the current and the reference frames.
 * It must be called after the image alignment has converged because it uses the
 * visibility cache to check if a features are correctly reprojected.
 * Possible extensions:
 * - We could simply check the disparity for all features without checking if
 * they are reprojected into the field of view of the current image.
 * - Calculate the median on the GPU?
 */
FloatTypeGpu computeDisparity(
    const std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr>>& cur_pyramid,
    const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>>& cu_T_cur_ref_bundle,
    const std::vector<imp::cu::PinholeCamera::Ptr>& cu_camera_bundle,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    const size_t total_number_of_features,
    GpuCacheHandler& gpu_cache);

} // namespace sparse_img_align_host_utils
} // namespace svo
