// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/img_align/sparse_img_align_device_utils.cuh>

#include <imp/cu_core/cu_texture.cuh>
#include <imp/cu_core/cu_utils.hpp>

#include <svo/common/logging.h>
#include <svo/img_align/sparse_img_align_base.h>

namespace svo {

GpuCacheHandler::GpuCacheHandler():
  patch_area_(0), feature_capacity_(0), reduction_cache_capacity_(0)
{ }


GpuCacheHandler::GpuCacheHandler(const size_t patch_area):
  patch_area_(patch_area), feature_capacity_(0), reduction_cache_capacity_(0)
{ }

void GpuCacheHandler::setPatchArea(const size_t patch_area)
{
  patch_area_ = patch_area;
}

void GpuCacheHandler::reserveFeatureCapacity(const size_t capacity)
{
  CHECK_GT(patch_area_,0);
  // Check if enough or to much memory is allocated.
  if( (capacity > feature_capacity_) || (feature_capacity_ - capacity > kMaxStorageSurplus))
  {
    SVO_WARN_STREAM("Reallocate GPU memory. Changing capacity from " << feature_capacity_ << " to "
                    << capacity << " features.");
    uv_cache_.reset(new UvCache(capacity));
    xyz_ref_cache_.reset(new XyzRefCache(capacity));
    jacobian_proj_cache_.reset(new JacobianProjCache(capacity*kJacProjStride));
    jacobian_cache_.reset(new JacobianCache(capacity*kJacStride*patch_area_));
    residual_cache_.reset(new ResidualCache(capacity*patch_area_));
    visibility_mask_.reset(new VisibilityMask(capacity));
    ref_patch_cache_.reset(new RefPatchCache(capacity*patch_area_));
    disparity_cache_.reset(new DistparitiyCache(capacity));
    feature_capacity_ = capacity;
  }
}

void GpuCacheHandler::reserveReductionCacheCapacity(const size_t capacity)
{
  // Check if enough or to much memory is allocated.
  if( (capacity > reduction_cache_capacity_) || (reduction_cache_capacity_ - capacity > kMaxStorageSurplus))
  {
    SVO_WARN_STREAM("Reallocate memory for reduction step from " << reduction_cache_capacity_ << " to "
                    << capacity << " blocks.");
    hessian_reduction_cache_.reset(new HessianReductionCache(capacity*kHessianTriagStride));
    gradient_reduction_cache_.reset(new GradientReductionCache(capacity*kJacStride));
    chi2_reduction_cache_.reset(new GradientReductionCache(capacity));
    nr_visible_cache_.reset(new NrVisibleCache(capacity));
    hessian_reduction_cache_host_.reset(new HessianReductionCacheHost(capacity*kHessianTriagStride));
    gradient_reduction_cache_host_.reset(new GradientReductionCacheHost(capacity*kJacStride));
    chi2_reduction_cache_host_.reset(new GradientReductionCacheHost(capacity));
    nr_visible_cache_host_.reset(new NrVisibleCacheHost(capacity));
    reduction_cache_capacity_ = capacity;
  }
  else
  {
    // Set region of interest to the correct value to make
    // copying from device to host possible.
    if(nr_visible_cache_host_->roi().length() != capacity)
    {
      SVO_DEBUG_STREAM("Change region of interest of linear memory (before "
                       << nr_visible_cache_host_->roi().length() << ", after "
                       << capacity << " elements)");

      hessian_reduction_cache_->setRoi(imp::Roi1u(0,capacity*kHessianTriagStride));
      gradient_reduction_cache_->setRoi(imp::Roi1u(0,capacity*kJacStride));
      chi2_reduction_cache_->setRoi(imp::Roi1u(0,capacity));
      nr_visible_cache_->setRoi(imp::Roi1u(0,capacity));
      hessian_reduction_cache_host_->setRoi(imp::Roi1u(0,capacity*kHessianTriagStride));
      gradient_reduction_cache_host_->setRoi(imp::Roi1u(0,capacity*kJacStride));
      chi2_reduction_cache_host_->setRoi(imp::Roi1u(0,capacity));
      nr_visible_cache_host_->setRoi(imp::Roi1u(0,capacity));
    }
  }
}

inline void GpuCacheHandler::copyReductionCacheDeviceToHost()
{
  hessian_reduction_cache_->copyTo(*hessian_reduction_cache_host_);
  gradient_reduction_cache_->copyTo(*gradient_reduction_cache_host_);
  chi2_reduction_cache_->copyTo(*chi2_reduction_cache_host_);
  nr_visible_cache_->copyTo(*nr_visible_cache_host_);
}

namespace sparse_img_align_device_utils
{

__host__ __device__ __forceinline__
void setGx(imp::cu::Matrix<FloatTypeGpu,3,6>& __restrict__ g_x,
           const Float3TypeGpu& __restrict__ p_in_imu)
{
  g_x(0,0) = 1.0;
  g_x(0,1) = 0.0;
  g_x(0,2) = 0.0;
  g_x(0,3) = 0.0;
  g_x(0,4) = p_in_imu.z;
  g_x(0,5) = -p_in_imu.y;
  g_x(1,0) = 0.0;
  g_x(1,1) = 1.0;
  g_x(1,2) = 0.0;
  g_x(1,3) = -p_in_imu.z;
  g_x(1,4) = 0.0;
  g_x(1,5) = p_in_imu.x;
  g_x(2,0) = 0.0;
  g_x(2,1) = 0.0;
  g_x(2,2) = 1.0;
  g_x(2,3) = p_in_imu.y;
  g_x(2,4) = -p_in_imu.x;
  g_x(2,5) = 0.0;
}

//Todo: This function should be a member function of the CPU camera
__host__ __device__ __forceinline__
void setPinholeJacobian(imp::cu::Matrix<FloatTypeGpu,2,3>& __restrict__ jac_cam,
                        const Float3TypeGpu& __restrict__ p_in_cam,
                        const FloatTypeGpu& __restrict__ focal_length)
{
  FloatTypeGpu ratio_p_x_z_cam = p_in_cam.x/p_in_cam.z;
  FloatTypeGpu ratio_p_y_z_cam = p_in_cam.y/p_in_cam.z;
  FloatTypeGpu ratio_fl_p_z_cam = focal_length/p_in_cam.z;

  jac_cam(0,0) = ratio_fl_p_z_cam;
  jac_cam(0,1) = 0.0;
  jac_cam(0,2) = -ratio_fl_p_z_cam*ratio_p_x_z_cam;
  jac_cam(1,0) = 0.0;
  jac_cam(1,1) = ratio_fl_p_z_cam;
  jac_cam(1,2) = -ratio_fl_p_z_cam*ratio_p_y_z_cam;
}

__global__ void k_baseCachesGeneric(const imp::cu::Matrix<FloatTypeGpu,3,4> T_imu_cam,
                                    const imp::cu::Matrix<FloatTypeGpu,3,3> R_imu_cam,
                                    const FloatTypeGpu focal_length,
                                    const Float3TypeGpu* __restrict__  p_in_cam,
                                    FloatTypeGpu* __restrict__ jac_proj_cache,
                                    const unsigned int nr_features)
{

  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i < nr_features)
  {
    const Float3TypeGpu p_in_imu = transform(T_imu_cam,p_in_cam[i]);
    imp::cu::Matrix<FloatTypeGpu,3,6> g_x;
    setGx(g_x,p_in_imu);

    imp::cu::Matrix<FloatTypeGpu,2,3> jac_cam;
    setPinholeJacobian(jac_cam,p_in_cam[i],focal_length);

    imp::cu::Matrix<FloatTypeGpu,2,6> jac_proj = ((jac_cam*R_imu_cam)*g_x);

    // wite to buffer
    int offset = 2*6*i;
#pragma unroll
    for(int row = 0; row < 2;++row)
    {
#pragma unroll
      for(int col = 0; col < 6; ++col)
      {
        // times (-1) because of our definition of the photometric error
        jac_proj_cache[offset + col] = -1.0f*jac_proj(row,col);
      }
      offset +=6;
    }
  }
}

__global__ void k_baseCachesPinhole(const imp::cu::Matrix<FloatTypeGpu,3,4> T_imu_cam,
                                    const imp::cu::Matrix<FloatTypeGpu,3,3> R_cam_imu,
                                    const FloatTypeGpu focal_length,
                                    const Float3TypeGpu* __restrict__  p_in_cam,
                                    FloatTypeGpu* __restrict__ jac_proj_cache,
                                    const unsigned int nr_features)
{
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < nr_features)
  {
    Float3TypeGpu p_in_imu = transform(T_imu_cam,p_in_cam[i]);
    FloatTypeGpu ratio_p_x_z_cam = p_in_cam[i].x/p_in_cam[i].z;
    FloatTypeGpu ratio_p_y_z_cam = p_in_cam[i].y/p_in_cam[i].z;
    // times (-1) because of our definition of the photometric error
    FloatTypeGpu ratio_fl_p_z_cam = (-1.0)*focal_length/p_in_cam[i].z;

    FloatTypeGpu r00 = ratio_fl_p_z_cam*(R_cam_imu(0,0) - R_cam_imu(2,0)*ratio_p_x_z_cam);
    FloatTypeGpu r01 = ratio_fl_p_z_cam*(R_cam_imu(0,1) - R_cam_imu(2,1)*ratio_p_x_z_cam);
    FloatTypeGpu r02 = ratio_fl_p_z_cam*(R_cam_imu(0,2) - R_cam_imu(2,2)*ratio_p_x_z_cam);
    FloatTypeGpu r10 = ratio_fl_p_z_cam*(R_cam_imu(1,0) - R_cam_imu(2,0)*ratio_p_y_z_cam);
    FloatTypeGpu r11 = ratio_fl_p_z_cam*(R_cam_imu(1,1) - R_cam_imu(2,1)*ratio_p_y_z_cam);
    FloatTypeGpu r12 = ratio_fl_p_z_cam*(R_cam_imu(1,2) - R_cam_imu(2,2)*ratio_p_y_z_cam);

    const int offset = 2*6*i;
    jac_proj_cache[offset] = r00;
    jac_proj_cache[offset + 1] = r01;
    jac_proj_cache[offset + 2] = r02;
    jac_proj_cache[offset + 3] = -p_in_imu.z*r01 + p_in_imu.y*r02;
    jac_proj_cache[offset + 4] = p_in_imu.z*r00 - p_in_imu.x*r02;
    jac_proj_cache[offset + 5] = -p_in_imu.y*r00 + p_in_imu.x*r01;
    jac_proj_cache[offset + 6] = r10;
    jac_proj_cache[offset + 7] = r11;
    jac_proj_cache[offset + 8] = r12;
    jac_proj_cache[offset + 9] = -p_in_imu.z*r11 + p_in_imu.y*r12;
    jac_proj_cache[offset + 10] = p_in_imu.z*r10 - p_in_imu.x*r12;
    jac_proj_cache[offset + 11] = -p_in_imu.y*r10 + p_in_imu.x*r11;
  }
}

void precomputeBaseCaches(std::vector<Float2TypeGpu>& uv_cache,
                          std::vector<Float3TypeGpu>&  xyz_ref_cache,
                          const std::vector<size_t>& first_ftr_index,
                          const std::vector<size_t>& nbr_of_ftrs,
                          const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr>& cu_T_imu_cam_bundle,
                          const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>::Ptr>& cu_T_cam_imu_bundle,
                          const std::vector<imp::cu::PinholeCamera::Ptr>& cu_camera_bundle,
                          const size_t& nbr_fts_to_track,
                          GpuCacheHandler& gpu_cache)
{
  // Prepare the GPU buffers.
  gpu_cache.reserveFeatureCapacity(nbr_fts_to_track);

  // Transfer data from CPU to GPU.
  LinearMemoryFloat2 uv_linear(reinterpret_cast<Float2PixelGpu*>(uv_cache.data()),uv_cache.size(),true);
  LinearMemoryFloat3 xyz_linear(reinterpret_cast<Float3PixelGpu*>(xyz_ref_cache.data()),xyz_ref_cache.size(),true);

  gpu_cache.uv().setRoi(uv_linear.roi());
  gpu_cache.uv().copyFrom(uv_linear);
  gpu_cache.xyzRef().setRoi(xyz_linear.roi());
  gpu_cache.xyzRef().copyFrom(xyz_linear);

  // Fill base caches.
  for(int i = 0; i< static_cast<int>(cu_camera_bundle.size()); ++i)
  {
    imp::cu::Fragmentation<32,1> frag(nbr_of_ftrs.at(i));
    k_baseCachesPinhole
        <<<
          frag.dimGrid,frag.dimBlock
        >>>(*cu_T_imu_cam_bundle.at(i),
            cu_T_cam_imu_bundle.at(i)->block<3,3>(0,0),
            cu_camera_bundle.at(i)->fx(),
            &gpu_cache.xyzRef().cuData()[first_ftr_index.at(i)],
        &gpu_cache.jacProj().cuData()[first_ftr_index.at(i)*GpuCacheHandler::kJacProjStride],
        nbr_of_ftrs.at(i));
  }
  cudaDeviceSynchronize();
}

__global__ void k_jacobianAndRefPatches(imp::cu::Texture2D ref_tex,
                                        const Float2TypeGpu* __restrict__  uv,
                                        const FloatTypeGpu* __restrict__ jac_proj_cache,
                                        const int patch_size,
                                        const int level,
                                        const unsigned int nrFeatures,
                                        FloatTypeGpu* __restrict__ jacobian_cache,
                                        FloatTypeGpu* __restrict__ ref_patch_cache)
{
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i < nrFeatures)
  {
    const FloatTypeGpu scale = 1.0f/(1<<level);
    const FloatTypeGpu patch_area = patch_size*patch_size;
    const FloatTypeGpu upper_left_coord_x = uv[i].x*scale - (patch_size - 1)/2.0f;
    const FloatTypeGpu upper_left_coord_y = uv[i].y*scale - (patch_size - 1)/2.0f;
    size_t ref_patch_index_offset = patch_area*i;
    size_t jacobian_index_offset = patch_area*GpuCacheHandler::kJacStride*i;
    size_t jac_proj_cache_index_offset = GpuCacheHandler::kJacProjStride*i;

#pragma unroll 4
    for(int row = 0; row < patch_size; ++row)
    {
#pragma unroll 4
      for(int col = 0; col < patch_size; ++col, ++ref_patch_index_offset, jacobian_index_offset += 8)
      {
        FloatTypeGpu center_texel;
        imp::cu::tex2DFetch(center_texel, ref_tex,upper_left_coord_x + col, upper_left_coord_y + row);
        ref_patch_cache[ref_patch_index_offset] = 255.0f*center_texel;
        FloatTypeGpu dx_left,dx_right,dy_up,dy_down;
        imp::cu::tex2DFetch(dx_left, ref_tex,upper_left_coord_x + col - 1, upper_left_coord_y + row);
        imp::cu::tex2DFetch(dx_right, ref_tex,upper_left_coord_x + col + 1, upper_left_coord_y + row);
        imp::cu::tex2DFetch(dy_up, ref_tex,upper_left_coord_x + col, upper_left_coord_y + row - 1);
        imp::cu::tex2DFetch(dy_down, ref_tex,upper_left_coord_x + col, upper_left_coord_y + row + 1);
        const FloatTypeGpu dx = 0.5f*(dx_right - dx_left)*255.0f;
        const FloatTypeGpu dy = 0.5f*(dy_down - dy_up)*255.0f;

#pragma unroll
        for(int i = 0; i < 6; ++i)
        {
          jacobian_cache[jacobian_index_offset + i] = (dx*(jac_proj_cache[jac_proj_cache_index_offset + i])
              + dy*(jac_proj_cache[jac_proj_cache_index_offset + 6 + i]))*scale;
        }

        //jacobian_cache[jacobian_index_offset + 6] = -255*center_texel;
        //jacobian_cache[jacobian_index_offset + 7] = -1;
        jacobian_cache[jacobian_index_offset + 6] = 0.0;
        jacobian_cache[jacobian_index_offset + 7] = 0.0;
      }
    }
  }
}

// TODO: imp::ImagePyramid version (currently not working)
//void precomputeJacobiansAndRefPatches(
//    std::vector<imp::ImagePyramid8uC1::Ptr>& ref_pyramid,
//    const int level,
//    const int patch_size,
//    const bool estimate_alpha,
//    const bool estimate_beta,
//    const std::vector<size_t>& first_ftr_index,
//    const std::vector<size_t>& nbr_of_ftrs,
//    GpuCacheHandler& gpu_cache)
//{
//  int patch_area = patch_size*patch_size;

//  for(int ii = 0; ii < static_cast<int>(pyramid.size());++ii)
//  {
//    std::shared_ptr<imp::cu::Texture2D> ref_tex  =
// std::dynamic_pointer_cast<imp::cu::ImageGpu8uC1>(ref_pyramid.at(ii)->at(level))
//        ->genTexture(false,cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat);

//    dim3 threads(32);
//    dim3 blocks((nbr_of_ftrs.at(ii) + threads.x-1)/threads.x);
//    std::cout << " features " << ii << " = " << nbr_of_ftrs.at(ii) << std::endl;
//    k_jacobianAndRefPatches<<<blocks,threads>>>(*ref_tex.get(),
//        reinterpret_cast<float2*>(&gpu_cache.uv().data()[first_ftr_index.at(ii)]),
//        reinterpret_cast<float*>(&gpu_cache.jacProj().data()[first_ftr_index.at(ii)*12]),
//        patch_size , level , nbr_of_ftrs.at(ii),
//        reinterpret_cast<float*>(&gpu_cache.jacobian().data()[first_ftr_index.at(ii)*8*patch_area]),
//        reinterpret_cast<float*>(&gpu_cache.refPatch().data()[first_ftr_index.at(ii)*patch_area]));

//  }
//  cudaDeviceSynchronize();
//}

void precomputeJacobiansAndRefPatches(
    const std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr> >& ref_pyramid,
    const int level,
    const int patch_size,
    const bool estimate_alpha,
    const bool estimate_beta,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    GpuCacheHandler& gpu_cache)
{
  int patch_area = patch_size*patch_size;

  for(int i = 0; i < static_cast<int>(ref_pyramid.size());++i)
  {
    std::shared_ptr<imp::cu::Texture2D> ref_tex  =
        std::dynamic_pointer_cast<imp::cu::ImageGpu8uC1>(ref_pyramid.at(i).at(level))
        ->genTexture(false,cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat);

    imp::cu::Fragmentation<32,1> frag(nbr_of_ftrs.at(i));
    k_jacobianAndRefPatches
        <<<
          frag.dimGrid, frag.dimBlock
        >>>(*ref_tex,&gpu_cache.uv().cuData()[first_ftr_index.at(i)],
        &gpu_cache.jacProj().cuData()[first_ftr_index.at(i)*GpuCacheHandler::kJacProjStride],
        patch_size , level , nbr_of_ftrs.at(i),
        &gpu_cache.jacobian().cuData()[first_ftr_index.at(i)*GpuCacheHandler::kJacStride*patch_area],
        &gpu_cache.refPatch().cuData()[first_ftr_index.at(i)*patch_area]);

  }
  cudaDeviceSynchronize();
}

__global__ void k_residuals(const imp::cu::Texture2D cur_tex,
                            const int width,
                            const int height,
                            const imp::cu::Matrix<FloatTypeGpu,3,4> T_cur_ref,
                            const imp::cu::PinholeCamera cam,
                            const FloatTypeGpu* __restrict__ ref_patch_cache,
                            const Float3TypeGpu* __restrict__  xyz_ref,
                            FloatTypeGpu alpha_illumination,
                            FloatTypeGpu beta_illumionation,
                            const int patch_size,
                            const int level,
                            const unsigned int nrFeatures,
                            BoolTypeGpu* __restrict__ visibility_cache,
                            FloatTypeGpu* __restrict__ residual_cache)
{
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i < nrFeatures)
  {
    FloatTypeGpu scale = 1.0f/(1<<level);
    const int patch_area = patch_size*patch_size;
    Float2TypeGpu uv_cur = static_cast<Float2TypeGpu>(
          cam.world2cam(static_cast<Float3TypeGpu>(transform(T_cur_ref,xyz_ref[i]))));
    const FloatTypeGpu upper_left_coord_x = uv_cur.x*scale - (patch_size - 1)/2.0f;
    const FloatTypeGpu upper_left_coord_y = uv_cur.y*scale - (patch_size - 1)/2.0f;

    // Check if projection is within the image.
    if(upper_left_coord_x  < 0.0f || upper_left_coord_y  < 0.0f
       || upper_left_coord_x + patch_size >= width - 1 || upper_left_coord_y + patch_size >= height - 1)
    {
      visibility_cache[i] = 0;
    }
    else
    {
      visibility_cache[i] = 1;

      int pixel = 0;
#pragma unroll 4
      for(int row = 0; row < patch_size; ++row)
      {
#pragma unroll 4
        for(int col = 0; col < patch_size; ++col,++pixel)
        {
          FloatTypeGpu cur_Texel;
          imp::cu::tex2DFetch(cur_Texel, cur_tex,upper_left_coord_x + col, upper_left_coord_y + row);
          residual_cache[i*patch_area + pixel] = static_cast<FloatTypeGpu>(
                255.0*cur_Texel*(1.0 + alpha_illumination) + beta_illumionation)
              - ref_patch_cache[i*patch_area + pixel];
        }
      }
    }
  }
}

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
    GpuCacheHandler& gpu_cache)
{
  const size_t patch_area = patch_size*patch_size;
  for(int i = 0; i < static_cast<int>(cur_pyramid.size());++i)
  {
    std::shared_ptr<imp::cu::Texture2D> cur_tex  = std::dynamic_pointer_cast<imp::cu::ImageGpu8uC1>(
          cur_pyramid.at(i).at(level))
        ->genTexture(false,cudaFilterModeLinear,cudaAddressModeBorder,cudaReadModeNormalizedFloat);
    imp::cu::Fragmentation<32,1> frag(nbr_of_ftrs.at(i));
    k_residuals
        <<<
          frag.dimGrid,frag.dimBlock
        >>>(*cur_tex, cur_pyramid.at(i).at(level)->width(),
            cur_pyramid.at(i).at(level)->height(),
            cu_T_cur_ref_bundle.at(i),
            *cu_camera_bundle.at(i),
            &gpu_cache.refPatch().cuData()[first_ftr_index.at(i)*patch_area],
        &gpu_cache.xyzRef().cuData()[first_ftr_index.at(i)],
        estimate_alpha, estimate_beta, patch_size, level, nbr_of_ftrs.at(i),
        &gpu_cache.visibility().cuData()[first_ftr_index.at(i)],
        &gpu_cache.residual().cuData()[first_ftr_index.at(i)*patch_area]);

  }
  cudaDeviceSynchronize();
}

inline unsigned int nextPow2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

bool isPow2(unsigned int x)
{
  return ((x&(x-1))==0);
}

template <size_t _n_elements>
__host__ __device__ __forceinline__
void setToZero(FloatTypeGpu*  mem)
{
#pragma unroll
  for(int ind = 0; ind < _n_elements; ++ind)
  {
    mem[ind] = 0.0;
  }
}

template <size_t _matrix_size>
__host__ __device__ __forceinline__
void setVVTUpperTriag(FloatTypeGpu* __restrict__ upper_triag_row_maj,
                      const FloatTypeGpu* __restrict__ vect,
                      const FloatTypeGpu& __restrict__ weight = 1.0)
{
  int index = 0;
#pragma unroll
  for(int row = 0; row < _matrix_size; ++row)
  {
#pragma unroll
    for(int col = row; col < _matrix_size; ++col,++index)
    {
      upper_triag_row_maj[index] = weight*vect[row]*vect[col];
    }
  }
}

template <size_t _matrix_size>
__host__ __device__ __forceinline__
void addVVTUpperTriag(FloatTypeGpu* __restrict__ upper_triag_row_maj,
                      const FloatTypeGpu* __restrict__ vect,
                      const FloatTypeGpu& __restrict__ weight = 1.0)
{
  int index = 0;
#pragma unroll
  for(int row = 0; row < _matrix_size; ++row)
  {
#pragma unroll
    for(int col = row; col < _matrix_size; ++col,++index)
    {
      upper_triag_row_maj[index] += weight*vect[row]*vect[col];
    }
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__
void addVector(FloatTypeGpu* __restrict__ sum_vect,
               const FloatTypeGpu* __restrict__ addend_vect)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    sum_vect[ind] += addend_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__
void addWeightedVector(FloatTypeGpu* __restrict__ sum_vect,
                       const FloatTypeGpu* __restrict__ addend_vect,
                       const FloatTypeGpu& __restrict__ weight = 1.0)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    sum_vect[ind] += weight*addend_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__
void subWeightedVector(FloatTypeGpu* __restrict__ sum_vect,
                       const FloatTypeGpu* __restrict__ addend_vect,
                       const FloatTypeGpu& __restrict__ weight = 1.0)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    sum_vect[ind] -= weight*addend_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__
void setWeightedVector(FloatTypeGpu* __restrict__ dest_vect,
                       const FloatTypeGpu* __restrict__ src_vect,
                       const FloatTypeGpu& __restrict__ weight = 1.0)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    dest_vect[ind] = weight*src_vect[ind];
  }
}

template <size_t _vector_size>
__host__ __device__ __forceinline__
void copyVector(FloatTypeGpu* __restrict__ dest_vect,
                const FloatTypeGpu* __restrict__ src_vect)
{

#pragma unroll
  for(int ind = 0; ind < _vector_size; ++ind)
  {
    dest_vect[ind] = src_vect[ind];
  }
}

// _block_size must be power of 2
template <unsigned int _block_size, bool n_is_pow2>
__global__ void k_reduceHessianGradient(const FloatTypeGpu* __restrict__ jacobian_cache,
                                        const FloatTypeGpu* __restrict__ residual_cache,
                                        const BoolTypeGpu* __restrict__ visibility_cache,
                                        FloatTypeGpu* __restrict__ gradient_cache,
                                        FloatTypeGpu* __restrict__ hessian_cache,
                                        UIntTypeGpu* __restrict__ nr_meas,
                                        FloatTypeGpu* __restrict__ chi2,
                                        const unsigned int n_elements,
                                        const unsigned int patch_area)
{
  constexpr unsigned int kHessianTriagN = SparseImgAlignBase::kHessianTriagN;
  constexpr unsigned int kJacobianSize = SparseImgAlignBase::kJacobianSize;

  __shared__ FloatTypeGpu s_hessian_data[_block_size*kHessianTriagN];
  __shared__ FloatTypeGpu s_gradient_data[_block_size*kJacobianSize];
  __shared__ FloatTypeGpu s_chi2[_block_size];
  __shared__ UIntTypeGpu s_chi2_nr_meas[_block_size];
  FloatTypeGpu jacobian[kJacobianSize];
  FloatTypeGpu gradient[kJacobianSize];
  FloatTypeGpu hessian[kHessianTriagN];
  FloatTypeGpu chi2_temp;
  UIntTypeGpu chi2_nr_meas = 0;

  const unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*_block_size*2 + threadIdx.x;
  const unsigned int gridSize = _block_size*2*gridDim.x;
  const unsigned int hessian_index = tid*kHessianTriagN;
  const unsigned int gradient_index = tid*kJacobianSize;

  // We reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  // We reading from global memory and write to shared memory.

  // Get first element.
  if((!n_is_pow2)&&(i >= n_elements))
  {
    setToZero<kJacobianSize>(gradient);
    setToZero<kHessianTriagN>(hessian);
    chi2_temp = 0.0;
  }
  else
  {
    const unsigned int visib_index = i/patch_area;
    BoolTypeGpu visible = visibility_cache[visib_index];
    if(visible == 1)
    {
      FloatTypeGpu residual = residual_cache[i];
      // TODO: add weighting function
      FloatTypeGpu weight = 1.0;// weight_function(residual/weight_scale);

      copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
      setVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
      setWeightedVector<kJacobianSize>(gradient,jacobian, -weight*residual);
      chi2_temp = residual*residual*weight;
      ++chi2_nr_meas;
    }
    else
    {
      setToZero<kJacobianSize>(gradient);
      setToZero<kHessianTriagN>(hessian);
      chi2_temp = 0.0;
    }

    // Get second element.
    // Ensure we don't read out of bounds -- this is optimized away for powerOf2 problem size.
    if (n_is_pow2 || i + _block_size < n_elements)
    {
      i += _block_size;
      const unsigned int visib_index = i/patch_area;
      BoolTypeGpu visible = visibility_cache[visib_index];
      if(visible == 1)
      {
        FloatTypeGpu residual = residual_cache[i];
        //TODO: add weighting function
        FloatTypeGpu weight = 1.0;//weight_function(residual/weight_scale);

        copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
        addVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
        subWeightedVector<kJacobianSize>(gradient,jacobian, weight*residual);
        chi2_temp += residual*residual*weight;
        ++chi2_nr_meas;
      }
    }
    i += (gridSize - _block_size);
  }

  // Add further elements if available.
  while (i < n_elements)
  {
    const unsigned int visib_index = i/patch_area;
    BoolTypeGpu visible = visibility_cache[visib_index];
    if(visible == 1)
    {
      FloatTypeGpu residual = residual_cache[i];
      //TODO: add weighting function
      FloatTypeGpu weight = 1.0;// weight_function(residual/weight_scale);

      copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
      addVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
      subWeightedVector<kJacobianSize>(gradient,jacobian, weight*residual);
      chi2_temp += residual*residual*weight;
      ++chi2_nr_meas;
    }

    // Add second element.
    // Ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays.
    if (n_is_pow2 || i + _block_size < n_elements)
    {
      i += _block_size;
      const unsigned int visib_index = i/patch_area;
      BoolTypeGpu visible = visibility_cache[visib_index];
      if(visible == 1)
      {
        FloatTypeGpu residual = residual_cache[i];
        //TODO: add weighting function
        FloatTypeGpu weight = 1.0;// visible*weight_function(residual/weight_scale);

        copyVector<kJacobianSize>(jacobian,&jacobian_cache[i*kJacobianSize]);
        addVVTUpperTriag<kJacobianSize>(hessian,jacobian,weight);
        subWeightedVector<kJacobianSize>(gradient,jacobian, weight*residual);
        chi2_temp += residual*residual*weight;
        ++chi2_nr_meas;
      }
    }
    i += (gridSize - _block_size);
  }

  // Each thread puts its local sum into shared memory.
  copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
  copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
  s_chi2[tid] = chi2_temp;
  s_chi2_nr_meas[tid] = chi2_nr_meas;
  __syncthreads();

  // Do reduction in shared mem.
  if ((_block_size >= 512) && (tid < 256))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 256)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 256)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 256];
    chi2_nr_meas += s_chi2_nr_meas[tid + 256];

    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >= 256) &&(tid < 128))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 128)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 128)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 128];
    chi2_nr_meas += s_chi2_nr_meas[tid + 128];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >= 128) && (tid <  64))
  {
    // add to local variable
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 64)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 64)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 64];
    chi2_nr_meas += s_chi2_nr_meas[tid + 64];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  // TODO: __shfl_down can be used for reduction when only a single warp (32 threads) is left.
  // #if (__CUDA_ARCH__ >= 300 )
  // Do reduction with __shfl_down ...
  // #else
  // Fully unroll reduction within a single warp. Theoretically __syncthreads() is not necessary anymore
  // as all threads are in the same warp. But with __syncthreads() the performance seems to be slightly increased.
  // The reason for this is not yet clear.

  if ((_block_size >=  64) && (tid < 32))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 32)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 32)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 32];
    chi2_nr_meas += s_chi2_nr_meas[tid + 32];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >=  32) && (tid < 16))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 16)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 16)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 16];
    chi2_nr_meas += s_chi2_nr_meas[tid + 16];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >=  16) && (tid <  8))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 8)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 8)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 8];
    chi2_nr_meas += s_chi2_nr_meas[tid + 8];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >=   8) && (tid <  4))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 4)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 4)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 4];
    chi2_nr_meas += s_chi2_nr_meas[tid + 4];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >=   4) && (tid <  2))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 2)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 2)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 2];
    chi2_nr_meas += s_chi2_nr_meas[tid + 2];
    // Store result to shared memory.
    copyVector<kJacobianSize>(&s_gradient_data[gradient_index],gradient);
    copyVector<kHessianTriagN>(&s_hessian_data[hessian_index],hessian);
    s_chi2[tid] = chi2_temp;
    s_chi2_nr_meas[tid] = chi2_nr_meas;
  }

  __syncthreads();

  if ((_block_size >=   2) && ( tid <  1))
  {
    // Add to local variable.
    addVector<kJacobianSize>(gradient,&s_gradient_data[(tid + 1)*kJacobianSize]);
    addVector<kHessianTriagN>(hessian,&s_hessian_data[(tid + 1)*kHessianTriagN]);
    chi2_temp += s_chi2[tid + 1];
    chi2_nr_meas += s_chi2_nr_meas[tid + 1];
  }

  __syncthreads();

  // Write result for this block to global memory.
  if (tid == 0)
  {
    copyVector<kJacobianSize>(&gradient_cache[blockIdx.x*kJacobianSize],gradient);
    copyVector<kHessianTriagN>(&hessian_cache[blockIdx.x*kHessianTriagN],hessian);
    chi2[blockIdx.x] = chi2_temp;
    nr_meas[blockIdx.x] = chi2_nr_meas;
  }
}

void reduceHessianGradient(const size_t size,
                           const size_t patch_area,
                           const int threads,
                           const int blocks,
                           const FloatTypeGpu* __restrict__ jacobian_input_device,
                           const BoolTypeGpu* __restrict__ visibility_input_device,
                           const FloatTypeGpu* __restrict__ residual_input_device,
                           FloatTypeGpu* __restrict__ gradient_output,
                           FloatTypeGpu* __restrict__ hessian_output,
                           UIntTypeGpu* __restrict__ nrMeas,
                           FloatTypeGpu* __restrict__ chi2)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  if (isPow2(size))
  {
    switch (threads)
    {

    case 512:
      SVO_ERROR_STREAM(" 512 threads exceed the 48kB of available shared memory per block!");
      //      k_jacobianReduceHessianGradient<512, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
      //                                                                          residual_input_device,
      //                                                                          visibility_input_device,
      //                                                                          gradient_output,
      //                                                                          hessian_output,
      //                                                                          size, patch_area);
      break;

    case 256:
      k_reduceHessianGradient<256, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                  residual_input_device,
                                                                  visibility_input_device,
                                                                  gradient_output,
                                                                  hessian_output,
                                                                  nrMeas,
                                                                  chi2,
                                                                  size,
                                                                  patch_area);
      break;

    case 128:
      k_reduceHessianGradient<128, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                  residual_input_device,
                                                                  visibility_input_device,
                                                                  gradient_output,
                                                                  hessian_output,
                                                                  nrMeas,
                                                                  chi2,
                                                                  size,
                                                                  patch_area);
      break;

    case 64:
      k_reduceHessianGradient<64, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;

    case 32:
      k_reduceHessianGradient<32, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;

    case 16:
      k_reduceHessianGradient<16, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;

    case  8:
      k_reduceHessianGradient<8, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                residual_input_device,
                                                                visibility_input_device,
                                                                gradient_output,
                                                                hessian_output,
                                                                nrMeas,
                                                                chi2,
                                                                size,
                                                                patch_area);
      break;

    case  4:
      k_reduceHessianGradient<4, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                residual_input_device,
                                                                visibility_input_device,
                                                                gradient_output,
                                                                hessian_output,
                                                                nrMeas,
                                                                chi2,
                                                                size,
                                                                patch_area);
      break;

    case  2:
      k_reduceHessianGradient<2, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                residual_input_device,
                                                                visibility_input_device,
                                                                gradient_output,
                                                                hessian_output,
                                                                nrMeas,
                                                                chi2,
                                                                size,
                                                                patch_area);
      break;

    case  1:
      k_reduceHessianGradient<1, true><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                residual_input_device,
                                                                visibility_input_device,
                                                                gradient_output,
                                                                hessian_output,
                                                                nrMeas,
                                                                chi2,
                                                                size,
                                                                patch_area);
      break;
    default:
      SVO_ERROR_STREAM("The block size must be a power of 2 for the reduction step! Block size is " << threads << ".");
      break;
    }
  }
  else
  {
    switch (threads)
    {
    case 512:
      SVO_ERROR_STREAM(" 512 threads exceed the 48kB of available shared memory per block!");
      //      k_reduceHessianGradient<256, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
      //                                                                   residual_input_device,
      //                                                                   visibility_input_device,
      //                                                                   gradient_output,
      //                                                                   hessian_output,
      //                                                                   nrMeas,
      //                                                                   chi2,
      //                                                                   size,patch_area);
      break;

    case 256:
      k_reduceHessianGradient<256, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                   residual_input_device,
                                                                   visibility_input_device,
                                                                   gradient_output,
                                                                   hessian_output,
                                                                   nrMeas,
                                                                   chi2,
                                                                   size,
                                                                   patch_area);
      break;

    case 128:
      k_reduceHessianGradient<128, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                   residual_input_device,
                                                                   visibility_input_device,
                                                                   gradient_output,
                                                                   hessian_output,
                                                                   nrMeas,
                                                                   chi2,
                                                                   size,
                                                                   patch_area);
      break;

    case 64:
      k_reduceHessianGradient<64, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                  residual_input_device,
                                                                  visibility_input_device,
                                                                  gradient_output,
                                                                  hessian_output,
                                                                  nrMeas,
                                                                  chi2,
                                                                  size,
                                                                  patch_area);
      break;

    case 32:
      k_reduceHessianGradient<32, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                  residual_input_device,
                                                                  visibility_input_device,
                                                                  gradient_output,
                                                                  hessian_output,
                                                                  nrMeas,
                                                                  chi2,
                                                                  size,
                                                                  patch_area);
      break;

    case 16:
      k_reduceHessianGradient<16, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                  residual_input_device,
                                                                  visibility_input_device,
                                                                  gradient_output,
                                                                  hessian_output,
                                                                  nrMeas,
                                                                  chi2,
                                                                  size,
                                                                  patch_area);
      break;

    case  8:
      k_reduceHessianGradient<8, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;

    case  4:
      k_reduceHessianGradient<4, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;

    case  2:
      k_reduceHessianGradient<2, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;

    case  1:
      k_reduceHessianGradient<1, false><<< dimGrid, dimBlock >>>(jacobian_input_device,
                                                                 residual_input_device,
                                                                 visibility_input_device,
                                                                 gradient_output,
                                                                 hessian_output,
                                                                 nrMeas,
                                                                 chi2,
                                                                 size,
                                                                 patch_area);
      break;
    default:
      SVO_ERROR_STREAM("The block size must be a power of 2 for the reduction step! Block size is " << threads << ".");
      break;
    }
  }
}

void getNumBlocksAndThreads(const size_t nr_elements,
                            const int max_grid_size_device,
                            const int max_block_size_device,
                            const int max_blocks ,
                            const int max_threads,
                            const int elements_per_thread,
                            int &blocks, int &threads)
{
  threads = (nr_elements < static_cast<size_t>(max_threads)*2) ? nextPow2((nr_elements + 1)/ 2) : max_threads;
  blocks = (nr_elements + (threads * elements_per_thread - 1)) / (threads * elements_per_thread);

  if ((float)threads*blocks > (float)max_grid_size_device * max_block_size_device)
  {
    throw std::runtime_error("Desired number of threads is too large.");
  }

  if(blocks > max_blocks)
  {
    blocks = max_blocks;
  }

  //TODO: comment this block if max_block is for sure < max_grid_size_device
  if (blocks > max_grid_size_device)
  {
    std::cout << "Desired number of blocks is bigger then the maximum grid size of the target device."
              << std::endl;
    blocks /= 2;
    threads *= 2;
  }
}

void computeNumBlocksAndThreadsReduction(const size_t nr_features,
                                         const size_t patch_area,
                                         const GPUProperties& gpu_props,
                                         int &num_blocks, int &num_threads)
{
  const int max_threads = 256;
  const int max_blocks = 64;
  const size_t nr_elements = nr_features*patch_area;

  // To reduce data of size N, log(N) elements should be reduced per thread for best performance.
  // (c.f. cuda reduction example)
  const int nr_elements_per_thread = std::max(
        static_cast<int>(std::floor(log2 (static_cast<double>(nr_elements)))),2);

  getNumBlocksAndThreads(nr_elements, gpu_props.maxGridSizeX(), gpu_props.maxThreadsPerBlock(),
                         max_blocks, max_threads, nr_elements_per_thread, num_blocks, num_threads);
}

void reduceHessianGradientCPU(const int num_blocks,
                              const LinearMemoryFloat& __restrict__ gradient_input_host,
                              const LinearMemoryFloat& __restrict__ hessian_input_host,
                              const LinearMemoryUInt& __restrict__ nMeas_input_host,
                              const LinearMemoryFloat& __restrict__ chi2_input_host,
                              FloatTypeGpu gradient_out[],
                              FloatTypeGpu hessian_out[],
                              FloatTypeGpu& chi2)
{
  memset(hessian_out,0,SparseImgAlignBase::kHessianTriagN*sizeof(FloatTypeGpu));
  memset(gradient_out,0,SparseImgAlignBase::kJacobianSize*sizeof(FloatTypeGpu));
  chi2 = 0;
  unsigned int n_meas = 0;

#pragma unroll 5
  for(unsigned int block = 0; block< static_cast<unsigned int>(num_blocks); ++block)
  {
#pragma unroll
    for(unsigned int i = 0; i < SparseImgAlignBase::kHessianTriagN; ++i)
    {
      hessian_out[i] += hessian_input_host[block*SparseImgAlignBase::kHessianTriagN + i];
    }
#pragma unroll
    for(unsigned int i = 0; i < SparseImgAlignBase::kJacobianSize; ++i)
    {
      gradient_out[i] += gradient_input_host[block*SparseImgAlignBase::kJacobianSize + i];
    }

    n_meas += nMeas_input_host[block];
    chi2 += chi2_input_host[block];
  }

  chi2 = chi2/n_meas;
}

FloatTypeGpu computeHessianAndGradient(SparseImgAlignBase::HessianMatrix* H,
                                       SparseImgAlignBase::GradientVector* g,
                                       const size_t nr_elements,
                                       const size_t patch_area,
                                       GpuCacheHandler& gpu_cache,
                                       const int num_blocks, const int num_threads)
{
  FloatTypeGpu hessian_triag[SparseImgAlignBase::kHessianTriagN];
  FloatTypeGpu gradient[SparseImgAlignBase::kJacobianSize];


  reduceHessianGradient(nr_elements, patch_area, num_threads, num_blocks,
                        gpu_cache.jacobian().cuData(),
                        gpu_cache.visibility().cuData(),
                        gpu_cache.residual().cuData(),
                        gpu_cache.gradientDevice().cuData(),
                        gpu_cache.hessianDevice().cuData(),
                        gpu_cache.nrVisibleDevice().cuData(),
                        gpu_cache.chi2Device().cuData());
  cudaDeviceSynchronize();

  // Sum the results of each block on CPU.
  FloatTypeGpu chi2;
  gpu_cache.copyReductionCacheDeviceToHost();
  reduceHessianGradientCPU(num_blocks,
                           gpu_cache.gradientHost(),
                           gpu_cache.hessianHost(),
                           gpu_cache.nrVisibleHost(),
                           gpu_cache.chi2Host(),
                           gradient,
                           hessian_triag,chi2);


  // Copy result to H and g.
#pragma unroll
  for(unsigned int row = 0, index = 0; row < SparseImgAlignBase::kJacobianSize; ++row)
  {
#pragma unroll
    for(unsigned int col = row; col < SparseImgAlignBase::kJacobianSize; ++col,++index)
    {
      (*H)(row,col) = (*H)(col,row) =  hessian_triag[index];
    }
  }

#pragma unroll
  for(unsigned int index = 0; index < SparseImgAlignBase::kJacobianSize; ++index)
  {
    (*g)(index,0) = gradient[index];
  }

  return chi2;
}

__global__ void k_disparities(const imp::cu::Matrix<FloatTypeGpu,3,4> T_cur_ref,
                              const imp::cu::PinholeCamera cam,
                              const Float3TypeGpu* __restrict__  xyz_ref,
                              const BoolTypeGpu* __restrict__ visibility_cache,
                              const Float2TypeGpu* __restrict__  uv,
                              const unsigned int nrFeatures,
                              const FloatTypeGpu not_visible_value,
                              FloatTypeGpu* __restrict__  disparity)
{
  const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

  if(i < nrFeatures)
  {
    // Check if projection is within the image.
    if(visibility_cache[i] == 1)
    {
      Float2TypeGpu uv_cur = static_cast<Float2TypeGpu>(
            cam.world2cam(static_cast<Float3TypeGpu>(transform(T_cur_ref,xyz_ref[i]))));
      Float2TypeGpu disparity_vec = make_float2(uv[i].x -  uv_cur.x, uv[i].y - uv_cur.y);
      disparity[i] = sqrt(disparity_vec.x*disparity_vec.x + disparity_vec.y*disparity_vec.y);
    }
    else
    {
      disparity[i] = not_visible_value;
    }
  }
}

template<class T>
size_t copyArrayNonNegative(T* __restrict__ dest, T* __restrict__ src,size_t number_elements)
{
  size_t number_elements_copied = 0;
  for(size_t i = 0; i < number_elements; ++i)
  {
    if(src[i] > 0)
    {
      dest[number_elements_copied++] = src[i];
    }
  }
  return number_elements_copied;
}

FloatTypeGpu computeDisparity(
    const std::vector<std::vector<imp::cu::ImageGpu8uC1::Ptr>>& cur_pyramid,
    const std::vector<imp::cu::Matrix<FloatTypeGpu,3,4>>& cu_T_cur_ref_bundle,
    const std::vector<imp::cu::PinholeCamera::Ptr>& cu_camera_bundle,
    const std::vector<size_t>& first_ftr_index,
    const std::vector<size_t>& nbr_of_ftrs,
    const size_t total_number_of_features,
    GpuCacheHandler& gpu_cache)
{
  for(int i = 0; i < static_cast<int>(cur_pyramid.size());++i)
  {
    imp::cu::Fragmentation<32,1> frag(nbr_of_ftrs.at(i));

    k_disparities
        <<<
          frag.dimGrid,frag.dimBlock
        >>>(cu_T_cur_ref_bundle.at(i),
            *cu_camera_bundle.at(i),
            &gpu_cache.xyzRef().cuData()[first_ftr_index.at(i)],
        &gpu_cache.visibility().cuData()[first_ftr_index.at(i)],
        &gpu_cache.uv().cuData()[first_ftr_index.at(i)],
        nbr_of_ftrs.at(i),
        FLT_MAX,
        &gpu_cache.disparity().cuData()[first_ftr_index.at(i)]);

  }
  cudaDeviceSynchronize();

  // Transfer disparities from GPU to CPU
  FloatTypeGpu* disparity_pointer;
  disparity_pointer = (FloatTypeGpu*) malloc(total_number_of_features*sizeof(FloatTypeGpu));
  cudaMemcpy(disparity_pointer,gpu_cache.disparity().cuData(),total_number_of_features*sizeof(FloatTypeGpu),cudaMemcpyDeviceToHost);

  /// If the "not_visible_value" is set to FLT_MAX, the median value is allways overestimating
  /// the true median value. We could also remove the values that are not reprojected by setting
  /// "not_visible_value" in k_disparities to -1 and than run the following operations
  //  {
  //    FloatTypeGpu* disparity_pointer_no_neg;
  //    disparity_pointer_no_neg = (FloatTypeGpu*) malloc(total_number_of_features*sizeof(FloatTypeGpu));
  //    size_t successfull_reprojections = copyArrayNonNegative<FloatTypeGpu>(disparity_pointer_no_neg,disparity_pointer,total_number_of_features);

  //    FloatTypeGpu* disparity_end = &disparity_pointer_no_neg[successfull_reprojections - 1];
  //    FloatTypeGpu* middle_ptr = &disparity_pointer_no_neg[successfull_reprojections/2];
  //    std::nth_element(disparity_pointer_no_neg, middle_ptr, disparity_end);

  //    std::cout << "Median with removal = " << *middle_ptr << std::endl;
  //    std::cout << "Total " << total_number_of_features << std::endl;
  //    std::cout << "successfull " << successfull_reprojections << std::endl;
  //    free(disparity_pointer_no_neg);
  //  }

  FloatTypeGpu* disparity_end = &disparity_pointer[total_number_of_features - 1];
  FloatTypeGpu* middle_ptr = &disparity_pointer[total_number_of_features/2];
  std::nth_element(disparity_pointer, middle_ptr, disparity_end);
  free(disparity_pointer);

  return *middle_ptr;
}

} // namespace sparse_img_align_device_utils
} // namespace svo
