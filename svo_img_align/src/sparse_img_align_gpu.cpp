// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/img_align/sparse_img_align_gpu.h>

#include <algorithm>

#include <opencv2/highgui/highgui.hpp>

#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <aslam/cameras/camera-pinhole.h>
#include <imp/cu_core/cu_image_gpu.cuh>
#include <imp/cu_core/cu_matrix.cuh>
#include <imp/cu_core/cu_pinhole_camera.cuh>
#include <imp/bridge/opencv/image_cv.hpp>
//TODO: remove
#include <imp/bridge/opencv/cu_cv_bridge.hpp>

#include <svo/common/logging.h>
#include <svo/common/point.h>
#include <svo/direct/depth_filter.h>
#include <svo/img_align/sparse_img_align_device_utils.cuh>
#include <svo/img_align/frame_gpu.h>

namespace svo {

SparseImgAlignGpu::SparseImgAlignGpu(
    SolverOptions optimization_options,
    SparseImgAlignOptions options)
  : SparseImgAlignBase(optimization_options,options)
{
  setPatchSize<SparseImgAlignGpu>(4);
}

size_t SparseImgAlignGpu::run(
    const FrameBundle::Ptr& ref_frames,
    const FrameBundle::Ptr& cur_frames)
{
  CHECK(!ref_frames->empty());
  CHECK_EQ(ref_frames->size(),cur_frames->size());

  // Set member variables
  ref_frames_ = ref_frames;
  cur_frames_ = cur_frames;
  T_iref_world_ = ref_frames->at(0)->T_imu_world();

  // Transform to GPU datatypes. TODO: This should be done only ones for transformation/camera data
  int nbr_cameras = static_cast<int>(ref_frames->frames_.size());
  cu_ref_imgs_pyramid_copy_.resize(nbr_cameras);
  cu_cur_imgs_pyramid_copy_.resize(nbr_cameras);
  cu_T_imu_cam_bundle_.resize(nbr_cameras);
  cu_T_cam_imu_bundle_.resize(nbr_cameras);
  cu_camera_bundle_.resize(nbr_cameras);

  // check if Gpu frame and if gpu frame data is initialized
  // TOTO (mg): modify sparse image align gpu function to take frame bundels.
  if(dynamic_cast<FrameGpu*>(ref_frames->at(0).get()) == nullptr || !dynamic_cast<FrameGpu*>(ref_frames->at(0).get())->cu_camera_)
  {

    SVO_WARN_STREAM("Sparse image align: Input frame is not a GPU. Creating GPU data structures now.");
    // Input is not a GPU frame bundle. GPU datastructures need to be instantiated.
    for(int i = 0; i < nbr_cameras; ++i)
    {
      cu_ref_imgs_pyramid_copy_.at(i).resize(options_.max_level + 1);
      cu_cur_imgs_pyramid_copy_.at(i).resize(options_.max_level + 1);

      for(int j = 0; j < static_cast<int>(cu_ref_imgs_pyramid_copy_.at(i).size()); ++j)
      {
        cu_ref_imgs_pyramid_copy_.at(i).at(j) = std::make_shared<imp::cu::ImageGpu8uC1>(
              imp::cu::ImageGpu8uC1(imp::ImageCv8uC1(ref_frames_->frames_.at(i)->img_pyr_.at(j))));
        cu_cur_imgs_pyramid_copy_.at(i).at(j) = std::make_shared<imp::cu::ImageGpu8uC1>(
              imp::cu::ImageGpu8uC1(imp::ImageCv8uC1(cur_frames_->frames_.at(i)->img_pyr_.at(j))));
      }

      cu_T_imu_cam_bundle_.at(i) = std::make_shared<imp::cu::Matrix<FloatTypeGpu,3,4> >(
            imp::cu::Matrix<FloatTypeGpu,3,4>(
              ref_frames->frames_.at(i)->T_imu_cam().getTransformationMatrix().block<3,4>(0,0).cast<FloatTypeGpu>()));
      cu_T_cam_imu_bundle_.at(i) = std::make_shared<imp::cu::Matrix<FloatTypeGpu,3,4> >(
            imp::cu::Matrix<FloatTypeGpu,3,4>(
              ref_frames->frames_.at(i)->T_cam_imu().getTransformationMatrix().block<3,4>(0,0).cast<FloatTypeGpu>()));

      aslam::PinholeCamera* pinhole = dynamic_cast<aslam::PinholeCamera*>(ref_frames->frames_.at(i)->cam().get());

      if((pinhole != nullptr) && (pinhole->getDistortion().getType() == aslam::Distortion::Type::kNoDistortion))
      {
        cu_camera_bundle_.at(i) = std::make_shared<imp::cu::PinholeCamera>(
              imp::cu::PinholeCamera(pinhole->fu(),pinhole->fv(),pinhole->cu(),pinhole->cv()));
      }
      else
      {
        SVO_ERROR_STREAM("Camera model not supported yet for GPU version");
        return 0;
      }
    }

  }
  else
  {
    // Input is a GPU frame bundle. Simply copy the pointers to the GPU data.
    for(int i = 0; i < nbr_cameras; ++i)
    {
      cu_ref_imgs_pyramid_copy_.at(i).resize(options_.max_level + 1);
      cu_cur_imgs_pyramid_copy_.at(i).resize(options_.max_level + 1);

      for(int j = 0; j < static_cast<int>(cu_ref_imgs_pyramid_copy_.at(i).size()); ++j)
      {
        cu_ref_imgs_pyramid_copy_.at(i).at(j) = std::static_pointer_cast<FrameGpu>(ref_frames_->at(i))->cu_img_pyramid_copy_.at(j);
        cu_cur_imgs_pyramid_copy_.at(i).at(j) = std::static_pointer_cast<FrameGpu>(cur_frames_->at(i))->cu_img_pyramid_copy_.at(j);
      }

      cu_T_imu_cam_bundle_.at(i) = std::static_pointer_cast<FrameGpu>(ref_frames_->at(i))->cu_T_imu_cam_;
      cu_T_cam_imu_bundle_.at(i) =  std::static_pointer_cast<FrameGpu>(ref_frames_->at(i))->cu_T_cam_imu_;
      cu_camera_bundle_.at(i) = std::static_pointer_cast<FrameGpu>(ref_frames_->at(i))->cu_camera_;
    }
  }

  // Clear caches on host. Capacity remains unchanged.
  host_cache_.clear();

  // Select all visible features and subsample if required.
  size_t nbr_fts_to_track = 0;
  size_t nbr_extracted = 0;
  for(auto frame : ref_frames->frames_)
  {
    host_cache_.first_ftr_index.push_back(nbr_fts_to_track);
    sparse_img_align_host_utils::extractFeaturesSubset(*frame,
                                                       options_.max_level,
                                                       patch_size_with_border_,
                                                       nbr_extracted,
                                                       host_cache_);
    nbr_fts_to_track += nbr_extracted;
    host_cache_.nbr_of_ftrs.push_back(nbr_extracted);
  }
  host_cache_.total_nbr_of_ftrs = nbr_fts_to_track;

  // The variable to be optimized is the imu-pose of the current frame.
  Transformation T_icur_iref =
      cur_frames_->at(0)->T_imu_world() * T_iref_world_.inverse();

  SparseImgAlignState state;
  state.T_icur_iref = T_icur_iref;
  state.alpha = alpha_init_;
  state.beta = beta_init_;

  // Precompute values common to all pyramid levels.
  sparse_img_align_device_utils::precomputeBaseCaches(host_cache_.uv_cache,
                                                      host_cache_.xyz_ref_cache,
                                                      host_cache_.first_ftr_index,
                                                      host_cache_.nbr_of_ftrs,
                                                      cu_T_imu_cam_bundle_,
                                                      cu_T_cam_imu_bundle_,
                                                      cu_camera_bundle_,
                                                      host_cache_.total_nbr_of_ftrs,
                                                      gpu_cache_);

  // To reserve sufficient memory for the reduction step we need to compute the number of blocks and threads
  // for the reduction step.
  sparse_img_align_device_utils::computeNumBlocksAndThreadsReduction(host_cache_.total_nbr_of_ftrs,
                                                                     patch_area_,
                                                                     gpu_props_,
                                                                     num_blocks_reduce_, num_threads_reduce_);


  gpu_cache_.reserveReductionCacheCapacity(static_cast<size_t>(num_blocks_reduce_));

  for(level_=options_.max_level; level_>=options_.min_level; --level_)
  {
    mu_ = 0.1;
    have_cache_ = false; // at every level, recompute the jacobians
    if(solver_options_.verbose)
      printf("\nPYRAMID LEVEL %i\n---------------\n", level_);
    optimize(state);
  }

  /// Uncomment if you need the median disparity
  //  median_disparity_ = sparse_img_align_device_utils::computeDisparity(
  //        cu_cur_imgs_pyramid_copy_,
  //        cu_T_cur_ref_bundle_,
  //        cu_camera_bundle_,
  //        host_cache_.first_ftr_index,
  //        host_cache_.nbr_of_ftrs,
  //        host_cache_.total_nbr_of_ftrs,
  //        gpu_cache_);

  // Finished, we save the pose in the frame.
  for(auto f : cur_frames->frames_)
  {
    f->T_f_w_ = f->T_cam_imu()*state.T_icur_iref*T_iref_world_;
  }

  // Reset initial values of illumination estimation TODO: make reset function
  alpha_init_ = 0.0;
  beta_init_ = 0.0;

  return host_cache_.total_nbr_of_ftrs;
}

double SparseImgAlignGpu::evaluateError(
    const SparseImgAlignState& state,
    HessianMatrix* H,
    GradientVector* g)
{

  if(!have_cache_) // is reset at every new level.
  {
    //TODO check why gpu image pyramid does not work
    //    sparse_img_align_device_utils::precomputeJacobiansAndRefPatches(
    //          cu_ref_pyramids_device_,
    //          level_,
    //          kPatchSize_,
    //          options_.estimate_illumination_gain,
    //          options_.estimate_illumination_offset,
    //          host_cache_.first_ftr_index,
    //          host_cache_.nbr_of_ftrs,
    //          gpu_cache_);

    sparse_img_align_device_utils::precomputeJacobiansAndRefPatches(
          cu_ref_imgs_pyramid_copy_,
          level_,
          patch_size_,
          options_.estimate_illumination_gain,
          options_.estimate_illumination_offset,
          host_cache_.first_ftr_index,
          host_cache_.nbr_of_ftrs,
          gpu_cache_);

    have_cache_ = true;
  }

  // Store T_cur_ref in GPU compatible format.
  cu_T_cur_ref_bundle_.resize(cu_T_cam_imu_bundle_.size());
  for(size_t i = 0; i < cu_T_cam_imu_bundle_.size();++i)
  {
    const Transformation T_cur_ref =
        cur_frames_->at(i)->T_cam_imu() * state.T_icur_iref
        * ref_frames_->at(i)->T_imu_cam();
    cu_T_cur_ref_bundle_.at(i) =
        imp::cu::Matrix<FloatTypeGpu,3,4>(
          T_cur_ref.getTransformationMatrix().block<3,4>(0,0).cast<FloatTypeGpu>());
  }

  //TODO: Bring camera control back
  sparse_img_align_device_utils::computeResidualsOfFrame(
        cu_cur_imgs_pyramid_copy_, cu_T_cur_ref_bundle_,
        cu_camera_bundle_,
        host_cache_.first_ftr_index,
        host_cache_.nbr_of_ftrs,
        level_, patch_size_,
        state.alpha, state.beta,
        gpu_cache_);

  FloatTypeGpu chi2 = sparse_img_align_device_utils::computeHessianAndGradient(
        H, g,
        host_cache_.total_nbr_of_ftrs*patch_area_,
        patch_area_,
        gpu_cache_,
        num_blocks_reduce_,
        num_threads_reduce_);

  return chi2;
}

namespace sparse_img_align_host_utils
{

void extractFeaturesSubset(const Frame& ref_frame,
                           const int max_level,
                           const int patch_size_wb, // patch_size + border (usually 2 for gradient),
                           size_t& nr_fts_extracted,
                           HostCacheHandler& host_cache)
{
  const FloatTypeGpu scale = 1.0f/(1<<max_level);
  const Vector3ft ref_pos = ref_frame.pos();
  const cv::Mat& ref_img = ref_frame.img_pyr_.at(max_level);
  const int rows_minus_one = ref_img.rows - 1;
  const int cols_minus_one = ref_img.cols - 1;
  const FloatTypeGpu patch_center_wb = (patch_size_wb - 1)/2.0f;
  nr_fts_extracted = 0;

  for(size_t i = 0; i < ref_frame.num_features_; ++i)
  {
    if(ref_frame.landmark_vec_[i] == nullptr &&
       ref_frame.seed_ref_vec_[i].keyframe == nullptr)
    {
      continue;
    }

    const FloatTypeGpu u_tl = ref_frame.px_vec_(0,i)*scale - patch_center_wb;
    const FloatTypeGpu v_tl = ref_frame.px_vec_(1,i)*scale - patch_center_wb;
    const int u_tl_i = std::floor(u_tl);
    const int v_tl_i = std::floor(v_tl);
    if(!(u_tl_i < 0 || v_tl_i < 0
         || u_tl_i + patch_size_wb >= cols_minus_one || v_tl_i + patch_size_wb >= rows_minus_one))
    {
      host_cache.push_uv(static_cast<FloatTypeGpu>(ref_frame.px_vec_(0,i)),
                         static_cast<FloatTypeGpu>(ref_frame.px_vec_(1,i)));

      // evaluate jacobian. cannot just take the 3d points coordinate because of
      // the reprojection errors in the reference image!!!
      FloatTypeGpu depth = 0;
      if(ref_frame.landmark_vec_[i])
      {
        depth = ((ref_frame.landmark_vec_[i]->pos_ - ref_pos).norm());
      }
      else if(ref_frame.seed_ref_vec_[i].keyframe)
      {
        const SeedRef& seed_ref = ref_frame.seed_ref_vec_[i];
        const Position pos = seed_ref.keyframe->T_world_cam()
            * (seed_ref.keyframe->f_vec_.col(seed_ref.seed_id)
               * 1.0/seed_ref.keyframe->invmu_sigma2_a_b_vec_(0, seed_ref.seed_id));
        depth = (pos - ref_pos).norm();
      }

      const Vector3ftGpu xyz_ref(ref_frame.f_vec_.col(i).cast<FloatTypeGpu>()*depth);
      host_cache.push_xyz(xyz_ref(0),xyz_ref(1),xyz_ref(2));
      nr_fts_extracted++;
    }
  }
  SVO_DEBUG_STREAM("Img Align: Maximum Number of Features = " << ref_frame.num_features_);
}

} // namespace sparse_img_align_host_utils
} // namespace svo
