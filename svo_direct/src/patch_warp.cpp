// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/patch_warp.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/feature_wrapper.h>
#include <svo/common/camera.h>
#include <glog/logging.h>

namespace svo {
namespace warp {

// TODO(cfo) take inverse depth!
void getWarpMatrixAffine(
    const CameraPtr& cam_ref,
    const CameraPtr& cam_cur,
    const Eigen::Ref<Keypoint>& px_ref,
    const Eigen::Ref<BearingVector>& f_ref,
    const double depth_ref,
    const Transformation& T_cur_ref,
    const int level_ref,
    AffineTransformation2* A_cur_ref)
{
  CHECK_NOTNULL(A_cur_ref);

  // Compute affine warp matrix A_ref_cur
  const int kHalfPatchSize = 5;
  const Position xyz_ref = f_ref * depth_ref;
  Position xyz_du_ref, xyz_dv_ref;
  // NOTE: project3 has no guarantee that the returned vector is unit length
  // - for pinhole: z component is 1 (unit plane)
  // - for omnicam: norm is 1 (unit sphere)
  cam_ref->backProject3(px_ref + Eigen::Vector2d(kHalfPatchSize,0) * (1<<level_ref), &xyz_du_ref);
  cam_ref->backProject3(px_ref + Eigen::Vector2d(0,kHalfPatchSize) * (1<<level_ref), &xyz_dv_ref);
  if(cam_ref->getType() == Camera::Type::kPinhole)
  {
    xyz_du_ref *= xyz_ref[2];
    xyz_dv_ref *= xyz_ref[2];
  }
  else
  {
    xyz_du_ref.normalize();
    xyz_dv_ref.normalize();
    xyz_du_ref *= depth_ref;
    xyz_dv_ref *= depth_ref;
  }

  Keypoint px_cur, px_du_cur, px_dv_cur;
  cam_cur->project3(T_cur_ref * xyz_ref, &px_cur);
  cam_cur->project3(T_cur_ref * xyz_du_ref, &px_du_cur);
  cam_cur->project3(T_cur_ref * xyz_dv_ref, &px_dv_cur);
  A_cur_ref->col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref->col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void getWarpMatrixAffineHomography(
    const CameraPtr& cam_ref,
    const CameraPtr& cam_cur,
    const Keypoint& px_ref,
    const BearingVector& f_ref,
    const BearingVector& normal_ref,
    const double depth_ref,
    const Transformation T_cur_ref,
    const int level_ref,
    AffineTransformation2& A_cur_ref)
{
  // normal = frame_ref_->T_f_w_.rotation_matrix()*point->normal_;
  // create homography matrix
  const Position t = T_cur_ref.inverse().getPosition();
  const BearingVector xyz_ref(f_ref*depth_ref);
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.getRotationMatrix()*
      (normal_ref.dot(xyz_ref)*Eigen::Matrix3d::Identity() - t*normal_ref.transpose());

  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 5;
  const BearingVector f_cur(H_cur_ref * f_ref);
  BearingVector f_du_ref, f_dv_ref;
  cam_ref->backProject3(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1<<level_ref), &f_du_ref);
  cam_ref->backProject3(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1<<level_ref), &f_dv_ref);
  const BearingVector f_du_cur = H_cur_ref * f_du_ref;
  const BearingVector f_dv_cur = H_cur_ref * f_dv_ref;
  Keypoint px_cur, px_du_cur, px_dv_cur;
  cam_cur->project3(f_cur, &px_cur);
  cam_cur->project3(f_du_cur, &px_du_cur);
  cam_cur->project3(f_dv_cur, &px_dv_cur);
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

int getBestSearchLevel(
    const AffineTransformation2& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

bool warpAffine(
    const AffineTransformation2& A_cur_ref,
    const cv::Mat& img_ref,
    const Eigen::Ref<Keypoint>& px_ref,
    const int level_ref,
    const int search_level,
    const int halfpatch_size,
    uint8_t* patch)
{
  Eigen::Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>()*(1<<search_level);
  if(std::isnan(A_ref_cur(0,0)))
  {
    LOG(WARNING) << "Affine warp is NaN, probably camera has no translation";
    return false;
  }

  // Perform the warp on a larger patch.
  uint8_t* patch_ptr = patch;
  const Eigen::Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  const int stride = img_ref.step.p[0];
  for (int y=-halfpatch_size; y<halfpatch_size; ++y)
  {
    for (int x=-halfpatch_size; x<halfpatch_size; ++x, ++patch_ptr)
    {
      const Eigen::Vector2f px_patch(x, y);
      const Eigen::Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
      const int xi = std::floor(px[0]);
      const int yi = std::floor(px[1]);
      if (xi<0 || yi<0 || xi+1>=img_ref.cols || yi+1>=img_ref.rows)
        return false;
      else
      {
        const float subpix_x = px[0]-xi;
        const float subpix_y = px[1]-yi;
        const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
        const float w01 = (1.0f-subpix_x)*subpix_y;
        const float w10 = subpix_x*(1.0f-subpix_y);
        const float w11 = 1.0f - w00 - w01 - w10;
        const uint8_t* const ptr = img_ref.data + yi*stride + xi;
        *patch_ptr = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride+1]);
      }
    }
  }
  return true;
}

bool warpPixelwise(
    const Frame& cur_frame,
    const Frame& ref_frame,
    const FeatureWrapper& ref_ftr,
    const int level_ref,
    const int level_cur,
    const int halfpatch_size,
    uint8_t* patch)
{
  double depth_ref = (ref_frame.pos() - ref_ftr.landmark->pos()).norm();
  double depth_cur = (cur_frame.pos() - ref_ftr.landmark->pos()).norm();

  // backproject to 3D points in reference frame
  Eigen::Vector3d xyz_ref;
  ref_frame.cam()->backProject3(ref_ftr.px, &xyz_ref);
  xyz_ref = xyz_ref.normalized() * depth_ref;

  // project to current frame and convert to search level
  Eigen::Vector3d xyz_cur = cur_frame.T_cam_world()*(ref_frame.T_cam_world().inverse())*xyz_ref;
  Eigen::Vector2d px_cur;
  cur_frame.cam()->project3(xyz_cur, &px_cur);
  Eigen::Vector2d px_cur_search = px_cur / (1<<level_cur);

  // for each pixel in the patch(on search level):
  // - convert to image level
  // - backproject to 3D points
  // - project to ref frame and find pixel value in ref level
  uint8_t* patch_ptr = patch;
  const cv::Mat& img_ref = ref_frame.img_pyr_[level_ref];
  const int stride = img_ref.step.p[0];

  for(int y=-halfpatch_size; y<halfpatch_size; ++y)
  {
    for(int x=-halfpatch_size; x<halfpatch_size; ++x, ++patch_ptr)
    {
      const Eigen::Vector2d ele_patch(x, y);
      Eigen::Vector2d ele_search = ele_patch + px_cur_search;
      Eigen::Vector3d ele_xyz_cur;
      cur_frame.cam()->backProject3(ele_search*(1<<level_cur), &ele_xyz_cur);
      ele_xyz_cur = ele_xyz_cur.normalized() * depth_cur;
      Eigen::Vector3d ele_xyz_ref = ref_frame.T_cam_world() * (cur_frame.T_cam_world().inverse()) * ele_xyz_cur;
      Eigen::Vector2d ele_ref;
      ref_frame.cam()->project3(ele_xyz_ref, &ele_ref);
      ele_ref = ele_ref / (1<<level_ref);

      const int xi = std::floor(ele_ref[0]);
      const int yi = std::floor(ele_ref[1]);
      if (xi<0 || yi<0 || xi+1>=img_ref.cols || yi+1>=img_ref.rows)
      {
        VLOG(200) << "ref image: col-" << img_ref.cols
                  << ", row-" << img_ref.rows;
        VLOG(200) << "xi: " << xi << ", " << "yi: " << yi;
        return false;
      }
      else
      {
        const float subpix_x = ele_ref[0]-xi;
        const float subpix_y = ele_ref[1]-yi;
        const float w00 = (1.0f-subpix_x)*(1.0f-subpix_y);
        const float w01 = (1.0f-subpix_x)*subpix_y;
        const float w10 = subpix_x*(1.0f-subpix_y);
        const float w11 = 1.0f - w00 - w01 - w10;
        const uint8_t* const ptr = img_ref.data + yi*stride + xi;
        *patch_ptr = static_cast<uint8_t>(w00*ptr[0] + w01*ptr[stride] + w10*ptr[1] + w11*ptr[stride+1]);
      }
    }
  }

  return true;
}

void createPatchNoWarp(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    const int halfpatch_size,
    uint8_t* patch)
{
  CHECK_NOTNULL(patch);
  CHECK(px(0) >= halfpatch_size
        && px(1) >= halfpatch_size
        && px(0) < img.cols - halfpatch_size
        && px(1) < img.rows - halfpatch_size);

  const int patch_size = 2 * halfpatch_size;
  uint8_t* patch_ptr = patch;
  const int step = img.step.p[0];
  for (int y = 0; y < patch_size; ++y)
  {
    uint8_t* img_ptr =
        (uint8_t*) img.data
        + (px[1] - halfpatch_size + y) * step
        + (px[0] - halfpatch_size);
    for (int x = 0; x < patch_size; ++x, ++patch_ptr, ++img_ptr)
    {
      *patch_ptr = *img_ptr;
    }
  }
}

void createPatchNoWarpInterpolated(
    const cv::Mat& img,
    const Eigen::Ref<Keypoint>& px,
    const int halfpatch_size,
    uint8_t* patch)
{
  CHECK_NOTNULL(patch);

  // TODO(cfo): This could be easily implemented using SIMD instructions.

  const int step = img.step.p[0];
  const float u = px(0);
  const float v = px(1);
  const int u_r = std::floor(u);
  const int v_r = std::floor(v);
  CHECK(u_r >= halfpatch_size
        && v_r >= halfpatch_size
        && u_r < img.cols - halfpatch_size
        && v_r < img.rows - halfpatch_size);

  // compute interpolation weights
  const float subpix_x = u - u_r;
  const float subpix_y = v - v_r;
  const float wTL = (1.0-subpix_x) * (1.0-subpix_y);
  const float wTR = subpix_x * (1.0-subpix_y);
  const float wBL = (1.0-subpix_x)*subpix_y;
  const float wBR = subpix_x * subpix_y;

  const int patch_size = 2 * halfpatch_size;
  uint8_t* patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    uint8_t* img_ptr =
        (uint8_t*) img.data
        + (v_r - halfpatch_size + y) * step
        + (u_r - halfpatch_size);
    for (int x = 0; x < patch_size; ++x, ++patch_ptr, ++img_ptr)
    {
      *patch_ptr =
          wTL*img_ptr[0] + wTR*img_ptr[1] + wBL*img_ptr[step] + wBR*img_ptr[step+1];;
    }
  }
}

} // namespace warp
} // namespace svo
