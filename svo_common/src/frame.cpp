// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/common/frame.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <fast/fast.h>
#include <vikit/math_utils.h>
#include <vikit/vision.h>

#include <svo/common/logging.h>
#include <svo/common/point.h>
#include <svo/common/camera.h>

namespace svo {

int Frame::frame_counter_ = 0;

Frame::Frame(
    const CameraPtr& cam,
    const cv::Mat& img,
    const int64_t timestamp_ns,
    size_t n_pyr_levels)
  : id_(frame_counter_++) // TEMPORARY
  , cam_(cam)
  , key_pts_(5, std::make_pair(-1, BearingVector::Zero()))
  , timestamp_(timestamp_ns)
{
  initFrame(img, n_pyr_levels);
}

Frame::Frame(
    const int id,
    const int64_t timestamp_ns,
    const CameraPtr& cam,
    const Transformation& T_world_cam)
  : id_(id)
  , cam_(cam)
  , T_f_w_(T_world_cam.inverse())
  , key_pts_(5, std::make_pair(-1, BearingVector::Zero()))
  , timestamp_(timestamp_ns)
{}

Frame::~Frame()
{}

void Frame::initFrame(const cv::Mat& img, size_t n_pyr_levels)
{
  CHECK_EQ(key_pts_[0].first, -1);
  CHECK_EQ(key_pts_[4].first, -1);

  // check image
  CHECK(!img.empty());
  CHECK_EQ(img.cols, static_cast<int>(cam_->imageWidth()));
  CHECK_EQ(img.rows, static_cast<int>(cam_->imageHeight()));

  if (img.type() == CV_8UC1)
  {
    frame_utils::createImgPyramid(img, n_pyr_levels, img_pyr_);
  }
  else if (img.type() == CV_8UC3)
  {
    cv::Mat gray_image;
    cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);
    frame_utils::createImgPyramid(gray_image, n_pyr_levels, img_pyr_);
    original_color_image_ = img;
  }
  else
  {
    LOG(FATAL) << "Unknown image type " << img.type() << "!";
  }
  accumulated_w_T_correction_.setIdentity();
}

void Frame::setKeyframe()
{
  is_keyframe_ = true;
  setKeyPoints();
}

void Frame::deleteLandmark(const size_t& feature_index)
{
  landmark_vec_.at(feature_index) = nullptr;
}

void Frame::resizeFeatureStorage(size_t num)
{
  if(static_cast<size_t>(px_vec_.cols()) < num)
  {
    const size_t n_new = num - num_features_;

    px_vec_.conservativeResize(Eigen::NoChange, num);
    f_vec_.conservativeResize(Eigen::NoChange, num);
    score_vec_.conservativeResize(num);
    level_vec_.conservativeResize(num);
    grad_vec_.conservativeResize(Eigen::NoChange, num);
    invmu_sigma2_a_b_vec_.conservativeResize(Eigen::NoChange, num);
    track_id_vec_.conservativeResize(num);

    type_vec_.resize(num, FeatureType::kCorner);
    landmark_vec_.resize(num, nullptr);
    seed_ref_vec_.resize(num);
    in_ba_graph_vec_.resize(num, false);

    // initial values
    level_vec_.tail(n_new).setZero();
    track_id_vec_.tail(n_new).setConstant(-1);
    score_vec_.tail(n_new).setConstant(-1);
  }
  else if(num < static_cast<size_t>(px_vec_.cols()))
  {
    SVO_ERROR_STREAM("Downsizing storage not implemented. cols = " << px_vec_.cols()
                     << " , desired = " << num << ", num features = " << num_features_);
  }
}

void Frame::clearFeatureStorage()
{
  px_vec_.resize(Eigen::NoChange, 0);
  f_vec_.resize(Eigen::NoChange, 0);
  score_vec_.resize(0);
  level_vec_.resize(0);
  grad_vec_.resize(Eigen::NoChange, 0);
  invmu_sigma2_a_b_vec_.resize(Eigen::NoChange, 0);
  track_id_vec_.resize(0);
  type_vec_.clear();
  landmark_vec_.clear();
  seed_ref_vec_.clear();
  in_ba_graph_vec_.clear();
  num_features_ = 0;
}

void Frame::copyFeaturesFrom(const Frame& other)
{
  px_vec_ = other.px_vec_;
  f_vec_ = other.f_vec_;
  score_vec_ = other.score_vec_;
  level_vec_ = other.level_vec_;
  grad_vec_ = other.grad_vec_;
  type_vec_ = other.type_vec_;
  landmark_vec_ = other.landmark_vec_;
  seed_ref_vec_ = other.seed_ref_vec_;
  invmu_sigma2_a_b_vec_ = other.invmu_sigma2_a_b_vec_;
  track_id_vec_ = other.track_id_vec_;
  num_features_ = other.num_features_;
  in_ba_graph_vec_ = other.in_ba_graph_vec_;
}

FeatureWrapper Frame::getFeatureWrapper(size_t index)
{
  CHECK_LT(index, static_cast<size_t>(px_vec_.cols()));
  return FeatureWrapper(
        type_vec_[index], px_vec_.col(index), f_vec_.col(index),
        grad_vec_.col(index), score_vec_(index), level_vec_(index), landmark_vec_[index],
        seed_ref_vec_[index], track_id_vec_(index));
}

FeatureWrapper Frame::getEmptyFeatureWrapper()
{
  return getFeatureWrapper(num_features_);
}

void Frame::setKeyPoints()
{
  const FloatType cu = cam_->imageWidth()/2;
  const FloatType cv = cam_->imageHeight()/2;

  for(size_t i = 0; i < num_features_; ++i)
  {
    if(landmark_vec_[i] == nullptr || type_vec_[i] == FeatureType::kOutlier)
      continue;

    const FloatType& u = px_vec_(0,i);
    const FloatType& v = px_vec_(1,i);

    // center
    if(key_pts_[0].first == -1)
      key_pts_[0] = std::make_pair(i, landmark_vec_[i]->pos_);

    else if(std::max(std::fabs(u-cu), std::fabs(v-cv))
            < std::max(std::fabs(px_vec_(0, key_pts_[0].first) - cu),
                       std::fabs(px_vec_(1, key_pts_[0].first) - cv)))
      key_pts_[0] = std::make_pair(i, landmark_vec_[i]->pos_);

    // corner
    if(u >= cu && v >= cv)
    {
      if(key_pts_[1].first == -1)
        key_pts_[1] = std::make_pair(i, landmark_vec_[i]->pos_);
      else if((u-cu) * (v-cv)
              > (px_vec_(0, key_pts_[1].first) - cu) * (px_vec_(1, key_pts_[1].first)-cv))
        key_pts_[1] = std::make_pair(i, landmark_vec_[i]->pos_);
    }
    if(u >= cu && v < cv)
    {
      if(key_pts_[2].first == -1)
        key_pts_[2] = std::make_pair(i, landmark_vec_[i]->pos_);
      else if((u-cu) * (v-cv)
              > (px_vec_(0, key_pts_[2].first) - cu) * (px_vec_(1, key_pts_[2].first)-cv))
        key_pts_[2] = std::make_pair(i, landmark_vec_[i]->pos_);
    }
    if(u < cv && v < cv)
    {
      if(key_pts_[3].first == -1)
        key_pts_[3] = std::make_pair(i, landmark_vec_[i]->pos_);
      else if((u-cu) * (v-cv)
              > (px_vec_(0, key_pts_[3].first) - cu) * (px_vec_(1, key_pts_[3].first)-cv))
        key_pts_[3] = std::make_pair(i, landmark_vec_[i]->pos_);
    }
    if(u < cv && v >= cv)
    {
      if(key_pts_[4].first == -1)
        key_pts_[4] = std::make_pair(i, landmark_vec_[i]->pos_);
      else if((u-cu) * (v-cv)
              > (px_vec_(0, key_pts_[4].first) - cu) * (px_vec_(1, key_pts_[4].first)-cv))
        key_pts_[4] = std::make_pair(i, landmark_vec_[i]->pos_);
    }
  }
}

bool Frame::isVisible(const Eigen::Vector3d& xyz_w,
                      Eigen::Vector2d* px) const
{
  Eigen::Vector3d xyz_f = T_f_w_*xyz_w;
  if (cam_->getType() == Camera::Type::kPinhole)
  {
    Eigen::Vector2d px_top_left(0.0, 0.0);
    Eigen::Vector3d f_top_left;
    cam_->backProject3(px_top_left, &f_top_left);
    f_top_left.normalize();
    const Eigen::Vector3d z(0.0, 0.0, 1.0);
    const double min_cos = f_top_left.dot(z);
    const double cur_cos = xyz_f.normalized().dot(z);
    if (cur_cos < min_cos)
    {
      return false;
    }
  }

  if (px)
  {
    return cam_->project3(xyz_f, px).isKeypointVisible();
  }
  else
  {
    Eigen::Vector2d px_temp;
    return cam_->project3(xyz_f, &px_temp).isKeypointVisible();
  }
}

const cv::Mat& Frame::getMask() const
{
  return cam_->getMask();
}

double Frame::getErrorMultiplier() const
{
  return cam_->errorMultiplier();
}

double Frame::getAngleError(double img_err) const
{
  return cam_->getAngleError(img_err);
}

void Frame::jacobian_xyz2image_imu(
    const svo::Camera& cam,
    const Transformation& T_cam_imu,
    const Eigen::Vector3d& p_in_imu,
    Eigen::Matrix<double,2,6>& J)
{
  Eigen::Matrix<double,3,6> G_x; // Generators times pose
  G_x.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
  G_x.block<3,3>(0,3) = -vk::skew(p_in_imu);
  const Eigen::Vector3d p_in_cam = T_cam_imu * p_in_imu;

  Eigen::Matrix<double,2,3> J_proj; // projection derivative
  Eigen::Vector2d out_point;
  cam.project3(p_in_cam, &out_point, &J_proj);

  J = J_proj * T_cam_imu.getRotation().getRotationMatrix() * G_x;
}


FrameBundle::FrameBundle(const std::vector<FramePtr>& frames)
  : frames_(frames)
{
  static BundleId bundle_counter = 0;
  bundle_id_ = bundle_counter++;
  for(const FramePtr& frame : frames)
  {
    frame->bundle_id_ = bundle_id_;
  }
}

FrameBundle::FrameBundle(const std::vector<FramePtr>& frames, const int bundle_id)
  : frames_(frames)
{
  for(const FramePtr& frame : frames)
  {
    frame->bundle_id_ = bundle_id;
  }
}

size_t FrameBundle::numFeatures() const
{
  size_t n = 0;
  for(const FramePtr& f : frames_)
    n += f->numFeatures();
  return n;
}

size_t FrameBundle::numTrackedFeatures() const
{
  size_t n = 0;
  for(const FramePtr& f : frames_)
    n += f->numTrackedFeatures();
  return n;
}

size_t FrameBundle::numTrackedLandmarks() const
{
  size_t n = 0;
  for(const FramePtr& f : frames_)
    n += f->numTrackedLandmarks();
  return n;
}

size_t FrameBundle::numLandmarks() const
{
  size_t n = 0;
  for(const FramePtr& f : frames_)
    n += f->numLandmarks();
  return n;
}

size_t FrameBundle::numLandmarksInBA() const
{
  size_t n = 0;
  for(const FramePtr& f : frames_)
    n += f->numLandmarksInBA();
  return n;
}

size_t FrameBundle::numTrackedLandmarksInBA() const
{
  size_t n = 0;
  for(const FramePtr& f : frames_)
    n += f->numTrackedLandmarksInBA();
  return n;
}

size_t FrameBundle::numFixedLandmarks() const
{
  size_t n = 0;
  for (const FramePtr& f: frames_)
    n += f->numFixedLandmarks();
  return n;
}

/// Utility functions for the Frame class
namespace frame_utils {

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  CHECK_EQ(img_level_0.type(), CV_8U);
  CHECK_GT(img_level_0.rows, 0);
  CHECK_GT(img_level_0.cols, 0);
  CHECK_GT(n_levels, 0);

  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    vk::halfSample(pyr[i-1], pyr[i]);
  }
}

bool getSceneDepth(const FramePtr& frame, double& depth_median, double& depth_min, double& depth_max)
{
  std::vector<double> depth_vec;
  depth_vec.reserve(frame->num_features_);
  depth_min = std::numeric_limits<double>::max();
  depth_max = 0;
  double depth = 0;
  const Position ref_pos = frame->pos();
  for(size_t i = 0; i < frame->num_features_; ++i)
  {
    if(frame->landmark_vec_[i])
    {
      depth = (frame->T_cam_world()*frame->landmark_vec_[i]->pos_).norm();
    }
    else if(frame->seed_ref_vec_[i].keyframe)
    {
      const SeedRef& seed_ref = frame->seed_ref_vec_[i];
      const Position pos = seed_ref.keyframe->T_world_cam() *
          seed_ref.keyframe->getSeedPosInFrame(seed_ref.seed_id);
      depth = (pos - ref_pos).norm();
    }
    else
    {
      continue;
    }

    depth_vec.push_back(depth);
    depth_min = std::min(depth, depth_min);
    depth_max = std::max(depth, depth_max);
  }
  if(depth_vec.empty())
  {
    SVO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
    return false;
  }
  depth_median = vk::getMedian(depth_vec);
  return true;
}

void computeNormalizedBearingVectors(
    const Keypoints& px_vec,
    const Camera& cam,
    Bearings* f_vec)
{
  CHECK_NOTNULL(f_vec);
  std::vector<bool> success;
  cam.backProject3(px_vec, f_vec, &success);
  for (const bool s : success) {
    CHECK(s);
  }
  *f_vec = f_vec->array().rowwise() / f_vec->colwise().norm().array();
}

} // namespace frame_utils
} // namespace svo
