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
#include <svo/common/types.h>
#include <svo/common/transformation.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/feature_wrapper.h>
#include <svo/common/point.h>
#include <svo/common/seed.h>
#include <svo/common/conversions.h>

namespace svo {

// custom (frame related) types
using ImgPyr = std::vector<cv::Mat>;

//------------------------------------------------------------------------------
/// A frame saves the image, the associated features and the estimated pose.
class Frame
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using mutex_t = std::mutex;
  using ulock_t = std::unique_lock<std::mutex>;
  using Landmarks = std::vector<PointPtr>;
  using SeedRefs  = std::vector<SeedRef>;

  static int                    frame_counter_;         //!< Counts the number of created frames. Used to set the unique id.
  int                           id_;                    //!< Unique id of the frame.
  BundleId                      bundle_id_;
  int                           nframe_index_ = -1;     //!< Storage index in NFrame.
  CameraPtr                     cam_;                   //!< Camera model.
  Transformation                T_f_w_;                 //!< Transform (f)rame from (w)orld.
  ImgPyr                        img_pyr_;               //!< Image Pyramid.
  cv::Mat                       original_color_image_;
  // safe without the Eigen aligned allocator, since Position is 24 bytes
  std::vector<std::pair<int, Position>, Eigen::aligned_allocator<
  std::pair<int, Position> > > key_pts_;                //!< Five features and associated 3D points which are used to detect if two frames have overlapping field of view. store index + depth
  bool                          is_keyframe_ = false;   //!< Was this frames selected as keyframe?
  int                           last_published_ts_;     //!< Timestamp of last publishing.
  Quaternion                    R_imu_world_;           //!< IMU Attitude provided by *external* attitude extimator

  int64_t        timestamp_ = -1;    ///!< Timestamp of when the image was recorded in nanoseconds.
  Transformation T_body_cam_;
  Transformation T_cam_body_;

  // Features
  // All vectors must have the same length/cols at all time!
  // {
  size_t num_features_ = 0u;    ///< the vectors below are initialized to the maximum number of feature-tracks
  Keypoints px_vec_;            ///< Pixel Coordinates
  Bearings f_vec_;              ///< Bearing Vector
  Scores score_vec_;            ///< Keypoint detection scores
  Levels  level_vec_;           ///< Level of the feature
  Gradients grad_vec_;          ///< Gradient direction of edgelet normal
  FeatureTypes type_vec_;       ///< Is the feature a corner or an edgelet?
  Landmarks landmark_vec_;      ///< Reference to 3D point. Can contain nullpointers!
  TrackIds track_id_vec_;       ///< ID of every observed 3d point. -1 if no point assigned.
  SeedRefs seed_ref_vec_;       ///< Only for seeds during reprojection
  SeedStates invmu_sigma2_a_b_vec_; ///< Vector containing all necessary information for seed update.
  std::vector<bool> in_ba_graph_vec_;
  // }

  FloatType seed_mu_range_;
  bool is_stable_ = true;

  // imu states
  Eigen::Vector3d imu_vel_w_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_gyr_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_acc_bias_ = Eigen::Vector3d::Zero();

  // loop correction
  Transformation accumulated_w_T_correction_;

  /// Default Constructor
  Frame(
      const CameraPtr& cam,
      const cv::Mat& img,
      const int64_t timestamp_ns,
      const size_t n_pyr_levels);

  /// Constructor without image. Just for testing!
  Frame(
      const int id,
      const int64_t timestamp_ns,
      const CameraPtr& cam,
      const Transformation& T_world_cam);

  /// Empty constructor. Just for testing!
  Frame() {}

  /// Destructor
  virtual ~Frame();

  // no copy
  Frame(const Frame&) = delete;
  Frame& operator=(const Frame&) = delete;

  /// Initialize new frame and create image pyramid.
  void initFrame(const cv::Mat& img, size_t n_pyr_levels);

  /// Select this frame as keyframe.
  void setKeyframe();

  /// Delete feature.
  void deleteLandmark(const size_t& feature_index);

  /// Resize the number of features. (NOT THREADSAFE)
  void resizeFeatureStorage(size_t num);

  void clearFeatureStorage();

  /// Copy feature data from another frame.
  void copyFeaturesFrom(const Frame& other);

  FeatureWrapper getFeatureWrapper(size_t idx);

  FeatureWrapper getEmptyFeatureWrapper();

  /// Get depth at seed.
  inline FloatType getSeedDepth(size_t idx) const {
    return seed::getDepth(invmu_sigma2_a_b_vec_.col(idx));
  }

  /// Get coordinates of seed in frame coordinates.
  inline Position getSeedPosInFrame(size_t idx) const {
    return f_vec_.col(idx) * getSeedDepth(idx);
  }

  /// Number of features. Not necessarily succesfully tracked ones.
  inline size_t numFeatures() const {
    return num_features_;
  }

  /// Check if i-th keypoint has a reference to a landmark.
  inline bool isValidLandmark(size_t i) const {
    return (landmark_vec_.at(i) != nullptr);
  }

  /// Number of successfully tracked features: seeds and actual landmarks.
  inline size_t numTrackedFeatures() const {
    size_t count = 0;
    for(size_t i = 0; i < num_features_; ++i)
    {
      if((isValidLandmark(i) && !isFixedLandmark(type_vec_[i]) &&
          !isMapPoint(type_vec_[i]))
         || isCornerEdgeletSeed(type_vec_[i]))
        ++count;
    }
    return count;
  }

  inline size_t numTrackedLandmarks() const {
    size_t count = 0;
    for(size_t i = 0; i < num_features_; ++i)
    {
      if(isValidLandmark(i) && !isFixedLandmark(type_vec_[i]) &&
         !isMapPoint(type_vec_[i]))
        ++count;
    }
    return count;
  }

  /// Number of landmark references in frames.
  inline size_t numLandmarks() const {
    return static_cast<size_t>(
          std::count_if(landmark_vec_.begin(), landmark_vec_.end(),
                        [](const PointPtr& p) { return p != nullptr; }));
  }

  inline size_t numFixedLandmarks() const {
    size_t count = 0;
    for(size_t i = 0; i < num_features_; ++i)
    {
      if(isValidLandmark(i) && isFixedLandmark(type_vec_[i]))
        ++count;
    }
    return count;
  }

  /// Number of landmark references in frames.
  inline size_t numLandmarksInBA() const {
    return static_cast<size_t>(
          std::count_if(landmark_vec_.begin(), landmark_vec_.end(),
                        [](const PointPtr& p)
    { return p != nullptr && p->in_ba_graph_; }));
  }

  inline size_t numTrackedLandmarksInBA() const {
    size_t count = 0;
    for(size_t i = 0; i < num_features_; ++i)
    {
      if (isValidLandmark(i) && !isFixedLandmark(type_vec_[i]) &&
          landmark_vec_[i]->in_ba_graph_)
      {
        count++;
      }
    }
    return count;
  }

  /// The KeyPoints are those five features which are closest to the 4 image corners
  /// and to the center and which have a 3D point assigned. These points are used
  /// to quickly check whether two frames have overlapping field of view.
  void setKeyPoints();
  inline void resetKeyPoints()
  {
    key_pts_.resize(5, std::make_pair(-1, BearingVector()));
  }

  /// Check if we can select five better key-points.
  void checkKeyPoints(const FeaturePtr& ftr);

  /// Check if a point in (w)orld coordinate frame is visible in the image.
  bool isVisible(const Eigen::Vector3d& xyz_w,
                 Eigen::Vector2d* px=nullptr) const;

  /// Masks is same size as the image. Convention: 0 == masked, nonzero == valid.
  const cv::Mat& getMask() const;

  /// Full resolution image stored in the frame.
  inline const cv::Mat& img() const { return img_pyr_[0]; }

  /// Id of frame.
  inline int id() const { return id_; }

  /// Id of frame bundle
  inline BundleId bundleId() const { return bundle_id_; }

  /// Get storage index of frame in parent NFrame.
  inline int getNFrameIndex() const {
    CHECK_GE(nframe_index_, 0) << "NFrame Index not set in frame.";
    return nframe_index_;
  }

  /// Set storage index of frame in parent NFrame.
  inline void setNFrameIndex(size_t nframe_index) { nframe_index_ = nframe_index; }

  /// Get camera pose in imu frame.
  inline const Transformation& T_imu_cam() const { return T_body_cam_; }

  /// Get imu pose in camera frame.
  inline const Transformation& T_cam_imu() const { return T_cam_body_; }

  /// Get pose of world origin in frame coordinates.
  inline const Transformation& T_cam_world() const { return T_f_w_; }

  /// Get pose of the cam in world coordinates.
  inline Transformation T_world_cam() const { return T_f_w_.inverse(); }

  /// Get pose of imu in world coordinates.
  inline Transformation T_world_imu() const { return (T_imu_cam()*T_f_w_).inverse(); }

  /// Get pose of world-origin in IMU coordinates.
  inline Transformation T_imu_world() const { return T_imu_cam()*T_f_w_; }

  /// Set camera to imu transformation.
  inline void set_T_cam_imu(const Transformation& T_cam_imu)
  {
    T_cam_body_ = T_cam_imu;
    T_body_cam_ = T_cam_imu.inverse();
  }

  /// set new pose
  /// If setting T_f_w_ is desired, one should access T_f_w_ directly.
  inline void set_T_w_imu(const Transformation& T_w_imu)
  {
    T_f_w_ = (T_w_imu * T_body_cam_).inverse();
  }

  /// set IMU state
  inline void setIMUState(const Eigen::Vector3d& imu_vel_w,
                          const Eigen::Vector3d& gyr_bias,
                          const Eigen::Vector3d& acc_bias)
  {
    imu_vel_w_ = imu_vel_w;
    imu_gyr_bias_ = gyr_bias;
    imu_acc_bias_ = acc_bias;
  }

  /// Camera model.
  inline const CameraPtr& cam() const { return cam_; }

  /// Timestamp of frame in nanoseconds.
  inline int64_t getTimestampNSec() const { return timestamp_; }

  /// Timestamp of frame in seconds.
  inline double getTimestampSec() const { return static_cast<double>(timestamp_)/1e9; }

  /// Was this frame selected as keyframe?
  inline bool isKeyframe() const { return is_keyframe_; }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Eigen::Vector3d pos() const { return T_world_cam().getPosition(); }

  /// Get the pose of the imu frame in (w)orld coordinate frame.
  inline Eigen::Vector3d imuPos() const { return T_world_imu().getPosition(); }

  double getErrorMultiplier() const;

  double getAngleError(double img_err) const;

  /// Frame jacobian for projection of 3D point in (f)rame coordinate to
  /// unit plane coordinates uv (focal length = 1).
  inline static void jacobian_xyz2uv(
      const Eigen::Vector3d& xyz_in_f,
      Eigen::Matrix<double,2,6>& J)
  {
    const double x = xyz_in_f[0];
    const double y = xyz_in_f[1];
    const double z_inv = 1./xyz_in_f[2];
    const double z_inv_2 = z_inv*z_inv;

    J(0,0) = -z_inv;              // -1/z
    J(0,1) = 0.0;                 // 0
    J(0,2) = x*z_inv_2;           // x/z^2
    J(0,3) = y*J(0,2);            // x*y/z^2
    J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
    J(0,5) = y*z_inv;             // y/z

    J(1,0) = 0.0;                 // 0
    J(1,1) = -z_inv;              // -1/z
    J(1,2) = y*z_inv_2;           // y/z^2
    J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
    J(1,4) = -J(0,3);             // -x*y/z^2
    J(1,5) = -x*z_inv;            // -x/z
  }

  /// Jacobian of reprojection error (on unit plane) w.r.t. IMU pose.
  inline static void jacobian_xyz2uv_imu(
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu,
      Eigen::Matrix<double,2,6>& J)
  {
    Eigen::Matrix<double,3,6> G_x; // Generators times pose
    G_x.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    G_x.block<3,3>(0,3) = -vk::skew(p_in_imu);
    const Eigen::Vector3d p_in_cam = T_cam_imu * p_in_imu;

    Eigen::Matrix<double,2,3> J_proj; // projection derivative
    J_proj << 1, 0, -p_in_cam[0]/p_in_cam[2],
        0, 1, -p_in_cam[1]/p_in_cam[2];

    J = - 1.0/p_in_cam[2] * J_proj * T_cam_imu.getRotation().getRotationMatrix() * G_x;
  }

  /// Jacobian of reprojection error (on image plane) w.r.t. IMU pose.
  inline static void jacobian_xyz2img_imu(
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu,
      const Eigen::Matrix<double, 2, 3>& J_cam,
      Eigen::Matrix<double,2,6>& J)
  {
    Eigen::Matrix<double,3,6> G_x; // Generators times pose
    G_x.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    G_x.block<3,3>(0,3) = -vk::skew(p_in_imu);

    J =  J_cam * T_cam_imu.getRotation().getRotationMatrix() * G_x;
  }

  /// Jacobian of using unit bearing vector for map point w.r.t IMU pose.
  inline static void jacobian_xyz2f_imu(
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu,
      Eigen::Matrix<double, 3, 6>& J)
  {
    Eigen::Matrix<double,3,6> G_x;
    G_x.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    G_x.block<3,3>(0,3) = -vk::skew(p_in_imu);
    const Eigen::Vector3d p_in_cam = T_cam_imu * p_in_imu;

    Eigen::Matrix<double, 3, 3> J_normalize;
    double x2 = p_in_cam[0]*p_in_cam[0];
    double y2 = p_in_cam[1]*p_in_cam[1];
    double z2 = p_in_cam[2]*p_in_cam[2];
    double xy = p_in_cam[0]*p_in_cam[1];
    double yz = p_in_cam[1]*p_in_cam[2];
    double zx = p_in_cam[2]*p_in_cam[0];
    J_normalize << y2+z2, -xy, -zx,
        -xy, x2+z2, -yz,
        -zx, -yz, x2+y2;
    J_normalize *= 1 / std::pow(x2+y2+z2, 1.5);

    J = J_normalize * T_cam_imu.getRotationMatrix() * G_x;
  }

  /// Jacobian to image plane(also taking distortion into consideration) w.r.t IMU pose.
  static void jacobian_xyz2image_imu(
      const Camera& cam,
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu,
      Eigen::Matrix<double,2,6>& J);


  template<typename Archive>
  void serialize(Archive& ar, const unsigned /*version*/)
  {
    ar & frame_counter_;
    ar & id_;
    ar & timestamp_;
    //ar & cam_;
    ar & T_f_w_;
    //ar & Cov_;
    ar & img_pyr_;
    //ar & fts_;
    //ar & key_pts_;
    ar & is_keyframe_;
    //ar & v_kf_;
    ar & last_published_ts_;
    // ar & R_imu_world_;
  }
};

class FrameBundle
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<FrameBundle> Ptr;
  typedef std::vector<FramePtr> FrameList;

  FrameBundle(const std::vector<FramePtr>& frames);
  FrameBundle(const std::vector<FramePtr>& frames, const int bundle_id);
  ~FrameBundle() = default;

  FrameBundle(const FrameBundle&) = delete;
  FrameBundle& operator=(const FrameBundle&) = delete;

  inline const FramePtr& at(size_t i) const {
    return frames_.at(i);
  }

  inline size_t size() const {
    return frames_.size();
  }

  inline bool empty() const {
    return frames_.empty();
  }

  inline BundleId getBundleId() const {
    return bundle_id_;
  }

  /// Get timestamp of camera rig.
  inline int64_t getMinTimestampNanoseconds() const {
    CHECK(!frames_.empty());
    return frames_[0]->getTimestampNSec();
  }

  /// Get timestamp of camera rig.
  inline double getMinTimestampSeconds() const {
    const double cur_t_sec =
        common::conversions::kNanoSecondsToSeconds *
        static_cast<double>(getMinTimestampNanoseconds());
    return cur_t_sec;
  }

  /// Get pose of camera rig.
  Transformation get_T_W_B() const {
    CHECK(!frames_.empty());
    return frames_[0]->T_world_imu();
  }

  /// Set pose of camera rig.
  void set_T_W_B(const Transformation& T_W_B) {
    for(const FramePtr& frame : frames_)
      frame->T_f_w_ = (T_W_B*frame->T_body_cam_).inverse();
  }

  inline void setIMUState(const Eigen::Vector3d& imu_vel_w,
                          const Eigen::Vector3d& gyr_bias,
                          const Eigen::Vector3d& acc_bias)
  {
    for(const FramePtr& f : frames_)
    {
      f->setIMUState(imu_vel_w, gyr_bias, acc_bias);
    }
  }

  void setKeyframe() {
    is_keyframe_ = true;
  }

  bool isKeyframe() const {
    return is_keyframe_;
  }

  /// Number of landmark references in frames.
  size_t numLandmarks() const;

  /// Number of landmark references in frames.
  size_t numLandmarksInBA() const;

  size_t numTrackedLandmarksInBA() const;

  size_t numFixedLandmarks() const;

  /// Number of features. Not necessarily succesfully tracked ones.
  size_t numFeatures() const;

  /// Number of tracked features: both landmarks and matched seeds.
  size_t numTrackedFeatures() const;

  /// Number of tracked features: both landmarks and matched seeds.
  size_t numTrackedLandmarks() const;

  FrameList frames_;

  /// IMU measurements since the last nframe (including the last IMU measurement
  /// of the previous edge for integration).
  Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps_ns_;
  Eigen::Matrix<double, 6, Eigen::Dynamic> imu_measurements_; // Order: [acc, gyro]

  // Make class iterable:
  typedef FrameList::value_type value_type;
  typedef FrameList::iterator iterator;
  typedef FrameList::const_iterator const_iterator;
  FrameList::iterator begin() { return frames_.begin(); }
  FrameList::iterator end() { return frames_.end(); }
  FrameList::const_iterator begin() const { return frames_.begin(); }
  FrameList::const_iterator end() const { return frames_.end(); }
  FrameList::const_iterator cbegin() const { return frames_.cbegin(); }
  FrameList::const_iterator cend() const { return frames_.cend(); }

private:

  bool is_keyframe_ = false;

  BundleId bundle_id_;
};

/// Some helper functions for the frame object.
namespace frame_utils {

/// Creates an image pyramid of half-sampled images.
void createImgPyramid(
    const cv::Mat& img_level_0,
    int n_levels,
    ImgPyr& pyr);

/// Get the average depth of the features in the image.
bool getSceneDepth(
    const FramePtr& frame,
    double& depth_median,
    double& depth_min,
    double& depth_max);

void computeNormalizedBearingVectors(
    const Keypoints& px_vec,
    const Camera& cam,
    Bearings* f_vec);

} // namespace frame_utils
} // namespace svo
