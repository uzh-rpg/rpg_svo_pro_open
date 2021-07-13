/*
 * Keyframe.h
 *
 * This class contains all the relevant information about a keyframe that
 * is required for either loop closing or pose graph optimization.
 *
 *  Created on: Nov 8, 2018
 *      Author: kunal71091
 */

#pragma once

#include <svo/common/types.h>
#include <svo/common/transformation.h>

#include <vector>
#include <string>

// DBoW2 (courtesy: Dorian Galvez)
#include <DBoW2/DBoW2.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>

// logging
#include <glog/logging.h>

namespace svo
{
using BearingVecs =
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
class KeyFrame
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /*default constructor*/
  KeyFrame(int Nframeid)
  {
    NframeID_ = Nframeid;
  }

  /* Destructor */
  ~KeyFrame()
  {
  }

  /* Transform the landmarks given a Transformation */
  inline void transformMap(const Transformation& T)
  {
    LOG(FATAL) << "This should not be called with points"
                  " represented now in the camera frame.";
    for (size_t i = 0; i < svo_landmarksvector_cam_.size(); i++)
    {
      Eigen::Vector3d pos =
          Eigen::Vector3d(svo_landmarksvector_cam_[i].x, svo_landmarksvector_cam_[i].y,
                          svo_landmarksvector_cam_[i].z);
      pos = T.transform(pos);
      svo_landmarksvector_cam_[i] = cv::Point3f(pos.x(), pos.y(), pos.z());
    }
  }

  inline void getTwcCvMat(cv::Mat* Twc_cvmat)
  {
    Eigen::Matrix<double, 4, 4> Twc_mat = T_w_c_.getTransformationMatrix();
    (*Twc_cvmat) = (cv::Mat_<double>(4, 4) << Twc_mat(0, 0), Twc_mat(0, 1),
                    Twc_mat(0, 2), Twc_mat(0, 3), Twc_mat(1, 0), Twc_mat(1, 1),
                    Twc_mat(1, 2), Twc_mat(1, 3), Twc_mat(2, 0), Twc_mat(2, 1),
                    Twc_mat(2, 2), Twc_mat(2, 3), Twc_mat(3, 0), Twc_mat(3, 1),
                    Twc_mat(3, 2), Twc_mat(3, 3));
  }

  inline void clearSVOFeatureInfo()
  {
    svo_keypointsvector_.clear();
    svo_bearingvectors_.clear();
    svo_landmarksvector_cam_.clear();
    svo_landmark_ids_.clear();
    svo_depthsvector_.clear();
    svo_trackIDsvector_.clear();
    svo_featuretypevector_.clear();
    svo_original_indexvec_.clear();

    svo_features_mat_.release();
    svo_features_.clear();
    svo_node_ids_.clear();
  }

  inline void getLandmarksInWorld(std::vector<cv::Point3f>* pw_vec)
  {
    pw_vec->clear();
    for (const cv::Point3f& pc : svo_landmarksvector_cam_)
    {
      Eigen::Vector3d pw = T_w_c_.transform(Eigen::Vector3d(pc.x, pc.y, pc.z));
      pw_vec->emplace_back(cv::Point3f(pw.x(), pw.y(), pw.z()));
    }
  }

  // general information
  int NframeID_;
  int frame_id_;
  size_t lc_frame_count_;
  double timestamp_sec_abs_;
  cv::Mat keyframe_image_;

  // pose
  Transformation T_w_c_;

  // bow features without depth
  std::vector<cv::Point2f> bow_keypoints_;
  std::vector<cv::Mat> bow_features_;
  std::vector<int> bow_node_ids_;
  DBoW2::BowVector vec_bow_;

  // descriptors of SVO features: descriptor in Mat/vector and vocabulary node
  cv::Mat svo_features_mat_;
  std::vector<cv::Mat> svo_features_;
  std::vector<int> svo_node_ids_;

  // svo features
  std::vector<cv::Point2f> svo_keypointsvector_;
  BearingVecs svo_bearingvectors_;
  std::vector<cv::Point3f> svo_landmarksvector_cam_;
  std::vector<int> svo_landmark_ids_;
  std::vector<double> svo_depthsvector_;

  // track id: used to determine the common features
  std::vector<int> svo_trackIDsvector_;
  FeatureTypes svo_featuretypevector_;
  std::vector<size_t> svo_original_indexvec_;  // original index in frame class

  // mixed features for 2D check
  std::vector<cv::Point2f> mixed_keypoints_;
  std::vector<int> mixed_node_ids_;
  std::vector<cv::Mat> mixed_features_;

  int num_bow_features_;
  bool skip_frame_ = false;
};
using KeyFramePtr = std::shared_ptr<KeyFrame>;
}
