/*
 * Contains functions that aid in providing geometric verification between the
 * current
 * frame and loop closure candidates obtained from bow approach.
 */

/*
 * File:   geometricVerification.h
 * Author: kunal71091
 *,
 * Created on November 8, 2017
 */
#pragma once

#include <limits>
#include <vector>
#include <stdlib.h>
#include <math.h>

// eigen
#include <Eigen/Dense>

// opencv
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

// opengv
#ifdef SVO_USE_OPENGV
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#endif

// svo
#include <svo/common/frame.h>

// BoW
#include "svo/online_loopclosing/bow.h"

namespace svo
{
/* Function featureMatchingFast
 * Takes unmatched feature descriptor vectors as well as node information from
 * bag-of-words
 * and returns a matrix with matched indices. The search area is reduced using
 * the node information
 * which makes feature matching fast */

void featureMatchingFast(const std::vector<cv::Mat>& feature_vec1,
                         const std::vector<cv::Mat>& feature_vec2,
                         const std::vector<cv::Mat>& feature_vec3,
                         const std::vector<cv::Mat>& feature_vec4,
                         cv::Mat& svo_feature_mat1, cv::Mat& svo_feature_mat2,
                         const std::vector<int>& node_id1,
                         const std::vector<int>& node_id2,
                         const std::vector<int>& node_id3,
                         const std::vector<int>& node_id4, const int dist_th,
                         Eigen::MatrixXd* match_indices);

/* Function geometricVerification
 * Takes Keypoint vectors from two matched images as well as two empty point2f
 * vectors along with a Matrix of matched indexes
 * and populates the empty vectors with index matched keypoints. It then
 * calculates the relative pose and returns it */

void geometricVerification(const std::vector<cv::Point2f>& keypoints1,
                           const std::vector<cv::Point2f>& keypoints2,
                           const BearingVecs& svo_bearingvector1,
                           const BearingVecs& svo_bearingvector2,
                           const Eigen::MatrixXd& match_indices,
                           const cv::Mat& K, const Eigen::VectorXd& dist_par,
                           const int& bow_featurevec1_size,
                           const int& bow_featurevec2_size,
                           const bool& use_open_gv, cv::Mat* inliers,
                           std::vector<cv::Point2f>* keypoints_matched1,
                           std::vector<cv::Point2f>* keypoints_matched2,
                           std::vector<cv::Point2f>* keypoints_matched1_udist,
                           std::vector<cv::Point2f>* keypoints_matched2_udist,
                           Eigen::MatrixXd* T);

/* Function getRelativePose
 * Takes Fundamental matrix as well as intrinsic calibration matrix as input
 * and provides the relative pose between two frames. */

void getRelativePose(const cv::Mat& eMatrix, const cv::Mat& K,
                     const std::vector<cv::Point2f>& keypoints_matched1,
                     const std::vector<cv::Point2f>& keypoints_matched2,
                     cv::Mat* inliers, Eigen::MatrixXd* T);

/* Function getScale
 * Finds the scale of relative pose between two frames.
 * Takes landmark as well as keypoint data as input (vectors, where
 * corresponding
 * features and landmarks have same size), finds the common landmarks, and
 * retrieves the scale based on ratio of distances to current camera centre */

float getScaleCL(const std::vector<cv::Point2f>& keypoints1,
                 const std::vector<cv::Point2f>& keypoints2,
                 const std::vector<cv::Point3f>& landmarks1,
                 const std::vector<cv::Point3f>& landmarks2,
                 const std::vector<int>& track_IDs1,
                 const std::vector<int>& track_IDs2,
                 const cv::Mat& cam_pose_current, const cv::Mat& K,
                 const Eigen::MatrixXd& relative_pose,
                 const float mindist_thresh = 0.3);

/* Function commonLandMarkCheck
 * Checks whether the new keyframe does not have enough common features as
 * compared to last added keyframe
 * parameter th, defines the threshold for the fraction of common landmarks */

bool commonLandMarkCheck(const std::vector<int>& track_IDs1,
                         const std::vector<int>& track_IDs2, const double th);

float getScaleMK(const std::vector<cv::Point2f>& keypoints_matched1,
                 const std::vector<cv::Point2f>& keypoints_matched2,
                 const std::vector<cv::Point3f>& landmarks1,
                 const std::vector<cv::Point3f>& landmarks2,
                 const Eigen::MatrixXd& match_indices, const cv::Mat& inliers,
                 const cv::Mat& cam_pose, const cv::Mat& K,
                 const Eigen::MatrixXd& relative_pose,
                 const int num_bow_features);
}  // namespace svo
