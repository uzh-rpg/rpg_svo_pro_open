/*
 * geometric_verification.cpp
 *
 *  Created on: Nov 8, 2017
 *      Author: kunal71091
 * Description: Includes functions necessary for geometric verification and pose
 * retrieval
 */

#include "svo/online_loopclosing/geometric_verification.h"

using namespace std;

namespace svo
{
void featureMatchingFast(const std::vector<cv::Mat>& feature_vec1,
                         const std::vector<cv::Mat>& feature_vec2,
                         const std::vector<cv::Mat>& feature_vec3,
                         const std::vector<cv::Mat>& feature_vec4,
                         cv::Mat& svo_feature_mat1, cv::Mat& svo_feature_mat2,
                         const std::vector<int>& node_id1,
                         const std::vector<int>& node_id2,
                         const std::vector<int>& node_id3,
                         const std::vector<int>& node_id4, const int dist_th,
                         Eigen::MatrixXd* match_indices)
{
  /* For every entry in feature_vec1, find a match in feature_vec2 with least
   hamming distance.
   Use nodeids to reduce search area. This is only done for bow features that
   have no 3d information*/
  int num_matches = 0;
  size_t feature_vec1_size = feature_vec1.size();
  size_t feature_vec2_size = feature_vec2.size();
  size_t feature_vec3_size = feature_vec3.size();
  size_t feature_vec4_size = feature_vec4.size();
  match_indices->conservativeResize(2, 1000);
  for (unsigned int i = 0; i < feature_vec1_size; i++)
  {
    int matched_index = -1;
    double hamming_dist_min = std::numeric_limits<double>::max();
    std::vector<int>::const_iterator iter = node_id2.begin();
    while ((iter = std::find(iter, node_id2.end(), node_id1[i])) !=
           node_id2.end())
    {
      int index = std::distance(node_id2.begin(), iter);
      iter++;
      if (feature_vec1[i].empty() || feature_vec2[index].empty())
      {
        std::cout << "Invalid Feature Mat 1&2" << std::endl;
        continue;
      }
      double hamming_dist =
          cv::norm(feature_vec1[i], feature_vec2[index], cv::NORM_HAMMING);
      if (hamming_dist < hamming_dist_min && hamming_dist < dist_th)
      {
        hamming_dist_min = hamming_dist;
        matched_index = index;
      }
    }
    if (matched_index != -1)
    {
      num_matches++;
      match_indices->col(num_matches - 1) << i, matched_index;
    }
  }
  //  std::cout<<"Feature Vec 1 Size "<<feature_vec1_size<<std::endl;
  //  std::cout<<"Feature Vec 2 Size "<<feature_vec2_size<<std::endl;
  /* For svo descriptors with 3d information, do a brute force matching */

  //  /*---------------------------------------------------------------------
  //  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  ////  cv::FlannBasedMatcher f_matcher;
  //  std::vector< cv::DMatch > matches;

  ////  if(svo_feature_mat1.type()!=CV_32F)
  ////  {
  ////    svo_feature_mat1.convertTo(svo_feature_mat1, CV_32F);
  ////  }
  ////  if(svo_feature_mat2.type()!=CV_32F)
  ////  {
  ////    svo_feature_mat2.convertTo(svo_feature_mat2, CV_32F);
  ////  }
  ////  std::cout<<"Type "<<svo_feature_mat1.type()<<" and
  ///"<<svo_feature_mat2.type()<<std::endl;
  ////  std::cout<<"Cols "<<svo_feature_mat1.cols <<" and
  ///"<<svo_feature_mat2.cols<<std::endl;
  //  if(svo_feature_mat1.cols != svo_feature_mat2.cols)
  //  {
  //    match_indices->conservativeResize(2, 0);
  //    return;
  //  }
  //  matcher.match( svo_feature_mat2, svo_feature_mat1, matches );
  ////  f_matcher.match( svo_feature_mat1, svo_feature_mat2, matches );

  ////  std::cout<<"================== New Matching Started
  ///================"<<std::endl;
  //  for(size_t i = 0; i < matches.size(); i++)
  //  {
  //    num_matches++;
  //    match_indices->col(num_matches - 1) << matches[i].trainIdx +
  //    feature_vec1_size, matches[i].queryIdx + feature_vec2_size;
  ////    std::cout<<match_indices->col(num_matches - 1)<<std::endl;
  ////    std::cout<<"--------------"<<std::endl;
  //  }
  //  std::cout<<"================== Matching Finished
  //  ================"<<std::endl;
  //  ----------------------------------------------------------------------------
  //  */
  /* For svo descriptors with 3d information, use a higher vocabulary level */
  //  std::cout<<"================== New Matching Started
  //  ================"<<std::endl;
  for (unsigned int i = 0; i < feature_vec4_size; i++)
  {
    int matched_index_svo = -1;
    double hamming_dist_min_svo = std::numeric_limits<double>::max();
    std::vector<int>::const_iterator iter = node_id3.begin();
    while ((iter = std::find(iter, node_id3.end(), node_id4[i])) !=
           node_id3.end())
    {
      int index = std::distance(node_id3.begin(), iter);
      iter++;
      if (feature_vec4[i].empty() || feature_vec3[index].empty())
      {
        std::cout << "Invalid Feature Mat 3&4" << std::endl;
        continue;
      }
      double hamming_dist_svo =
          cv::norm(feature_vec4[i], feature_vec3[index], cv::NORM_HAMMING);
      if (hamming_dist_svo < hamming_dist_min_svo && hamming_dist_svo < dist_th)
      {
        hamming_dist_min_svo = hamming_dist_svo;
        matched_index_svo = index;
      }
    }
    if (matched_index_svo != -1)
    {
      num_matches++;
      //      std::cout<<"Min Dist SVO for this match
      //      "<<hamming_dist_min_svo<<std::endl;
      //      std::cout<<"Index "<<i<<" matched to
      //      "<<matched_index_svo<<std::endl;
      //      std::cout<<"Index offset "<<feature_vec2_size<<std::endl;
      match_indices->col(num_matches - 1)
          << matched_index_svo + feature_vec1_size,
          i + feature_vec2_size;
    }
  }
  //  std::cout<<"================== Matching Finished
  //  ================"<<std::endl;
  VLOG(40) << "Total matches " << num_matches;
  match_indices->conservativeResize(2, num_matches);
}

void geometricVerification(const vector<cv::Point2f>& keypoints1,
                           const vector<cv::Point2f>& keypoints2,
                           const BearingVecs& svo_bearingvector1,
                           const BearingVecs& svo_bearingvector2,
                           const Eigen::MatrixXd& match_indices,
                           const cv::Mat& K, const Eigen::VectorXd& dist_par,
                           const int& bow_featurevec1_size,
                           const int& bow_featurevec2_size,
                           const bool& use_open_gv, cv::Mat* inliers,
                           vector<cv::Point2f>* keypoints_matched1,
                           vector<cv::Point2f>* keypoints_matched2,
                           vector<cv::Point2f>* keypoints_matched1_udist,
                           vector<cv::Point2f>* keypoints_matched2_udist,
                           Eigen::MatrixXd* T_rel)
{
  static int numInliers = 0;

  /* Distorted Keypoints From Fast Matching */
  //  vector<cv::Point2f> keypoints_matched1_dist(match_indices.cols());
  //  vector<cv::Point2f> keypoints_matched2_dist(match_indices.cols());
  vector<cv::Point2f>* keypoints_matched1_norm_udist = new vector<cv::Point2f>(
      match_indices.cols());  // normalised and undistorted matches
  vector<cv::Point2f>* keypoints_matched2_norm_udist =
      new vector<cv::Point2f>(match_indices.cols());

  /* Convert Eigen Vector to std::vector, so that it can be passed to
   * undistortPoints */
  std::vector<double> D;
  D.resize(dist_par.size());
  Eigen::VectorXd::Map(&D[0], dist_par.size()) = dist_par;
  if (D.size() < 4)
  {
    D.resize(4);
  }

  for (int i = 0; i < match_indices.cols(); i++)
  {
    keypoints_matched1->at(i) = keypoints1[match_indices(0, i)];
    keypoints_matched2->at(i) = keypoints2[match_indices(1, i)];
  }

  cv::undistortPoints(*keypoints_matched1, *keypoints_matched1_norm_udist, K, D,
                      cv::noArray(), cv::noArray());
  cv::undistortPoints(*keypoints_matched2, *keypoints_matched2_norm_udist, K, D,
                      cv::noArray(), cv::noArray());

  if (use_open_gv)
  {
    //  std::cout<<"================ Matches ================" <<std::endl;
    BearingVecs mixed_vec1, mixed_vec2;
    for (int i = 0; i < match_indices.cols(); i++)
    {
      //    std::cout<<match_indices(0,i)<<" matched with
      //    "<<match_indices(1,i)<<std::endl;
      if (match_indices(0, i) < bow_featurevec1_size &&
          match_indices(1, i) < bow_featurevec2_size)
      {
        Eigen::Vector3d bv1 =
            Eigen::Vector3d(keypoints_matched1_norm_udist->at(i).x,
                            keypoints_matched1_norm_udist->at(i).y, 1.0);
        bv1 = bv1 / bv1.norm();
        mixed_vec1.push_back(bv1);
        Eigen::Vector3d bv2 =
            Eigen::Vector3d(keypoints_matched2_norm_udist->at(i).x,
                            keypoints_matched2_norm_udist->at(i).y, 1.0);
        bv2 = bv2 / bv2.norm();
        mixed_vec2.push_back(bv2);
      }
    }
    for (int i = 0; i < match_indices.cols(); i++)
    {
      if (match_indices(0, i) >= bow_featurevec1_size &&
          match_indices(1, i) >= bow_featurevec2_size)
      {
        mixed_vec1.push_back(
            svo_bearingvector1[match_indices(0, i) - bow_featurevec1_size]);
        mixed_vec2.push_back(
            svo_bearingvector2[match_indices(1, i) - bow_featurevec2_size]);
      }
    }
    if (mixed_vec1.size() != keypoints_matched1_udist->size())
    {
      std::cout << "CHECK: sizes of bearing vec 1 and keypoints vec 1 "
                << mixed_vec1.size() << " and "
                << keypoints_matched1_udist->size() << std::endl;
      std::cout << "CHECK: sizes of bearing vec 2 and keypoints vec 2 "
                << mixed_vec2.size() << " and "
                << keypoints_matched2_udist->size() << std::endl;
      LOG(FATAL);
    }
    static double inlier_threshold =
        1.0 - std::cos(std::atan(1.5 / (2.0 * K.at<double>(0, 0))) +
                       std::atan(1.5 / (2.0 * K.at<double>(1, 1))));
    typedef opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem
        CentralRelative;
    opengv::relative_pose::CentralRelativeAdapter adapter(mixed_vec1,
                                                          mixed_vec2);
    std::shared_ptr<CentralRelative> problem_ptr(
        new CentralRelative(adapter, CentralRelative::STEWENIUS));
    opengv::sac::Ransac<CentralRelative> ransac;
    ransac.sac_model_ = problem_ptr;
    ransac.threshold_ = inlier_threshold;
    ransac.max_iterations_ = 200;
    ransac.probability_ = 0.99;
    //    std::cout << " Starting RANSAC inlier threshold
    //    "<<inlier_threshold<<std::endl;
    ransac.computeModel();

    inliers->create((int)mixed_vec1.size(), 1, CV_8U);
    *inliers = 0;
    VLOG(40) << "RANSAC took " << ransac.iterations_ << " iterations.";
    //    std::cout << "RANSAC took " << ransac.iterations_ <<"
    //    iterations."<<std::endl;
    for (size_t i = 0; i < ransac.inliers_.size(); i++)
    {
      inliers->at<bool>(ransac.inliers_[i], 0) = 1;
    }
    T_rel->conservativeResize(4, 4);
    T_rel->setIdentity();
    Eigen::Vector3d t = ransac.model_coefficients_.rightCols(1);
    Eigen::Matrix3d R = ransac.model_coefficients_.leftCols(3);
    T_rel->block(0, 0, 3, 3) = R;
    T_rel->block(0, 3, 3, 1) = t;
  }
  else
  {
    for (int i = 0; i < match_indices.cols(); i++)
    {
      keypoints_matched1_udist->at(i).x =
          keypoints_matched1_norm_udist->at(i).x * K.at<double>(0, 0) +
          K.at<double>(0, 2);
      keypoints_matched1_udist->at(i).y =
          keypoints_matched1_norm_udist->at(i).y * K.at<double>(1, 1) +
          K.at<double>(1, 2);
      keypoints_matched2_udist->at(i).x =
          keypoints_matched2_norm_udist->at(i).x * K.at<double>(0, 0) +
          K.at<double>(0, 2);
      keypoints_matched2_udist->at(i).y =
          keypoints_matched2_norm_udist->at(i).y * K.at<double>(1, 1) +
          K.at<double>(1, 2);
    }
    cv::Mat ematrix;
    ematrix = cv::findEssentialMat(*keypoints_matched1_udist,
                                   *keypoints_matched2_udist, K, cv::RANSAC,
                                   0.995, 1., *inliers);
    //  ematrix = cv::findEssentialMat(*keypoints_matched1_udist,
    //  *keypoints_matched2_udist, K, cv::LMEDS, 0.999, 1.,
    //  *inliers);
    getRelativePose(ematrix, K, *keypoints_matched1_udist,
                    *keypoints_matched2_udist, inliers, T_rel);

    //     Also check if the returned essential matrix is feasible eg: if it is
    //     9X3 then we cannot reliably find the
    //     inliers and this will affect the pose retrieval
    if (ematrix.rows > 3 || ematrix.empty())
    {
      inliers = 0;
    }
  }
}

void getRelativePose(const cv::Mat& eMatrix, const cv::Mat& K,
                     const vector<cv::Point2f>& keypoints_matched1,
                     const vector<cv::Point2f>& keypoints_matched2,
                     cv::Mat* inliers, Eigen::MatrixXd* T)
{
  cv::Mat R, t, mask;
  double focal = (K.at<double>(0, 0) + K.at<double>(1, 1)) / 2;
  cv::Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));
  int inlierCount = cv::recoverPose(eMatrix, keypoints_matched1,
                                    keypoints_matched2, K, R, t, *inliers);
  if (cv::determinant(R) < 0)
  {
    R = -R;
  }

  T->conservativeResize(4, 4);
  *T << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
      t.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(1, 1),
      R.at<double>(1, 2), t.at<double>(1, 0), R.at<double>(2, 0),
      R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0), 0, 0, 0, 1;
}

float getScaleCL(const std::vector<cv::Point2f>& keypoints1,
                 const std::vector<cv::Point2f>& keypoints2,
                 const std::vector<cv::Point3f>& landmarks1,
                 const std::vector<cv::Point3f>& landmarks2,
                 const std::vector<int>& track_IDs1,
                 const std::vector<int>& track_IDs2,
                 const cv::Mat& cam_pose_current, const cv::Mat& K,
                 const Eigen::MatrixXd& relative_pose,
                 const float mindist_thresh)
{
  float scale;
  float cumulative_scale = 0;
  float mindist;
  int count = 0;
  signed int minindex;

  cv::Mat P1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  P1 = K * P1;
  cv::Mat P2 = (cv::Mat_<double>(3, 4) << relative_pose(0, 0),
                relative_pose(0, 1), relative_pose(0, 2), relative_pose(0, 3),
                relative_pose(1, 0), relative_pose(1, 1), relative_pose(1, 2),
                relative_pose(1, 3), relative_pose(2, 0), relative_pose(2, 1),
                relative_pose(2, 2), relative_pose(2, 3));
  P2 = K * P2;

  // get common landmarks
  for (unsigned int i = 0; i < landmarks1.size(); i++)
  {
    mindist = mindist_thresh;
    minindex = -1;
    for (unsigned int j = 0; j < landmarks2.size(); j++)
    {
      if (track_IDs1[i] == track_IDs2[j] &&
          cv::norm(landmarks1[i] - landmarks2[j]) < mindist)
      {
        minindex = j;
        break;
      }
      else if (cv::norm(landmarks1[i] - landmarks2[j]) < mindist)
      {
        mindist = cv::norm(landmarks1[i] - landmarks2[j]);
        minindex = j;
      }
    }
    if (minindex > -1)
    {
      count++;
      cv::Mat svo3DpntHW = (cv::Mat_<double>(4, 1) << landmarks1[i].x,
                            landmarks1[i].y, landmarks1[i].z, 1);
      cv::Mat svopnt3DH = (cv::Mat_<double>(4, 1) << 1, 1, 1, 1);
      svopnt3DH = cam_pose_current.inv() * svo3DpntHW;  // in camera 1 frame
      cv::Point3f svopnt3D;
      svopnt3D.x = svopnt3DH.at<double>(0, 0) / svopnt3DH.at<double>(3, 0);
      svopnt3D.y = svopnt3DH.at<double>(1, 0) / svopnt3DH.at<double>(3, 0);
      svopnt3D.z = svopnt3DH.at<double>(2, 0) / svopnt3DH.at<double>(3, 0);
      // in current camera frame
      float dl = cv::norm(svopnt3D);

      // Linear triangulation
      cv::Mat pnt3DH(4, 1, CV_64FC4);
      cv::triangulatePoints(P1, P2, cv::Mat(keypoints1[i]),
                            cv::Mat(keypoints2[minindex]), pnt3DH);
      cv::Point3f pnt3D;
      pnt3D.x = pnt3DH.at<float>(0, 0) / pnt3DH.at<float>(3, 0);
      pnt3D.y = pnt3DH.at<float>(1, 0) / pnt3DH.at<float>(3, 0);
      pnt3D.z = pnt3DH.at<float>(2, 0) / pnt3DH.at<float>(3, 0);

      // df
      float df = cv::norm(pnt3D);
      cumulative_scale += dl / df;
    }
  }

  if (count == 0 || std::isnan(cumulative_scale) || std::isnan(count))
  {
    scale = 1;
    VLOG(40) << "Scale could not be retrieved, use only rotation for pose "
                "optimisation";
  }
  else
  {
    scale = cumulative_scale / count;
  }

  return scale;
}

bool commonLandMarkCheck(const std::vector<int>& track_IDs1,
                         const std::vector<int>& track_IDs2, const double th)
{
  int common_landmarks = 0;
  for (size_t i = 0; i < track_IDs1.size(); i++)
  {
    for (size_t j = 0; j < track_IDs2.size(); j++)
    {
      if (track_IDs1[i] == track_IDs2[j])
      {
        common_landmarks++;
        break;
      }
    }
  }
  double common_landmark_percentage =
      double(common_landmarks) / track_IDs1.size();
  //  std::cout<<"+++++++++++ Number of Common Landmarks
  //  "<<common_landmark_percentage<<std::endl;
  return (common_landmark_percentage < th ||
          std::isnan(common_landmark_percentage));  // Nan means no information.
                                                    // In this case it should be
                                                    // safer to add a keyframe
                                                    // to database rather than
                                                    // not.
}

float getScaleMK(const std::vector<cv::Point2f>& keypoints_matched1,
                 const std::vector<cv::Point2f>& keypoints_matched2,
                 const std::vector<cv::Point3f>& landmarks1,
                 const std::vector<cv::Point3f>& landmarks2,
                 const Eigen::MatrixXd& match_indices, const cv::Mat& inliers,
                 const cv::Mat& cam_pose, const cv::Mat& K,
                 const Eigen::MatrixXd& relative_pose,
                 const int num_bow_features)
{
  float scale2;
  float cumulative_scale = 0;
  int count = 0;

  cv::Mat P1 = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  P1 = K * P1;
  Eigen::Matrix3d R;
  R << relative_pose(0, 0), relative_pose(0, 1), relative_pose(0, 2),
      relative_pose(1, 0), relative_pose(1, 1), relative_pose(1, 2),
      relative_pose(2, 0), relative_pose(2, 1), relative_pose(2, 2);
  Eigen::Vector3d t;
  t << relative_pose(0, 3), relative_pose(1, 3), relative_pose(2, 3);
  R.transposeInPlace();
  t = -R * t;
  Eigen::MatrixXd proj_mat(3, 4);
  proj_mat << R, t;
  cv::Mat P2 = (cv::Mat_<double>(3, 4) << proj_mat(0, 0), proj_mat(0, 1),
                proj_mat(0, 2), proj_mat(0, 3), proj_mat(1, 0), proj_mat(1, 1),
                proj_mat(1, 2), proj_mat(1, 3), proj_mat(2, 0), proj_mat(2, 1),
                proj_mat(2, 2), proj_mat(2, 3));
  //  cv::Mat P2 = (cv::Mat_<double>(3, 4) << relative_pose(0, 0),
  //  relative_pose(0, 1), relative_pose(0, 2), relative_pose(0, 3),
  //                                          relative_pose(1, 0),
  //                                          relative_pose(1, 1),
  //                                          relative_pose(1, 2),
  //                                          relative_pose(1, 3),
  //                                          relative_pose(2, 0),
  //                                          relative_pose(2, 1),
  //                                          relative_pose(2, 2),
  //                                          relative_pose(2, 3));
  P2 = K * P2;
  std::cout << "LC frame pose Mat from recover pose" << std::endl;
  std::cout << cam_pose << std::endl;
  for (int i = 0; i < match_indices.cols(); i++)
  {
    // check whether there are any inliers from svo keypoints
    if (inliers.at<bool>(i, 0) == 1 && match_indices(0, i) > num_bow_features)
    {
      std::cout << "match index " << match_indices(0, i) << std::endl;
      count++;
      // Linear triangulation
      cv::Mat pnt3DH(4, 1, CV_64FC4);
      cv::triangulatePoints(P1, P2, cv::Mat(keypoints_matched2[i]),
                            cv::Mat(keypoints_matched1[i]), pnt3DH);
      cv::Point3f pnt3D;
      pnt3D.x = pnt3DH.at<float>(0, 0) / pnt3DH.at<float>(3, 0);
      pnt3D.y = pnt3DH.at<float>(1, 0) / pnt3DH.at<float>(3, 0);
      pnt3D.z = pnt3DH.at<float>(2, 0) / pnt3DH.at<float>(3, 0);

      std::cout << "Triangulated Point " << std::endl;
      std::cout << pnt3D << std::endl;

      // df
      float df = cv::norm(pnt3D);

      // dl
      cv::Mat P_h =
          (cv::Mat_<double>(4, 1)
               << landmarks2[match_indices(1, i) - num_bow_features].x,
           landmarks2[match_indices(1, i) - num_bow_features].y,
           landmarks2[match_indices(1, i) - num_bow_features].z, 1.0);
      P_h = cam_pose.inv() * P_h;
      cv::Point3f P;
      P.x = P_h.at<double>(0, 0) / P_h.at<double>(3, 0);
      P.y = P_h.at<double>(1, 0) / P_h.at<double>(3, 0);
      P.z = P_h.at<double>(2, 0) / P_h.at<double>(3, 0);
      std::cout << "Actual Point " << std::endl;
      std::cout << P << std::endl;
      float dl = cv::norm(P);

      cumulative_scale += dl / df;
      std::cout << "Scale: " << dl / df << std::endl;
    }
  }

  if (count == 0 || std::isnan(cumulative_scale) || std::isnan(count))
  {
    scale2 = 1;
    VLOG(40) << "Scale could not be retrieved, use only rotation for pose "
                "optimisation";
  }
  else
  {
    scale2 = cumulative_scale / count;
  }

  return scale2;
}

}  // namespace svo
