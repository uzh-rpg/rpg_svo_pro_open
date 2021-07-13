/*
 * loop_closing.cpp
 *
 *  Created on: Nov 9, 2017
 *      Author: kunal71091
 */

/*
 * Place Recognition pipeline.
 */

#include "svo/online_loopclosing/loop_closing.h"

#include <unistd.h>

#include <glog/logging.h>

#include "svo/online_loopclosing/map_alignment.h"

using namespace std;
using namespace DBoW2;

namespace svo
{
std::map<std::string, LCScaleRetMethod> kStrToScaleRetMap{
  { std::string("CommonLM"), LCScaleRetMethod::kCommonLandmarks },
  { std::string("MixedKP"), LCScaleRetMethod::kMixedKeyPoints },
  { std::string("None"), LCScaleRetMethod::kNone }
};

std::map<std::string, GlobalMapType> kStrToGlobalMapType{
  { std::string("BuiltInPoseGraph"), GlobalMapType::kBuiltInPoseGraph },
  { std::string("ExternalGlobalMap"), GlobalMapType::kExternalGlobalMap },
  { std::string("None"), GlobalMapType::kNone }
};

LoopClosing::LoopClosing(const LoopClosureOptions& loopclosure_options,
                         const CameraBundle::Ptr& cams_)
  : options_(loopclosure_options)
{
  std::stringstream path;
  path << options_.voc_path << options_.voc_name;
  std::string voc_path_full = path.str();
  voc_ = loadVoc(voc_path_full);
  Eigen::VectorXd intrinsics =
      cams_->getCameraShared(0)->getIntrinsicParameters();
  K_ = (cv::Mat_<double>(3, 3) << intrinsics(0), 0, intrinsics(2), 0,
        intrinsics(1), intrinsics(3), 0, 0, 1);
  D_ = cams_->getCameraShared(0)->getDistortionParameters();
  T_C_B_ = cams_->get_T_C_B(0);
  T_B_C_ = T_C_B_.inverse();

  CHECK(kStrToScaleRetMap.find(options_.scale_ret_app) !=
        kStrToScaleRetMap.end());
  scale_retrieval_approach_ = kStrToScaleRetMap[options_.scale_ret_app];

  CHECK(kStrToGlobalMapType.find(options_.global_map_type) !=
        kStrToGlobalMapType.end());
  global_map_type_ = kStrToGlobalMapType[options_.global_map_type];

  if (global_map_type_ == GlobalMapType::kBuiltInPoseGraph)
  {
    pgo_ = std::make_shared<Pgo>();
  }

  CHECK_GE(options_.ignored_past_frames, 0);

  int sys =
      system(("exec rm -r " + options_.image_log_base_path + "*").c_str());
}

LoopClosing::~LoopClosing()
{
}

void LoopClosing::runPROnLatestKeyframe(const size_t n_ignored_latest,
                                        const bool run_lc_on_this_frame,
                                        const double score_expected)
{
  const KeyFramePtr& cur_kf = kf_list_.back();
  const size_t cur_idx = kf_list_.size() - 1;
  lc_frame_count_++;
  cur_kf->lc_frame_count_ = lc_frame_count_;
  int current_frame_ID = cur_kf->NframeID_;

  VLOG(40) << ">>>> Frame ID " << current_frame_ID << " added to PR. Timestamp "
           << cur_kf->timestamp_sec_abs_;

  /* Save The frame image for comparison*/
  if (options_.enable_image_logging)
  {
    std::stringstream image_path;
    image_path << options_.image_log_base_path
               << std::to_string(current_frame_ID) << ".jpg";
    std::string img_path = image_path.str();
    cv::imwrite(img_path, cur_kf->keyframe_image_);
  }

  vk::Timer timer_total;
  vk::Timer timer_each;
  timer_total.start();
  timer_each.start();
  extractBoWFeaturesFromImage(cur_kf->keyframe_image_, &cur_kf->bow_keypoints_,
                              &cur_kf->bow_features_);
  cur_kf->num_bow_features_ = static_cast<int>(cur_kf->bow_features_.size());
  createBOW(cur_kf->bow_features_, voc_, &cur_kf->vec_bow_,
            &cur_kf->bow_node_ids_);

  updateSVOPointsDescriptors(cur_idx, false);
  double t_extract = timer_each.stop();
  VLOG(40) << "Time Taken for extract information from this keyframe "
           << t_extract << " s";

  if (suspend_lc_after_correction_)
  {
    suspended_frames_counter_++;
    if (suspended_frames_counter_ < static_cast<int>(options_.alpha * 20 + 5))
    {
      VLOG(40) << "********** Loop Closing Suspended *********";
      completed_flags_.back() = true;
      return;
    }
    else
    {
      VLOG(40) << "********** Loop Closing Resumed *********";
      suspend_lc_after_correction_ = false;
      suspended_frames_counter_ = 0;
    }
  }

  /* Perform place recognition
   1. Find loop closure candidates using bow comparison
   2. Perform geometric verification and keep the candidates with
   high score. We also obtain the relative pose between current
   frame and loop closure candidates in this step. */
  if (static_cast<size_t>(lc_frame_count_) <= n_ignored_latest ||
      !run_lc_on_this_frame)
  {
    completed_flags_.back() = true;
    return;
  }

  VLOG(40) << "Running Loop Closure on Frame ID " << current_frame_ID;
  num_queries_.push_back(0);
  for (size_t query_kf_idx = 10;
       query_kf_idx < kf_list_.size() - static_cast<size_t>(n_ignored_latest);
       query_kf_idx++)
  {
    const KeyFramePtr& lc_kf_i = kf_list_[query_kf_idx];

    if (lc_kf_i->skip_frame_)
    {
      continue;
    }

    if (!recovery_after_loss_ && !proximityCheck(cur_kf, lc_kf_i))
    {
      continue;
    }

    cur_loop_check_viz_info_.emplace_back(LoopVizInfo());
    constructLoopViz(*cur_kf, *lc_kf_i, &(cur_loop_check_viz_info_.back()));
    // Step 1: BoW comparison
    int past_NframeID = lc_kf_i->NframeID_;
    VLOG(40) << "- Will check against frame (that passes the distance check) "
             << past_NframeID;
    num_queries_.back()++;
    double score = 0.0;
    double score_normalized = 0.0;
    timer_each.start();
    // Extract frame ID
    // Check if successful BoW candidate (step 1 above) by comparing
    // normalised score to a threshold.
    score = compareBOWs(cur_kf->vec_bow_, lc_kf_i->vec_bow_, voc_);
    score_normalized = score / score_expected;

    double t_bow_comp = timer_each.stop();
    addNewTimingSlot();
    bow_timing_.back() = t_bow_comp;
    VLOG(40) << "1-BOW: Time Taken for bow comparison of this query "
             << t_bow_comp << " s";
    if (score_normalized < options_.bowthresh)
    {
      VLOG(40) << "1-BOW check FAILED.";
      continue;
    }

    // logging
    vector<cv::KeyPoint> svo_keypoints_current_kp;
    vector<cv::KeyPoint> svo_keypoints_past_kp;
    if (options_.enable_image_logging)
    {
      for (size_t i = 0; i < cur_kf->svo_keypointsvector_.size(); ++i)
      {
        svo_keypoints_current_kp.push_back(
            cv::KeyPoint(cur_kf->svo_keypointsvector_[i], 1.f));
      }
      for (size_t i = 0; i < lc_kf_i->svo_keypointsvector_.size(); ++i)
      {
        svo_keypoints_past_kp.push_back(
            cv::KeyPoint(lc_kf_i->svo_keypointsvector_[i], 1.f));
      }
    }

    // Step 2: Geometric Verification (step 2)
    // 1. Get Feature matches
    timer_each.start();
    Eigen::MatrixXd match_indices;
    featureMatchingFast(cur_kf->bow_features_, lc_kf_i->bow_features_,
                        cur_kf->svo_features_, lc_kf_i->svo_features_,
                        cur_kf->svo_features_mat_, lc_kf_i->svo_features_mat_,
                        cur_kf->bow_node_ids_, lc_kf_i->bow_node_ids_,
                        cur_kf->svo_node_ids_, lc_kf_i->svo_node_ids_,
                        options_.orb_dist_thresh, &match_indices);

    /* here we check whether we even have enough number of matches. If this
     number is small, then later a high percent
     inlier match will be erroneous. */
    const double match_ratio =
        double(match_indices.cols()) / cur_kf->mixed_features_.size();
    VLOG(40) << "2-GV2D: Matching ratio of all features " << match_ratio;
    if (match_ratio < options_.gv_2d_match_thresh)
    {
      VLOG(40) << "2-GV2D check FAILED.";
      continue;
    }

    // 2. Perform essential-Matrix Ransac and check if number of inliers are
    // enough.
    int num_inliers = 0;
    int num_3d_inliers = 0;
    Eigen::MatrixXd T_rel;
    cv::Mat inliers;
    vector<cv::Point2f> keypoints_matched1(match_indices.cols());
    vector<cv::Point2f> keypoints_matched2(match_indices.cols());
    vector<cv::Point2f> keypoints_matched1_udist(match_indices.cols());
    vector<cv::Point2f> keypoints_matched2_udist(match_indices.cols());
    geometricVerification(cur_kf->mixed_keypoints_, lc_kf_i->mixed_keypoints_,
                          cur_kf->svo_bearingvectors_,
                          lc_kf_i->svo_bearingvectors_, match_indices, K_, D_,
                          cur_kf->num_bow_features_, lc_kf_i->num_bow_features_,
                          options_.use_opengv, &inliers, &keypoints_matched1,
                          &keypoints_matched2, &keypoints_matched1_udist,
                          &keypoints_matched2_udist, &T_rel);
    // count number of inliers.
    std::vector<IdCorrespondence> lc_to_cur_inlier_corresp_3d;
    std::vector<IdCorrespondence> lc_to_cur_inlier_corresp;
    for (int l = 0; l < match_indices.cols(); l++)
    {
      if (int(inliers.at<bool>(l, 0)) == 1)
      {
        lc_to_cur_inlier_corresp.push_back(IdCorrespondence());
        lc_to_cur_inlier_corresp.back() = std::pair<size_t, size_t>(
            size_t(match_indices(1, l)), size_t(match_indices(0, l)));
        num_inliers++;
        if (match_indices(0, l) > cur_kf->num_bow_features_ &&
            match_indices(1, l) > lc_kf_i->num_bow_features_)
        {
          lc_to_cur_inlier_corresp_3d.push_back(IdCorrespondence());
          lc_to_cur_inlier_corresp_3d.back() = std::pair<size_t, size_t>(
              size_t(match_indices(1, l) - lc_kf_i->num_bow_features_),
              size_t(match_indices(0, l) - cur_kf->num_bow_features_));
          num_3d_inliers++;
        }
      }
    }
    double t_match_gv = timer_each.stop();
    gv_timing_.back() = t_match_gv;
    VLOG(40) << "3-GV3D:Time Taken for geometric verification between "
                "keyframes "
             << current_frame_ID << " and " << past_NframeID << " with "
             << match_indices.cols() << " matches " << t_match_gv << " s";
    double gv_inliers_ratio = (double(num_inliers) / inliers.rows);
    VLOG(40) << "3-GV3D: Percentage of inliers " << gv_inliers_ratio
             << " and number of 3d inliers " << num_3d_inliers << std::endl;
    int cur_min_3d_thresh = options_.min_num_3d;
    if (gv_inliers_ratio > options_.gv_3d_inlier_thresh)
    {
    }
    if (recovery_after_loss_)
    {
      cur_min_3d_thresh = options_.min_num_3d - 3;
    }
    if (gv_inliers_ratio < options_.gv_3d_inlier_thresh ||
        num_3d_inliers < cur_min_3d_thresh)
    {
      VLOG(40) << "3-GV3D check failed.";
      continue;
    }

    loop_detect_viz_info_.emplace_back(LoopVizInfo());
    constructLoopViz(*cur_kf, *lc_kf_i, &(loop_detect_viz_info_.back()));

    vector<cv::Point2f> mixed_keypoints_norm_udist_cf(
        cur_kf->mixed_keypoints_.size());
    vector<cv::Point2f> mixed_keypoints_norm_udist_lc(
        lc_kf_i->mixed_keypoints_.size());
    undistortAndNormalise(cur_kf->mixed_keypoints_, lc_kf_i->mixed_keypoints_,
                          K_, D_, &mixed_keypoints_norm_udist_cf,
                          &mixed_keypoints_norm_udist_lc);

    // Get Scale: not needed if we use map alignment at the end
    // get scale of relative translation
    std::vector<cv::Point3f> cur_pw_vec;
    cur_kf->getLandmarksInWorld(&cur_pw_vec);
    std::vector<cv::Point3f> lc_pw_vec;
    lc_kf_i->getLandmarksInWorld(&lc_pw_vec);
    float scale = 1;
    vector<int> svo_trackIDs_current = cur_kf->svo_trackIDsvector_;
    vector<int> svo_trackIDs_past = lc_kf_i->svo_trackIDsvector_;
    if (scale_retrieval_approach_ == LCScaleRetMethod::kCommonLandmarks)
    {
      // common landmarks approach
      VLOG(40) << "Common Landmarks Approach";
      cv::Mat cur_cam_pose_mat;
      cur_kf->getTwcCvMat(&cur_cam_pose_mat);
      scale = getScaleCL(cur_kf->svo_keypointsvector_,
                         lc_kf_i->svo_keypointsvector_, cur_pw_vec, lc_pw_vec,
                         svo_trackIDs_current, svo_trackIDs_past,
                         cur_cam_pose_mat, K_, T_rel);
    }
    else if (scale_retrieval_approach_ == LCScaleRetMethod::kMixedKeyPoints)
    {
      // mixed keypoints approach
      VLOG(40) << "Mixed Keypoints Approach";
      cv::Mat cur_cam_pose_mat;
      cur_kf->getTwcCvMat(&cur_cam_pose_mat);
      scale =
          getScaleMK(keypoints_matched1_udist, keypoints_matched2_udist,
                     cur_pw_vec, lc_pw_vec, match_indices, inliers,
                     cur_cam_pose_mat, K_, T_rel, cur_kf->num_bow_features_);
    }
    else
    {
      VLOG(40) << "Will not retrieve scale.";
    }
    // Update the relative pose.
    T_rel.block(0, 3, 3, 1) = scale * T_rel.block(0, 3, 3, 1);
    VLOG(40) << "scale " << scale;
    VLOG(40) << T_rel;
    VLOG(40) << "Loop Closure Detected between frame " << current_frame_ID
             << " and " << past_NframeID;

    /* Draw keypoint matches to check how good correspondences are*/
    if (options_.enable_image_logging)
    {
      vector<cv::KeyPoint> keypoints1;
      vector<cv::KeyPoint> keypoints2;
      vector<cv::KeyPoint> keypoints1_all;
      vector<cv::KeyPoint> keypoints2_all;
      vector<cv::KeyPoint> keypoints1_all_svo;
      vector<cv::KeyPoint> keypoints2_all_svo;
      vector<cv::DMatch> matches12_all, matches12, matches12_all_svo;

      int o = -1;
      int p = -1;
      int m = -1;

      for (int i = 0; i < match_indices.cols(); i++)
      {
        if (int(inliers.at<bool>(i, 0)) == 1)
        {
          p++;
          keypoints1_all.push_back(cv::KeyPoint());
          keypoints2_all.push_back(cv::KeyPoint());
          matches12_all.push_back(cv::DMatch());
          keypoints1_all.back() = cv::KeyPoint(keypoints_matched1[i], 1.f);
          keypoints2_all.back() = cv::KeyPoint(keypoints_matched2[i], 1.f);
          matches12_all.back() = cv::DMatch(p, p, 1.0);
        }
        if (int(inliers.at<bool>(i, 0)) == 1 &&
            match_indices(0, i) >= cur_kf->num_bow_features_ &&
            match_indices(1, i) >= lc_kf_i->num_bow_features_)
        {
          o++;
          keypoints1.push_back(cv::KeyPoint());
          keypoints2.push_back(cv::KeyPoint());
          matches12.push_back(cv::DMatch());
          keypoints1.back() = cv::KeyPoint(keypoints_matched1[i], 1.f);
          keypoints2.back() = cv::KeyPoint(keypoints_matched2[i], 1.f);
          matches12.back() = cv::DMatch(o, o, 1.0);
        }
        if (match_indices(0, i) >= cur_kf->num_bow_features_ &&
            match_indices(1, i) >= lc_kf_i->num_bow_features_)
        {
          m++;
          keypoints1_all_svo.push_back(
              cv::KeyPoint(keypoints_matched1[i], 1.f));
          keypoints2_all_svo.push_back(
              cv::KeyPoint(keypoints_matched2[i], 1.f));
          matches12_all_svo.push_back(cv::DMatch(m, m, 1.0));
        }
      }
      VLOG(40) << "Marked keypoints number " << o + 1;

      std::stringstream prev_img_path, window_path, window_path2, window_path3,
          out_img_cf_path, out_img_lc_path;
      prev_img_path << options_.image_log_base_path
                    << std::to_string(past_NframeID) << ".jpg";
      cv::Mat prev_img, out_img, out_img_all, out_img_all_svo, out_img_cf,
          out_img_lc;
      prev_img = cv::imread(prev_img_path.str());
      //--------------------------------------------------------------------------------------------//
      cv::drawKeypoints(cur_kf->keyframe_image_, svo_keypoints_current_kp,
                        out_img_cf);
      out_img_cf_path << options_.image_log_base_path
                      << std::to_string(current_frame_ID) << ".jpg";
      cv::imwrite(out_img_cf_path.str(), out_img_cf);
      cv::drawKeypoints(prev_img, svo_keypoints_past_kp, out_img_lc);
      out_img_lc_path << options_.image_log_base_path
                      << std::to_string(past_NframeID) << ".jpg";
      cv::imwrite(out_img_lc_path.str(), out_img_lc);
      //--------------------------------------------------------------------------------------------//
      cv::drawMatches(cur_kf->keyframe_image_, keypoints1, prev_img, keypoints2,
                      matches12, out_img);
      window_path << options_.image_log_base_path
                  << std::to_string(past_NframeID) << "_"
                  << std::to_string(current_frame_ID) << ".jpg";
      cv::imwrite(window_path.str(), out_img);
      cv::drawMatches(cur_kf->keyframe_image_, keypoints1_all, prev_img,
                      keypoints2_all, matches12_all, out_img_all);
      window_path2 << options_.image_log_base_path
                   << std::to_string(past_NframeID) << "_"
                   << std::to_string(current_frame_ID) << "_all.jpg";
      cv::imwrite(window_path2.str(), out_img_all);
      cv::drawMatches(cur_kf->keyframe_image_, keypoints1_all_svo, prev_img,
                      keypoints2_all_svo, matches12_all_svo, out_img_all_svo);
      window_path3 << options_.image_log_base_path
                   << std::to_string(past_NframeID) << "_"
                   << std::to_string(current_frame_ID) << "_all_svo.jpg";
      cv::imwrite(window_path3.str(), out_img_all_svo);
    }

    // get 6 DoF correction
    Transformation w_T_new_old;
    std::vector<int> inlier_3d_indices;
    timer_each.start();
    bool suc = calculateTransformationInWorldFrame(
        lc_pw_vec, cur_pw_vec,
        lc_to_cur_inlier_corresp_3d, current_frame_ID, past_NframeID,
        &w_T_new_old, &inlier_3d_indices);
    hm_timing_.back() = timer_each.stop();
    if (!suc)
    {
      continue;
    }

    {
      MatchedPointsInfo cur_match_info;
      cur_match_info.lc_kf_id_ = lc_kf_i->frame_id_;
      cur_match_info.cur_kf_id_ = cur_kf->frame_id_;
      for (const int idx_3d : inlier_3d_indices)
      {
        auto v = lc_to_cur_inlier_corresp_3d[idx_3d];
        cur_match_info.pt_id_matches_.insert(
            std::make_pair(lc_kf_i->svo_landmark_ids_[v.first],
                           cur_kf->svo_landmark_ids_[v.second]));
      }
      std::lock_guard<std::mutex> lock(lc_info_lock_);
      lc_matched_points_info_.push_back(cur_match_info);
    }

    VLOG(40) << "4.HM: successfully get 6 DoF transformation.";
    loop_correction_viz_info_.emplace_back(LoopVizInfo());
    constructLoopViz(*cur_kf, *lc_kf_i, &(loop_correction_viz_info_.back()));

    // save loop information for other modules (visualization, backend)
    {
      std::lock_guard<std::mutex> lock(lc_info_lock_);
      if (recovery_after_loss_ ||
          w_T_new_old.getPosition().norm() > options_.force_correction_dist_thresh_meter)
      {
        lc_correction_info_.emplace_back(lc_kf_i->NframeID_, cur_kf->NframeID_,
                                         w_T_new_old);
        recovery_after_loss_ = false;
        suspend_lc_after_correction_ = true;
        cumulative_distance_ = 0;
      }
      lc_closed_loops_.emplace_back(
          lc_kf_i->NframeID_, cur_kf->NframeID_, lc_kf_i->timestamp_sec_abs_,
          cur_kf->timestamp_sec_abs_,
          lc_kf_i->T_w_c_.inverse() * w_T_new_old * cur_kf->T_w_c_);
    }

    break;
  }

  completed_flags_.back() = true;
  VLOG(40) << "<<< Total Time Taken for processing keyframe "
           << cur_kf->NframeID_ << " " << timer_total.stop()
           << "s and query count " << num_queries_.back();
}

void LoopClosing::addFrameToPR(const svo::FrameBundlePtr& last_frames_)
{
  if (completed_flags_.size() > 1 && !completed_flags_.back())
  {
    return;
  }
  svo_keyframe_count_++;
  const cv::Mat current_frame_image = last_frames_->frames_[0]->img();

  // create bag of words vector for retrieval
  {
    vector<cv::Mat> feature_kf;
    vector<cv::Point2f> keypoints_kf;
    extractBoWFeaturesFromImage(current_frame_image, &keypoints_kf,
                                &feature_kf);
    svokf_bow_vec_.push_back(BowVector());
    createBOW(feature_kf, voc_, &svokf_bow_vec_.back(), nullptr);
  }

  // decide whether we need to run loop closing on this frame
  bool add_this_frame;
  bool run_lc_on_this_frame;
  cur_loop_check_viz_info_.clear();
  /* Call place recognition function using multithreading */
  if (svo_keyframe_count_ == 1)
  {
    run_lc_on_this_frame = true;

    VLOG(40) << "************ Adding First Keyframe to PR *****************";
    std::shared_ptr<KeyFrame> cur_kf =
        std::make_shared<KeyFrame>(last_frames_->getBundleId());
    this->svoFrameToKeyframe(last_frames_->at(0), cur_kf.get());
    cur_kf->keyframe_image_ = current_frame_image;

    kf_list_.push_back(cur_kf);

    double score_expected = 1;
    completed_flags_.push_back(false);
    this->threads_.push_back(
        thread(&LoopClosing::runPROnLatestKeyframe, this,
               static_cast<size_t>(options_.ignored_past_frames),
               run_lc_on_this_frame, score_expected));
    this->threads_.back().detach();
    last_added_frame_trackIDs_ = cur_kf->svo_trackIDsvector_;
    last_run_lc_frame_trackIDs_ = cur_kf->svo_trackIDsvector_;
  }
  else
  {
    if (completed_flags_.back() == true)
    {
      std::shared_ptr<KeyFrame> cur_kf =
          std::make_shared<KeyFrame>(last_frames_->getBundleId());
      this->svoFrameToKeyframe(last_frames_->at(0), cur_kf.get());
      cur_kf->keyframe_image_ = current_frame_image;
      run_lc_on_this_frame =
          commonLandMarkCheck(last_run_lc_frame_trackIDs_,
                              cur_kf->svo_trackIDsvector_, options_.beta);
      add_this_frame =
          commonLandMarkCheck(last_added_frame_trackIDs_,
                              cur_kf->svo_trackIDsvector_, options_.alpha);
      if (run_lc_on_this_frame)
      {
        last_run_lc_frame_trackIDs_ = cur_kf->svo_trackIDsvector_;
        last_added_frame_trackIDs_ = cur_kf->svo_trackIDsvector_;
        add_this_frame = true;
      }

      if (add_this_frame)
      {
        if (ignore_next_constraint_in_pg_ && pgo_)
        {
          pgo_->ignore_seq_constraint_kfs_.push_back(
              last_frames_->getBundleId());
          ignore_next_constraint_in_pg_ = false;
        }

        kf_list_.push_back(cur_kf);
        last_added_frame_trackIDs_ = cur_kf->svo_trackIDsvector_;

        double score_expected =
            compareBOWs(svokf_bow_vec_[svo_keyframe_count_ - 1],
                        svokf_bow_vec_[svo_keyframe_count_ - 2], voc_);
        cumulative_distance_ +=
            (kf_list_.back()->T_w_c_.getPosition() -
             kf_list_[kf_list_.size() - 2]->T_w_c_.getPosition())
                .norm();
        prox_dist_thresh_ =
            options_.proximity_dist_ratio * cumulative_distance_ +
            options_.proximity_offset;
        completed_flags_.push_back(false);
        this->threads_.pop_back();
        this->threads_.push_back(thread(&LoopClosing::runPROnLatestKeyframe,
                                        this,
                                        std::ref(options_.ignored_past_frames),
                                        run_lc_on_this_frame, score_expected));
        this->threads_.back().detach();
      }
    }
    else
    {
      VLOG(40) << "########## WARNING: Last thread still running ###########";
    }
  }
}

void LoopClosing::svoFrameToKeyframe(const FramePtr& frame, KeyFrame* kf) const
{
  CHECK_NOTNULL(kf);
  kf->clearSVOFeatureInfo();
  kf->frame_id_ = frame->id();
  this->extractAndConvert(frame, &kf->timestamp_sec_abs_, &kf->T_w_c_,
                          &kf->svo_keypointsvector_,
                          &kf->svo_landmarksvector_cam_, &kf->svo_landmark_ids_,
                          &kf->svo_depthsvector_, &kf->svo_featuretypevector_,
                          &kf->svo_trackIDsvector_, &kf->svo_bearingvectors_,
                          &kf->svo_original_indexvec_);
}

void LoopClosing::extractAndConvert(
    const svo::FramePtr& frame, double* current_frame_timestamp_sec,
    Transformation* Twc, std::vector<cv::Point2f>* current_frame_SVOkeypoints,
    std::vector<cv::Point3f>* current_frame_SVOlandmarks_in_cam,
    std::vector<int>* current_frame_SVOlandmark_ids,
    std::vector<double>* current_frame_SVOdepths,
    FeatureTypes* current_frame_SVOtypevec,
    std::vector<int>* current_frame_SVOtrackIDs,
    BearingVecs* current_frame_SVObearingvectors,
    std::vector<size_t>* current_frame_originalindices) const
{
  // data extraction from SVO frame and conversion to formats usable with open
  // cv functions
  (*Twc) = frame->T_world_cam();
  *current_frame_timestamp_sec = frame->getTimestampSec();
  for (std::size_t i = 0; i < frame->landmark_vec_.size(); ++i)
  {
    if (frame->landmark_vec_[i] == nullptr ||
        isFixedLandmark(frame->type_vec_[i]))
    {
      continue;
    }
    const PointPtr& pt = frame->landmark_vec_[i];
    current_frame_SVOkeypoints->push_back(
        cv::Point2f(frame->px_vec_(0, i), frame->px_vec_(1, i)));
    Eigen::Vector3d p_c = frame->T_f_w_.transform(pt->pos());
    current_frame_SVOlandmarks_in_cam->push_back(
        cv::Point3f(p_c(0, 0), p_c(1, 0), p_c(2, 0)));
    current_frame_SVOlandmark_ids->push_back(pt->id());
    current_frame_SVObearingvectors->push_back(frame->f_vec_.col(i));
    const double depth =
        frame->T_cam_world().transform(frame->landmark_vec_[i]->pos_)(2, 0);
    current_frame_SVOdepths->push_back(depth);
    current_frame_SVOtypevec->push_back(frame->type_vec_[i]);
    current_frame_originalindices->push_back(i);
    current_frame_SVOtrackIDs->push_back(int(frame->track_id_vec_[i]));
  }
}

void LoopClosing::updateKeyframe(const svo::FramePtr& frame)
{
  if (kf_list_.size() == 0)
  {
    return;
  }

  int res = findKfIndexByNFrameID(frame->bundleId());
  if (res != -1)
  {
    size_t kf_idx = static_cast<size_t>(res);
    const KeyFramePtr& found_kf = kf_list_[kf_idx];
    if (!completed_flags_[found_kf->lc_frame_count_ - 1])
    {
      return;
    }

    found_kf->skip_frame_ = !frame->is_stable_;

    // update the database with the latest optimized results
    this->svoFrameToKeyframe(frame, found_kf.get());
    updateSVOPointsDescriptors(kf_idx, true);
    found_kf->keyframe_image_.release();

    if (global_map_type_ == GlobalMapType::kBuiltInPoseGraph)
    {
      CHECK(pgo_);
      // we need to add the sequential constraint using uncorrected pose
      // also the map points need to be consistent with the pose
      Transformation corrected_T_w_c = found_kf->T_w_c_;
      Transformation uncorrected_T_w_c =
          frame->accumulated_w_T_correction_.inverse() * found_kf->T_w_c_;
      found_kf->T_w_c_ = uncorrected_T_w_c;
      pgo_->addPoseToPgoProblem(found_kf->T_w_c_, found_kf->NframeID_);
      /* Add Sequential Constraint to Pose Graph */
      if (kf_idx > 0)
      {
        int id_prev = kf_list_[kf_idx - 1]->NframeID_;
        Transformation t_be =
            kf_list_[kf_idx - 1]->T_w_c_.inverse() * uncorrected_T_w_c;
        pgo_->addSequentialConstraintToPgoProblem(t_be, id_prev,
                                                  found_kf->NframeID_);

        /* If this frame has a loop closure then add a loop constraint and
         * optimize the pose graph in a separate
         * thread.*/
        auto search = cur_kf_to_lc_kf_bundle_id_map_.find(found_kf->NframeID_);
        if (search != cur_kf_to_lc_kf_bundle_id_map_.end())
        {
          int current_frame_id = search->first;
          int lc_frame_id = search->second;
          int lc_kf_idx = findKfIndexByNFrameID(lc_frame_id);
          CHECK_GE(lc_kf_idx, -1);
          // now we need to add the corrrect relative pose
          Transformation t_be_lc =
              kf_list_[static_cast<size_t>(lc_kf_idx)]->T_w_c_.inverse() *
              corrected_T_w_c;
          if (!pgo_->has_updated_result_)
          {
            std::thread pgo_thread(&Pgo::addLoopConstraintToPgoProblem, pgo_,
                                   t_be_lc, lc_frame_id, current_frame_id);
            pgo_thread.detach();
            last_pgo_id_ = current_frame_id;
          }
        }
      }
    }  // pose graph optimization
  }

  if (pgo_ && pgo_->has_updated_result_)
  {
    vk::Timer t;
    t.start();
    updateDatabaseFromPG();
    refreshCeresPgoProblem();
    last_pgo_id_ = std::numeric_limits<int>::max();
    pgo_->has_updated_result_ = false;
    need_to_update_pose_graph_viz_ = true;
    double time = t.stop();
    VLOG(40) << "%%%%% Time taken for posegraph update " << time;
  }
}

void LoopClosing::updateDatabaseFromPG()
{
  for (int i = 0; i < static_cast<int>(kf_list_.size()); i++)
  {
    Transformation T_old = kf_list_[i]->T_w_c_;
    ceres::MapOfPoses::iterator pose_iter =
        pgo_->poses_->find(kf_list_[i]->NframeID_);
    if (pose_iter != pgo_->poses_->end())
    {
      kindr::minimal::Position p = pose_iter->second.p;
      Quaternion q = Quaternion(pose_iter->second.q);
      Transformation T_new = Transformation(q, p);
      kf_list_[i]->T_w_c_ = T_new;
    }
  }
}

void LoopClosing::undistortAndNormalise(
    const std::vector<cv::Point2f>& keypoints_cf,
    const std::vector<cv::Point2f>& keypoints_lc, const cv::Mat& K,
    const Eigen::VectorXd& dist_par,
    std::vector<cv::Point2f>* keypoints_matched_norm_udist_cf,
    std::vector<cv::Point2f>* keypoints_matched_norm_udist_lc)
{
  std::vector<double> D;
  D.resize(dist_par.size());
  Eigen::VectorXd::Map(&D[0], dist_par.size()) = dist_par;
  if (D.size() < 4)
  {
    D.resize(4);
  }
  cv::undistortPoints(keypoints_cf, *keypoints_matched_norm_udist_cf, K, D,
                      cv::noArray(), cv::noArray());
  cv::undistortPoints(keypoints_lc, *keypoints_matched_norm_udist_lc, K, D,
                      cv::noArray(), cv::noArray());
}

void LoopClosing::updateSVOPointsDescriptors(const size_t kf_index,
                                             const bool replace_mixed_features)
{
  const KeyFramePtr& cur_kf = kf_list_[kf_index];
  extractFeaturesFromSVOKeypoints(
      cur_kf->keyframe_image_, &cur_kf->svo_landmarksvector_cam_,
      &cur_kf->svo_landmark_ids_, &cur_kf->svo_trackIDsvector_,
      &cur_kf->svo_keypointsvector_, &cur_kf->svo_bearingvectors_,
      &cur_kf->svo_depthsvector_, &cur_kf->svo_featuretypevector_,
      &cur_kf->svo_original_indexvec_, &cur_kf->svo_features_,
      &cur_kf->svo_features_mat_);
  cur_kf->svo_node_ids_.clear();
  getNodeID(cur_kf->svo_features_, voc_, 3, &cur_kf->svo_node_ids_);

  if (replace_mixed_features)
  {
    cur_kf->mixed_keypoints_.clear();
    cur_kf->mixed_features_.clear();
    cur_kf->mixed_node_ids_.clear();
  }

  /// Mix the keypoints from bow and svo for second approach to scale retrieval
  cur_kf->mixed_keypoints_.insert(cur_kf->mixed_keypoints_.end(),
                                  cur_kf->bow_keypoints_.begin(),
                                  cur_kf->bow_keypoints_.end());
  cur_kf->mixed_keypoints_.insert(cur_kf->mixed_keypoints_.end(),
                                  cur_kf->svo_keypointsvector_.begin(),
                                  cur_kf->svo_keypointsvector_.end());

  /// Mix the features from bow and svo
  cur_kf->mixed_features_.insert(cur_kf->mixed_features_.end(),
                                 cur_kf->bow_features_.begin(),
                                 cur_kf->bow_features_.end());
  cur_kf->mixed_features_.insert(cur_kf->mixed_features_.end(),
                                 cur_kf->svo_features_.begin(),
                                 cur_kf->svo_features_.end());

  /// Mix the node ids from bow and svo. These are used for fast feature
  /// matching
  cur_kf->mixed_node_ids_.insert(cur_kf->mixed_node_ids_.end(),
                                 cur_kf->bow_node_ids_.begin(),
                                 cur_kf->bow_node_ids_.end());
  cur_kf->mixed_node_ids_.insert(cur_kf->mixed_node_ids_.end(),
                                 cur_kf->svo_node_ids_.begin(),
                                 cur_kf->svo_node_ids_.end());
}

void LoopClosing::refreshCeresPgoProblem()
{
  pgo_->purgeProblem();
  for (size_t i = 1; i < kf_list_.size(); i++)
  {
    int id_prev = kf_list_[i - 1]->NframeID_;
    int id_cur = kf_list_[i]->NframeID_;
    Transformation t_be =
        kf_list_[i - 1]->T_w_c_.inverse() * kf_list_[i]->T_w_c_;
    pgo_->addSequentialConstraintToPgoProblem(t_be, id_prev, id_cur);
    if (kf_list_[i]->NframeID_ == last_pgo_id_)
    {
      break;
    }
  }
}

void LoopClosing::updateMapPointsUsingDepth(
    svo::Frame& frame, std::vector<cv::Point3f>& svo_landmarksvector,
    const Transformation& pose, const BearingVecs& svo_bearingvector,
    const std::vector<double>& svo_depthvector,
    const FeatureTypes& svo_featuretypevector,
    const std::vector<size_t>& svo_originalindicesvec)
{
  LOG(FATAL) << "This should not be called with points"
                " represented now in the camera frame.";
  for (size_t i = 0; i < svo_landmarksvector.size(); i++)
  {
    if (svo_featuretypevector.at(i) == FeatureType::kMapPoint)
    {
      // Get the new point in world coordinates based on bearing vector and
      // depth
      double x, y, z;
      double multiplier = svo_depthvector.at(i) / svo_bearingvector.at(i)(2);
      x = multiplier * svo_bearingvector.at(i)(0);
      y = multiplier * svo_bearingvector.at(i)(1);
      z = svo_depthvector.at(i);
      Position point_cam = Position(x, y, z);
      Position point_world = pose.transform(point_cam);
      svo_landmarksvector.at(i) =
          cv::Point3f((float)(point_world(0, 0)), (float)(point_world(1, 0)),
                      (float)(point_world(2, 0)));
      frame.landmark_vec_[svo_originalindicesvec.at(i)]->pos_ = point_world;
    }
  }
}

void LoopClosing::updateKeyframePoses(const BundleIdToTwb& pose_map)
{
  size_t update_cnt = 0;
  for (const KeyFramePtr& kf : kf_list_)
  {
    auto it = pose_map.find(kf->NframeID_);
    if (it == pose_map.end())
    {
      continue;
    }
    kf->T_w_c_ = it->second * T_B_C_;
    update_cnt++;
  }
  if (update_cnt > 0)
  {
    need_to_update_pose_graph_viz_ = true;
  }
}

bool LoopClosing::tracePoseGraph(const std::string& path) const
{
  std::ofstream trace;
  trace.open(path);
  trace.precision(15);
  if (!trace)
  {
    return false;
  }
  else
  {
    for (int i = 0; i < static_cast<int>(kf_list_.size()); i++)
    {
      ceres::MapOfPoses::iterator pose_iter =
          pgo_->poses_->find(kf_list_[i]->NframeID_);
      if (pose_iter != pgo_->poses_->end())
      {
        kindr::minimal::Position p = pose_iter->second.p;
        Quaternion q = Quaternion(pose_iter->second.q);
        Transformation T_new = Transformation(q, p);
        Transformation pose_imu = T_new * T_C_B_;
        trace << kf_list_[i]->timestamp_sec_abs_ << " "
              << pose_imu.getPosition()(0, 0) << " "
              << pose_imu.getPosition()(1, 0) << " "
              << pose_imu.getPosition()(2, 0) << " "
              << pose_imu.getRotation().x() << " " << pose_imu.getRotation().y()
              << " " << pose_imu.getRotation().z() << " "
              << pose_imu.getRotation().w() << std::endl;
      }
    }
    return true;
  }
  trace.close();
}

bool LoopClosing::traceTimingData(const std::string& path) const
{
  std::ofstream trace_file;
  trace_file.open(path);
  trace_file.precision(10);
  if (!trace_file)
  {
    return false;
  }
  else
  {
    for (size_t i = 0; i < bow_timing_.size(); i++)
    {
      trace_file << bow_timing_[i] << " " << gv_timing_[i] << " "
                 << hm_timing_[i] << " " << transformmap_timing_[i]
                 << std::endl;
    }
  }
  trace_file.close();
  return true;
}

bool LoopClosing::traceNumQueryData(const std::string& path) const
{
  std::ofstream trace_file;
  trace_file.open(path);
  if (!trace_file)
  {
    return false;
  }
  else
  {
    for (size_t i = 0; i < num_queries_.size(); i++)
    {
      trace_file << num_queries_[i] << std::endl;
    }
  }
  trace_file.close();
  return true;
}

bool LoopClosing::traceClosedLoops(const string& trace_dir,
                                   const string& suffix) const
{
  for (const ClosedLoop& c : lc_closed_loops_)
  {
    std::ofstream file;
    std::stringstream path;
    path << trace_dir << "/" << suffix << "_"  << c.lc_id_ << "_" << c.cf_id_
         << ".txt";
    file.open(path.str());
    if (!file)
    {
      return false;
    }
    file << c.lc_t_sec_ << "\n"
         << c.cf_t_sec_ << "\n"
         << c.T_lc_cf_corrected_.getPosition() << "\n"
         << c.T_lc_cf_corrected_.getRotation() << std::endl;
    file.close();
  }

  return true;
}

bool LoopClosing::calculateTransformationInWorldFrame(
    const std::vector<cv::Point3f>& landmarks_lc,
    const std::vector<cv::Point3f>& landmarks_cf,
    const CorrespondIds& point_correspondences, const int& current_frame_id,
    const int& lc_frame_id, Transformation* w_T_new_old,
    std::vector<int>* inlier_indices)
{
  // (consider all current 3D points, there might be old ones as well)
  if (point_correspondences.size() < static_cast<size_t>(options_.min_num_3d))
  {
    VLOG(20) << "Not Enough 3D Points " << point_correspondences.size();
    //! @todo deal with 2d points somehow, maybe use together with fixed
    //! lc_frame
    return false;
  }

  map_alignment_se3_->reset();

  VLOG(0) << "Closing loop at frame " << current_frame_id;
  for (const IdCorrespondence& correspondence : point_correspondences)
  {
    const cv::Point3f& landmark_lc = landmarks_lc[correspondence.first];
    const cv::Point3f& landmark_cf = landmarks_cf[correspondence.second];

    Position landmark_lc_pos, landmark_cf_pos;
    landmark_lc_pos << landmark_lc.x, landmark_lc.y, landmark_lc.z;
    landmark_cf_pos << landmark_cf.x, landmark_cf.y, landmark_cf.z;

    map_alignment_se3_->addCorrespondencePair(landmark_cf_pos, landmark_lc_pos);
  }

  // use alignment to obtain good priors for all the parameterblocks
  bool transform_success = map_alignment_se3_->getTransformRansac(
        options_.min_num_3d, recovery_after_loss_, w_T_new_old,
        inlier_indices);
  // sanity check
  if (w_T_new_old->getPosition().norm() > 1000 ||
      std::isnan(w_T_new_old->getRotationMatrix()(1, 1)))
  {
    LOG(WARNING) << "Loop closing correction is obviously wrong, abort.";
    transform_success = false;
  }

  if (!transform_success)
  {
    LOG(WARNING) << "Current detection is not a valid loop closing";
  }
  else
  {
    cur_kf_to_lc_kf_bundle_id_map_[current_frame_id] = lc_frame_id;
    w_T_new_old->getRotation().normalize();
    MapAlignmentSE3::getClosest4DOFTransformInPlace(*w_T_new_old);
  }

  return transform_success;
}

}  // namespace svo
