/*
 * Loop Closing Class Declaration: This includes the database, functions and
 * other variables associated with loop closure detection.
 */

/*
 * File:   run_place_recognition.h
 * Author: kunal71091
 *
 * Created on November 9, 2017
 */

#pragma once

#include <thread>
#include <mutex>
#include <memory>
#include <map>
#include <list>
#include <deque>
#include <fstream>

#include <vikit/timer.h>
#include <vikit/cameras/ncamera.h>
#include <vikit/cameras/camera_geometry_base.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/pgo/pgo.h>
#include <rpg_common/eigen_type.h>
#include <rpg_common/aligned.h>
#include <ros/ros.h>

#include "svo/online_loopclosing/bow.h"
#include "svo/online_loopclosing/geometric_verification.h"
#include "svo/online_loopclosing/read_file.h"
#include "svo/online_loopclosing/loop_closing_types.h"
#include "keyframe.h"

namespace svo
{
enum class LCScaleRetMethod
{
  kCommonLandmarks = 0,
  kMixedKeyPoints = 1,
  kNone = 2
};

enum class GlobalMapType
{
  kBuiltInPoseGraph = 0,
  kExternalGlobalMap = 1,
  kNone = 2
};

extern std::map<std::string, LCScaleRetMethod> kStrToScaleRetMap;
extern std::map<std::string, GlobalMapType> kStrToGlobalMapType;

class MapAlignmentSE3;

struct LoopClosureOptions
{
  /*Whether to run loop closure detection or not*/
  bool runlc;

  /*Name of the vocabulary to be used*/
  std::string voc_name;

  /*Path where the vocabulary is stored*/
  std::string voc_path;

  /*Data Extraction Threshold: Maximum common landmarks*/
  double alpha;

  /*Loop Closure detection Threshold: Maximum Common Landmarks*/
  double beta;

  /*Number of frames to ignore in immediate past*/
  int ignored_past_frames;

  /*Scale retrieval approach: 1 -> Common Landmarks Approach; 2 -> Mixed
   * Keypoints Approach*/
  std::string scale_ret_app;

  /*BOW threshold i.e minimum BOW score to qualify to Geometric Verification*/
  double bowthresh;

  /*Geometric verification threshold i.e Minimum percent inliers to qualify as
   * valid loop closure*/
  double gv_3d_inlier_thresh;

  /* 3D point threshold, minimum number of 3d points needed to be considered a
   * good loop closure*/
  int min_num_3d;

  /*Distance Threshold when matching orb features*/
  int orb_dist_thresh;

  /*Minimum fraction of matches in total number of features*/
  double gv_2d_match_thresh;

  /* Use opengv ransac for geometric verification */
  bool use_opengv;

  /* Enable this option to log images and inlier matches between loop closure
   * frames */
  bool enable_image_logging;

  /* Path to the folder where images will be locked. Make sure folder is already
   * created */
  std::string image_log_base_path;

  /* Constants for proximity check between current and query frames */
  double proximity_dist_ratio = 0.01;
  double proximity_offset = 0.3;

  // global map
  std::string global_map_type;

  // If the calculated correction is larger than this, we do explicit
  // correction anyway
  double force_correction_dist_thresh_meter = 0.1;
};

/* Forward Declaration */
class KeyFrame;

class LoopClosing
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /*default constructor*/
  LoopClosing(const LoopClosureOptions& loopclosure_options,
              const CameraBundle::Ptr& cams_);

  /*destructor*/
  virtual ~LoopClosing();

  /* Function addFrameToPR adds a frame to PR pipeline if it is a keyframe and
   * runs it */
  void addFrameToPR(const svo::FrameBundlePtr& last_frames_);

  /* Function runPR: Runs place recognition online along with SVO pipeline */
  void runPROnLatestKeyframe(const size_t ignored_frames,
                             const bool run_lc_on_this_frame,
                             const double score_expected);

  void svoFrameToKeyframe(const svo::FramePtr& frame, KeyFrame* kf) const;

  /* Function updatePoseGraph: given the pointer to a keyframe, this function
   * updates the landmark and
   * pose information if the keyframe with the given id exists in pose graph */
  void updateKeyframe(const svo::FramePtr& frame);

  inline int findKfIndexByNFrameID(const int nframe_id)
  {
    auto it = std::find_if(kf_list_.begin(), kf_list_.end(),
                           [&nframe_id](const KeyFramePtr& kf) {
                             return kf->NframeID_ == nframe_id;
                           });

    if (it == kf_list_.end())
    {
      return -1;
    }
    else
    {
      return static_cast<int>(it - kf_list_.begin());
    }
  }

  /* Update everything in the pose database with the result of latest pose graph
   * optimization */
  void updateDatabaseFromPG();

  /*
   * undistort and normalise keypoints using camera matrix
   */
  void undistortAndNormalise(
      const std::vector<cv::Point2f>& keypoints_cf,
      const std::vector<cv::Point2f>& keypoints_lc, const cv::Mat& K,
      const Eigen::VectorXd& dist_par,
      std::vector<cv::Point2f>* keypoints_matched_norm_udist_cf,
      std::vector<cv::Point2f>* keypoints_matched_norm_udist_lc);

  /*
   * This function effectively updates the keyframe database (given an index)
   * with a new set of SVO keypoints and landmarks
   * for a keyframe. This can be used when we first add a keyframe to the
   * database or when we update keyframe information
   * after a keyframe gets marginalised out of the sliding window.
   */
  void updateSVOPointsDescriptors(const size_t kf_index,
                                  const bool replace_mixed_features);

  /*
   * We keep a rolling PGO ceres problem. After every PGO the sequential
   * constraints should be updated.
   */
  void refreshCeresPgoProblem();

  /*
   * check whether two keyframes (from the database) are close enough to be
   * considered as loop candidates
   */
  inline bool proximityCheck(std::shared_ptr<KeyFrame> keyframe1,
                             std::shared_ptr<KeyFrame> keyframe2)
  {
    return (keyframe1->T_w_c_.getPosition() - keyframe2->T_w_c_.getPosition())
               .norm() <= prox_dist_thresh_;
  }

  /*
   * This function updates the map points in the database to make sure they are
   * consistent with the current pose
   */
  void updateMapPointsUsingDepth(
      svo::Frame& frame, std::vector<cv::Point3f>& svo_landmarksvector,
      const Transformation& pose, const BearingVecs& svo_bearingvector,
      const std::vector<double>& svo_depthvector,
      const FeatureTypes& svo_featuretypevector,
      const std::vector<size_t>& svo_originalindicesvec);

  /*
   * This function traces the pose graph to a file
   */
  bool tracePoseGraph(const std::string& path) const;

  /*
   * This function traces the timing vectors to a file
   */
  bool traceTimingData(const std::string& path) const;

  /*
   * This function traces the number of queries per frame to a file
   */
  bool traceNumQueryData(const std::string& path) const;

  bool traceClosedLoops(const std::string& trace_dir,
                        const std::string& suffix) const;

  // utilities
  inline int getNumKeyframes() const
  {
    return svo_keyframe_count_;
  }

  // get information for other modules, use with lock
  inline bool hasCorrectionInfo() const
  {
    return !lc_correction_info_.empty();
  }

  inline void consumeOldestCorrection(Transformation* w_T_correction)
  {
    (*w_T_correction) = lc_correction_info_.front().w_T_new_old_;
    lc_correction_info_.pop_front();
  }

  inline void consumePointMatchInfo(MatchedPointsInfo* match_point_info)
  {
    if (!lc_matched_points_info_.empty())
    {
      *match_point_info = lc_matched_points_info_.front();
      lc_matched_points_info_.pop_front();
    }
  }

  inline bool useExternalMap()
  {
    return global_map_type_ == GlobalMapType::kExternalGlobalMap;
  }

  void updateKeyframePoses(const BundleIdToTwb& pose_map);

  Transformation T_C_B_;
  Transformation T_B_C_;

  LoopClosureOptions options_;

  std::mutex lc_info_lock_;
  std::deque<LoopCorrectionInfo> lc_correction_info_;
  std::deque<MatchedPointsInfo> lc_matched_points_info_;
  std::vector<ClosedLoop> lc_closed_loops_;

  /* Hash map of valid loop closures. This is used to add constraints to pose
   * graph */
  std::map<int, int, std::less<int>,
           Eigen::aligned_allocator<std::pair<const int, int> > >
      cur_kf_to_lc_kf_bundle_id_map_;

  // actual list of keyframes in the database
  std::vector<std::shared_ptr<KeyFrame> > kf_list_;

  // pose graph related
  /* Pointer to pose graph optimization class*/
  std::shared_ptr<Pgo> pgo_;
  /* To save the frame id of the last frame that triggered PGO */
  int last_pgo_id_;

  // bookkeeping of frames that are added/run loop closing on
  // for common landmark check
  std::vector<int> last_added_frame_trackIDs_;
  std::vector<int> last_run_lc_frame_trackIDs_;

  /* flag to suspend loop closing after a loop correction. This is needed
   * because it takes some time
   * for the pose graph reoptimisation to run */
  bool suspend_lc_after_correction_ = false;
  int suspended_frames_counter_ = 0;

  /* Ignore sequential constraint between keyframes if there has been a long
   * occlusion */
  bool ignore_next_constraint_in_pg_ = false;

  /* Flag to relax thresholds on loop closing when we loose feature tracking for
   * long */
  bool recovery_after_loss_ = false;

  // compute valid transformation
  std::shared_ptr<MapAlignmentSE3> map_alignment_se3_;

  // visualization related
  /* This flag is used to rerun the visualisation when the whole pose graph is
   * updated */
  bool need_to_update_pose_graph_viz_ = false;
  /* loop closure info for visualiser. This just includes the locations of the
   * current and loop closure frames */
  LoopVizInfoVec cur_loop_check_viz_info_;
  LoopVizInfoVec loop_detect_viz_info_;
  LoopVizInfoVec loop_correction_viz_info_;

  // statistics and timing
  /* Vectors for recording loop detection and loop correction timings */
  std::vector<double> bow_timing_;
  std::vector<double> gv_timing_;
  std::vector<double> hm_timing_;
  std::vector<double> transformmap_timing_;
  std::vector<int> num_queries_;

  inline void addNewTimingSlot()
  {
    bow_timing_.push_back(0.0);
    gv_timing_.push_back(0.0);
    hm_timing_.push_back(0.0);
    transformmap_timing_.push_back(0.0);
  }

  inline void clearStats()
  {
    bow_timing_.clear();
    gv_timing_.clear();
    hm_timing_.clear();
    transformmap_timing_.clear();
  }

  inline void constructLoopViz(const KeyFrame& cur_kf, const KeyFrame& lc_kf,
                               LoopVizInfo* viz)
  {
    viz->block<1, 3>(0, 0) =
        cur_kf.T_w_c_.getPosition().transpose().cast<float>();
    viz->block<1, 3>(0, 3) =
        lc_kf.T_w_c_.getPosition().transpose().cast<float>();
  }

  inline void setRecoveryMode(const bool recover=true)
  {
    recovery_after_loss_ = recover;
    ignore_next_constraint_in_pg_ = recover;
  }

  inline bool lastFinished() const
  {
    return completed_flags_.back();
  }

private:
  void extractAndConvert(
      const svo::FramePtr& frame, double* current_frame_time_sec,
      Transformation* Twc, std::vector<cv::Point2f>* current_frame_SVOkeypoints,
      std::vector<cv::Point3f>* current_frame_SVOlandmarks_in_cam,
      std::vector<int>* current_frame_SVOlandmark_ids_,
      std::vector<double>* current_frame_SVOdepths,
      FeatureTypes* current_frame_SVOtypevec,
      std::vector<int>* current_frame_SVOtrackIDs,
      BearingVecs* current_frame_SVObearingvectors,
      std::vector<size_t>* current_frame_originalindices) const;

  bool calculateTransformationInWorldFrame(
      const std::vector<cv::Point3f>& landmarks_lc,
      const std::vector<cv::Point3f>& landmarks_cf,
      const CorrespondIds& point_correspondences, const int& current_frame_id,
      const int& lc_frame_id, Transformation* w_T_new_old,
      std::vector<int>* inlier_indices);

  /* This is used to obtain expected BOW score */
  std::vector<DBoW2::BowVector> svokf_bow_vec_;

  size_t lc_frame_count_ = 0u;
  size_t svo_keyframe_count_ = 0u;

  /* vector of threads*/
  std::vector<std::thread> threads_;

  /* DBoW2 Vocabulary */
  OrbVocabulary voc_;

  // camera related
  /* Intrinsic Camera Matrix */
  cv::Mat K_;
  /* Distortion Parameters */
  Eigen::VectorXd D_;

  /* Scale Retreival Approach 1 or 2 */
  LCScaleRetMethod scale_retrieval_approach_;

  /* Vector for thread management */
  std::vector<bool> completed_flags_;

  /* Proximity distance threshold for detecting close loop closure candidates */
  double prox_dist_thresh_ = 0;

  /* Cumulative distance until the last keyframe added to database */
  double cumulative_distance_ = 0;

  // global map method
  GlobalMapType global_map_type_;
};
}  // namespace svo
