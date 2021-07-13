// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/initialization.h>

#include <random> // std::mt19937
#include <vector>

#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/camera.h>
#include <svo/common/container_helpers.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/direct/feature_alignment.h>
#include <svo/direct/matcher.h>
#include <svo/stereo_triangulation.h>
#include <svo/pose_optimizer.h>
#include <svo/tracker/feature_tracker.h>
#include <svo/tracker/feature_tracking_utils.h>
#include <vikit/cameras/ncamera.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>
#include <vikit/sample.h>
#include <opencv2/video/tracking.hpp> // for lucas kanade tracking
#include <opencv2/opencv.hpp> // for display

#ifdef SVO_USE_OPENGV
// used for opengv
# include <opengv/sac/Ransac.hpp>
// used for TwoPoint and FivePoint
# include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>
# include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
# include <opengv/relative_pose/methods.hpp>
# include <opengv/relative_pose/CentralRelativeAdapter.hpp>
# include <opengv/triangulation/methods.hpp>
// used for array
# include <opengv/relative_pose/NoncentralRelativeAdapter.hpp>
# include <opengv/sac_problems/relative_pose/NoncentralRelativePoseSacProblem.hpp>
#endif

#ifdef SVO_USE_GTSAM
# include <gtsam/geometry/SimpleCamera.h>
# include <gtsam/linear/NoiseModel.h>
# include <gtsam/nonlinear/Values.h>
# include <gtsam/nonlinear/NonlinearFactorGraph.h>
# include <gtsam/slam/PriorFactor.h>
# include <gtsam/slam/ProjectionFactor.h>
# include <gtsam/inference/Symbol.h>
# include <gtsam/geometry/Point2.h>
# include <gtsam/geometry/Pose3.h>
# include <gtsam/geometry/Point3.h>
# include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
# include <gtsam/slam/BetweenFactor.h>
#endif

namespace svo {

AbstractInitialization::AbstractInitialization(
    const InitializationOptions& init_options,
    const FeatureTrackerOptions& tracker_options,
    const DetectorOptions& detector_options,
    const CameraBundlePtr& cams)
  : options_(init_options)
{
  tracker_.reset(new FeatureTracker(tracker_options, detector_options, cams));
}

AbstractInitialization::~AbstractInitialization()
{}

void AbstractInitialization::reset()
{
  frames_ref_.reset();
  have_rotation_prior_ = false;
  have_translation_prior_ = false;
  have_depth_prior_ = false;
  tracker_->reset();
}

bool AbstractInitialization::trackFeaturesAndCheckDisparity(const FrameBundlePtr& frames)
{
  tracker_->trackFrameBundle(frames);
  std::vector<size_t> num_tracked;
  std::vector<double> disparity;
  tracker_->getNumTrackedAndDisparityPerFrame(
        options_.init_disparity_pivot_ratio, &num_tracked, &disparity);

  const size_t num_tracked_tot =
      std::accumulate(num_tracked.begin(), num_tracked.end(), 0u);
  const double avg_disparity =
      std::accumulate(disparity.begin(), disparity.end(), 0.0) / disparity.size();
  VLOG(3) << "Init: Tracked " << num_tracked_tot << " features with disparity = " << avg_disparity;
  if(num_tracked_tot < options_.init_min_features)
  {
    tracker_->resetActiveTracks();
    for(const FramePtr& frame : frames->frames_)
      frame->clearFeatureStorage();
    const size_t n = tracker_->initializeNewTracks(frames);
    VLOG(3) << "Init: New Tracks initialized = " << n;
    frames_ref_ = frames;
    R_ref_world_ = R_cur_world_;
    return false;
  }
  if(avg_disparity < options_.init_min_disparity)
    return false;
  return true;
}

InitResult HomographyInit::addFrameBundle(
    const FrameBundlePtr& frames_cur)
{
  // Track and detect features.
  if(!trackFeaturesAndCheckDisparity(frames_cur))
    return InitResult::kTracking;

  // Create vector of bearing vectors
  const FrameBundlePtr frames_ref = tracker_->getOldestFrameInTrack(0);
  const Frame& frame_ref = *frames_ref->at(0);
  const Frame& frame_cur = *frames_cur->at(0);
  FeatureMatches matches_cur_ref;
  feature_tracking_utils::getFeatureMatches(frame_cur, frame_ref, &matches_cur_ref);
  const size_t n = matches_cur_ref.size();
  Bearings f_cur(3, n);
  Bearings f_ref(3, n);
  for(size_t i = 0; i < n; ++i)
  {
    f_cur.col(i) = frame_cur.f_vec_.col(matches_cur_ref[i].first);
    f_ref.col(i) = frame_ref.f_vec_.col(matches_cur_ref[i].second);
  }

  // Compute model
  const vk::Homography H_cur_ref = vk::estimateHomography(
        f_cur, f_ref, frames_ref_->at(0)->getErrorMultiplier(),
        options_.reproj_error_thresh, options_.init_min_inliers);
  if(H_cur_ref.score < options_.init_min_inliers)
  {
    SVO_WARN_STREAM("Init Homography: Have " << H_cur_ref.score << "inliers. "
                    << options_.init_min_inliers << " inliers minimum required.");
    return InitResult::kFailure;
  }
  T_cur_from_ref_ = Transformation(Quaternion(H_cur_ref.R_cur_ref), H_cur_ref.t_cur_ref);

  // Triangulate
  if(initialization_utils::triangulateAndInitializePoints(
        frames_cur->at(0), frames_ref_->at(0), T_cur_from_ref_, options_.reproj_error_thresh,
        depth_at_current_frame_, options_.init_min_inliers, matches_cur_ref))
  {
    return InitResult::kSuccess;
  }

  // Restart
  tracker_->reset();
  frames_cur->at(0)->clearFeatureStorage();
  return InitResult::kTracking;
}


#ifdef SVO_USE_OPENGV

//! Same as in opengv, but stores the triangulated values
class TranslationSacProblemWithTriangulation : public opengv::sac_problems::relative_pose::TranslationOnlySacProblem
{
public:
   typedef opengv::sac_problems::relative_pose::TranslationOnlySacProblem Base;
   TranslationSacProblemWithTriangulation(adapter_t & adapter) : Base(adapter) {}

   void getSelectedDistancesToModel(
       const model_t & model,
       const std::vector<int> & indices,
       std::vector<double> & scores) const
   {
      using namespace opengv;

      translation_t translation = model.col(3);
      rotation_t rotation = model.block<3,3>(0,0);
      _adapter.sett12(translation);
      _adapter.setR12(rotation);

      model_t inverseSolution;
      inverseSolution.block<3,3>(0,0) = rotation.transpose();
      inverseSolution.col(3) = -inverseSolution.block<3,3>(0,0)*translation;

      Eigen::Matrix<double,4,1> p_hom;
      p_hom[3] = 1.0;

      points_.resize(indices.size());
      for( size_t i = 0; i < indices.size(); i++ )
      {
        p_hom.block<3,1>(0,0) =
            opengv::triangulation::triangulate2(_adapter,indices[i]);
        points_[i] = p_hom.block<3,1>(0,0);
        bearingVector_t reprojection1 = p_hom.block<3,1>(0,0);
        bearingVector_t reprojection2 = inverseSolution * p_hom;
        reprojection1 = reprojection1 / reprojection1.norm();
        reprojection2 = reprojection2 / reprojection2.norm();
        bearingVector_t f1 = _adapter.getBearingVector1(indices[i]);
        bearingVector_t f2 = _adapter.getBearingVector2(indices[i]);

        //bearing-vector based outlier criterium (select threshold accordingly):
        //1-(f1'*f2) = 1-cos(alpha) \in [0:2]
        double reprojError1 = 1.0 - (f1.transpose() * reprojection1);
        double reprojError2 = 1.0 - (f2.transpose() * reprojection2);
        scores.push_back(reprojError1 + reprojError2);
      }
   }

   mutable opengv::points_t points_;
};
#endif

InitResult TwoPointInit::addFrameBundle(
    const FrameBundlePtr& frames_cur)
{
#ifdef SVO_USE_OPENGV
  if(!have_rotation_prior_)
  {
    SVO_ERROR_STREAM("TwoPointInit has no rotation prior.");
    return InitResult::kFailure;
  }
  have_rotation_prior_ = false;

  // Track and detect features.
  if(!trackFeaturesAndCheckDisparity(frames_cur))
    return InitResult::kTracking;

  // Create vector of bearing vectors
  const FrameBundlePtr frames_ref = tracker_->getOldestFrameInTrack(0);
  FeatureMatches matches_cur_ref;
  feature_tracking_utils::getFeatureMatches(
        *frames_cur->at(0), *frames_ref->at(0), &matches_cur_ref);

  // Create vector of bearing vectors
  BearingVectors f_cur;
  BearingVectors f_ref;
  initialization_utils::copyBearingVectors(
        *frames_cur->at(0), *frames_ref->at(0), matches_cur_ref, &f_cur, &f_ref);

  // Compute model
  static double inlier_threshold =
      1.0 - std::cos(frames_cur->at(0)->getAngleError(options_.reproj_error_thresh));
  const Eigen::Matrix3d R_cur_ref = (R_cur_world_ * R_ref_world_.inverse()).getRotationMatrix();
  opengv::relative_pose::CentralRelativeAdapter adapter(f_cur, f_ref, R_cur_ref);
  typedef TranslationSacProblemWithTriangulation TwoPtProblem;
  std::shared_ptr<TwoPtProblem> problem_ptr(new TwoPtProblem(adapter));
  opengv::sac::Ransac<TwoPtProblem> ransac;
  ransac.sac_model_ = problem_ptr;
  ransac.threshold_ = inlier_threshold;
  ransac.max_iterations_ = 100;
  ransac.probability_ = 0.995;
  ransac.computeModel();

  // enough inliers?
  if(ransac.inliers_.size() < options_.init_min_inliers)
   return InitResult::kNoKeyframe;

  Eigen::Vector3d t = ransac.model_coefficients_.rightCols(1);
  Matrix3d R = ransac.model_coefficients_.leftCols(3);
  T_cur_from_ref_ = Transformation(
        Quaternion(R),
        ransac.model_coefficients_.rightCols(1));

  VLOG(5) << "2Pt RANSAC:" << std::endl
          << "# Iter = " << ransac.iterations_ << std::endl
          << "# Inliers = " << ransac.inliers_.size() << std::endl
          << "Model = " << ransac.model_coefficients_ << std::endl
          << "Rot prior (imu) = " << R_cur_ref << std::endl
          << "H.rotation_matrix() = " << T_cur_from_ref_.getRotationMatrix() << std::endl
          << "H.translation() = " << T_cur_from_ref_.getPosition();

  // Triangulate
  if(initialization_utils::triangulateAndInitializePoints(
        frames_cur->at(0), frames_ref->at(0), T_cur_from_ref_, options_.reproj_error_thresh,
        depth_at_current_frame_, options_.init_min_inliers, matches_cur_ref))
  {
    return InitResult::kSuccess;
  }
  return InitResult::kFailure;

#else
  SVO_ERROR_STREAM("You need to compile SVO with OpenGV to use TwoPointInit!");
  return InitResult::kFailure;
#endif
}


InitResult FivePointInit::addFrameBundle(
    const FrameBundlePtr& frames_cur)
{
#ifdef SVO_USE_OPENGV
  // Track and detect features.
  if(!trackFeaturesAndCheckDisparity(frames_cur))
    return InitResult::kTracking;

  // Create vector of bearing vectors
  const FrameBundlePtr frames_ref = tracker_->getOldestFrameInTrack(0);
  FeatureMatches matches_cur_ref;
  feature_tracking_utils::getFeatureMatches(
        *frames_cur->at(0), *frames_ref->at(0), &matches_cur_ref);

  // Create vector of bearing vectors
  BearingVectors f_cur;
  BearingVectors f_ref;
  initialization_utils::copyBearingVectors(
        *frames_cur->at(0), *frames_ref->at(0), matches_cur_ref, &f_cur, &f_ref);

  // Compute model
  static double inlier_threshold = 1.0 - std::cos(frames_cur->at(0)->getAngleError(options_.reproj_error_thresh));
  typedef opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem CentralRelative;
  opengv::relative_pose::CentralRelativeAdapter adapter(f_cur, f_ref);
  std::shared_ptr<CentralRelative> problem_ptr(
        new CentralRelative(adapter, CentralRelative::NISTER));
  opengv::sac::Ransac<CentralRelative> ransac;
  ransac.sac_model_ = problem_ptr;
  ransac.threshold_ = inlier_threshold;
  ransac.max_iterations_ = 100;
  ransac.probability_ = 0.995;
  ransac.computeModel();

  // enough inliers?
  if(ransac.inliers_.size() < options_.init_min_inliers)
  {
    VLOG(3) << "5Pt RANSAC has only " << ransac.inliers_.size() << " inliers. "
            << options_.init_min_inliers << " required.";
    return InitResult::kNoKeyframe;
  }

  Eigen::Vector3d t = ransac.model_coefficients_.rightCols(1);
  Matrix3d R = ransac.model_coefficients_.leftCols(3);
  T_cur_from_ref_ = Transformation(
        Quaternion(R),
        ransac.model_coefficients_.rightCols(1));

  VLOG(5) << "5Pt RANSAC:" << std::endl
          << "# Iter = " << ransac.iterations_ << std::endl
          << "# Inliers = " << ransac.inliers_.size() << std::endl
          << "Model = " << ransac.model_coefficients_ << std::endl
          << "T.rotation_matrix() = " << T_cur_from_ref_.getRotationMatrix() << std::endl
          << "T.translation() = " << T_cur_from_ref_.getPosition();

  // Triangulate
  if(initialization_utils::triangulateAndInitializePoints(
        frames_cur->at(0), frames_ref->at(0), T_cur_from_ref_, options_.reproj_error_thresh,
        depth_at_current_frame_, options_.init_min_inliers, matches_cur_ref))
  {
    return InitResult::kSuccess;
  }
  return InitResult::kFailure;

#else
  SVO_ERROR_STREAM("You need to compile SVO with OpenGV to use TwoPointInit!");
  return InitResult::kFailure;
#endif
}

InitResult OneShotInit::addFrameBundle(const FrameBundlePtr &frames_cur)
{
  CHECK(frames_cur->size() == 1) << "OneShot Initialization doesn't work with Camera Array";

  // Track and detect features.
  trackFeaturesAndCheckDisparity(frames_cur);

  if(frames_cur->numFeatures() < options_.init_min_features)
  {
    return InitResult::kTracking;
  }

  if(frames_ref_ == frames_cur)
  {
    // First frame
    return InitResult::kTracking;
  }

  // Initialize 3D points at known depth
  const FrameBundlePtr frames_ref = tracker_->getOldestFrameInTrack(0);
  FramePtr frame_ref = frames_ref->at(0);
  FramePtr frame_cur = frames_cur->at(0);
  FeatureMatches matches_cur_ref;
  feature_tracking_utils::getFeatureMatches(*frame_cur, *frame_ref, &matches_cur_ref);
  for(const std::pair<size_t, size_t> it : matches_cur_ref)
  {
    const BearingVector f_ref = frame_ref->f_vec_.col(it.second);
    const Vector3d xyz_in_cam = (f_ref/f_ref.z()) * depth_at_current_frame_;
    const Vector3d xyz_in_world = frame_ref->T_world_cam() * xyz_in_cam;
    PointPtr new_point(new Point(xyz_in_world));
    new_point->addObservation(frame_ref, it.second);
    frame_ref->landmark_vec_.at(it.second) = new_point;
    frame_cur->landmark_vec_.at(it.first)  = new_point;
  }

  // to estimate pose of current frame we do pose optimization
  PoseOptimizer::SolverOptions options = PoseOptimizer::getDefaultSolverOptions();
  options.verbose = true;
  PoseOptimizer optimizer(options);
  optimizer.run(frames_cur, 2.0);

  // Add all references to 3d point
  for(size_t i = 0; i < frame_cur->numFeatures(); ++i)
  {
    if(!frame_cur->landmark_vec_.at(i))
      continue;

    const PointPtr point = frame_cur->landmark_vec_.at(i);
    point->addObservation(frame_cur, i);
    // TODO: we should also remove references in previous frame.
  }

  if(frames_cur->numLandmarks() < options_.init_min_features)
  {
    std::cout << "tracking " << frames_ref->numLandmarks() << " features" << std::endl;
    return InitResult::kFailure;
  }
  return InitResult::kSuccess;
}

StereoInit::StereoInit(
    const InitializationOptions& init_options,
    const FeatureTrackerOptions& tracker_options,
    const DetectorOptions& detector_options,
    const CameraBundlePtr& cams)
  : AbstractInitialization(init_options, tracker_options, detector_options, cams)
{
  StereoTriangulationOptions stereo_options;
  detector_ = feature_detection_utils::makeDetector(detector_options, cams->getCameraShared(0));
  stereo_.reset(new StereoTriangulation(stereo_options, detector_));
}

InitResult StereoInit::addFrameBundle(
    const FrameBundlePtr& frames)
{
  CHECK_EQ(frames->size(), 2u) << "StereoInit: Bundle has not two frames!";
  reset();
  frames_ref_ = frames;

  VLOG(20) << "FRAME 1" << std::endl << frames->at(0)->T_world_cam() <<std::endl
           << "FRAME 2" << std::endl << frames->at(1)->T_cam_world();

  stereo_->compute(frames->at(0), frames->at(1));
  if(frames->at(0)->numLandmarks() < options_.init_min_features)
    return InitResult::kFailure;
  return InitResult::kSuccess;
}

InitResult ArrayInitGeometric::addFrameBundle(
    const FrameBundlePtr& frames_cur)
{

#ifdef SVO_USE_OPENGV
  /*
  InitResult res = trackBundleFeatures(frames_cur);
  if(res != InitResult::kSuccess)
    return res;

  // Create vector of bearing vectors and camera correspondences
  opengv::bearingVectors_t f_cur; f_cur.reserve(frames_cur->numFeatures());
  opengv::bearingVectors_t f_ref; f_ref.reserve(f_cur.size());
  std::vector<int> cam_correspondences_cur, cam_correspondences_ref;
  cam_correspondences_cur.reserve(f_cur.size());
  cam_correspondences_ref.reserve(f_ref.size());
  opengv::translations_t cam_translations;
  opengv::rotations_t cam_rotations;
  for(size_t cam_id=0; cam_id<frames_cur->size(); ++cam_id)
  {
    cam_translations.push_back(frames_cur->at(cam_id)->T_imu_cam().getPosition());
    cam_rotations.push_back(frames_cur->at(cam_id)->T_imu_cam().getRotationMatrix());
    const FramePtr& frame_cur = frames_cur->at(cam_id);
    const FramePtr& frame_ref = frames_ref_->at(cam_id);
    for(size_t i = 0; i < frame_cur->numFeatures(); ++i)
    {
      if(ref_masks_[cam_id](i))
      {
        f_cur.push_back(frame_cur->f_vec_.col(i));
        cam_correspondences_cur.push_back(cam_id);
        f_ref.push_back(frame_ref->f_vec_.col(i));
        cam_correspondences_ref.push_back(cam_id);
      }
    }
  }

  // set prior
  Matrix3d R_cur_ref =
      frames_cur->at(0)->T_imu_world().getRotationMatrix()
      *frames_ref_->at(0)->T_world_imu().getRotationMatrix();

  // Compute model
  opengv::relative_pose::NoncentralRelativeAdapter adapter(
        f_cur, f_ref, cam_correspondences_cur, cam_correspondences_ref,
        cam_translations, cam_rotations, R_cur_ref);

  // Create RANSAC object
  typedef opengv::sac_problems::relative_pose::NoncentralRelativePoseSacProblem Problem;
  opengv::sac::Ransac<Problem> ransac;

  // Create Problem
  auto algorithm = opengv::sac_problems::relative_pose::NoncentralRelativePoseSacProblem::SIXPT;
  boost::shared_ptr<Problem> problem_ptr(new Problem(adapter, algorithm));

  // Run RANSAC:
  static double inlier_threshold = 1.0 - std::cos(frames_cur->at(0)->getAngleError(options_.reproj_error_thresh));
  ransac.sac_model_ = problem_ptr;
  ransac.threshold_ = inlier_threshold;
  ransac.max_iterations_ = 500;
  ransac.probability_ = 0.995;
  ransac.computeModel(1);

  VLOG(5) << "RANSAC:" << std::endl
          << "# Iter = " << ransac.iterations_ << std::endl
          << "# Inliers = " << ransac.inliers_.size() << std::endl
          << "Model = " << ransac.model_coefficients_;

  // enough inliers?
  if(ransac.inliers_.size() < options_.init_min_inliers)
   return InitResult::kFailure;

  Matrix3d R = ransac.model_coefficients_.leftCols(3);
  T_cur_from_ref_ = Transformation(
        Quaternion(R),
        ransac.model_coefficients_.rightCols(1));


  std::cout << "T_cur_ref computed = " << T_cur_from_ref_ << std::endl;
  T_cur_from_ref_ = Transformation(Quaternion(), Vector3d(-0.6, 0, 0));
  std::cout << "T_cur_ref set = " << T_cur_from_ref_ << std::endl;

  // for every frame individually
  for(size_t cam_id=0; cam_id<frames_cur->size(); ++cam_id)
  {
    const FramePtr& frame_cur = frames_cur->at(cam_id);
    const FramePtr& frame_ref = frames_ref_->at(cam_id);

    frame_cur->T_f_w_ =
        frame_cur->T_cam_imu()*T_cur_from_ref_*frame_ref->T_imu_world();
    Transformation T_cur_ref = frame_cur->T_f_w_*frame_ref->T_f_w_.inverse();


    // Triangulate
    std::vector<Vector3d> points_in_cur;
    //TODO(cfo)
    //initialization_utils::triangulatePoints(
    //      frame_cur, frame_ref, T_cur_ref,  options_.reproj_error_thresh, points_in_cur);

    VLOG(3) << "-- CAMERA " << frame_cur->cam()->getLabel()
            << ": inliers verified = " << points_in_cur.size();

     TODO(cfo)
    // Rescale 3D points and add to features
    Transformation T_world_cur = frame_cur->T_f_w_.inverse();
    auto it_cur = frame_cur->fts_.begin();
    auto it_ref = frame_ref->fts_.begin();
    auto it_xyz = points_in_cur.begin();
    for(; it_cur != frame_cur->fts_.end(); ++it_cur, ++it_ref, ++it_xyz)
    {
      const Vector3d xyz_in_world = T_world_cur * (*it_xyz);
      PointPtr new_point(new Point(xyz_in_world));
      new_point->addObservation(frame_ref, *it_ref);
      new_point->addObservation(frame_cur, *it_cur);
      (*it_ref)->point = new_point;
      (*it_cur)->point = new_point;
    }
  }
  if(options_.verbose)
    std::cout << "Total triangulated points =  " << frames_cur->nFts() << std::endl;

  if(frames_cur->nFts() < options_.init_min_inliers)
  {
    SVO_WARN_STREAM("Init WARNING: "<<options_.init_min_inliers<<" inliers minimum required.");
    return InitResult::kFailure;
  }

  for(size_t i=0; i<frames_cur->size(); ++i)
  {
    initialization_utils::displayFeatureTracks(frames_cur->at(i), frames_ref_->at(i));

  }
  */



  return InitResult::kSuccess;

#else
  SVO_ERROR_STREAM("You need to compile SVO with OpenGV to use TwoPointInit!");
  return InitResult::kFailure;
#endif
}

InitResult ArrayInitOptimization::addFrameBundle(
    const FrameBundlePtr& frames_cur)
{
#ifdef SVO_USE_GTSAM

  InitResult res = trackBundleFeatures(frames_cur);
  if(res != InitResult::kSuccess)
    return res;

  gtsam::Cal3_S2::shared_ptr K(new gtsam::Cal3_S2(1.0, 1.0, 0.0, 0.0, 0.0));

  gtsam::noiseModel::Isotropic::shared_ptr pixel_noise_base =
      gtsam::noiseModel::Isotropic::Sigma(2, 1.0/frames_cur->at(0)->getErrorMultiplier());

  gtsam::SharedNoiseModel pixel_noise =
      gtsam::noiseModel::Robust::Create(
        //gtsam::noiseModel::mEstimator::Tukey::Create(4.685), pixel_noise_base);
        //gtsam::noiseModel::mEstimator::Fair::Create(1.3998), pixel_noise_base);
        //gtsam::noiseModel::mEstimator::Cauchy::Create(0.1), pixel_noise_base);
        //gtsam::noiseModel::mEstimator::Welsh::Create(2.9846), pixel_noise_base);
        gtsam::noiseModel::mEstimator::Huber::Create(1.345), pixel_noise_base);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial_estimate;

  // Add a prior on reference pose.
  Matrix<double, 6, 1> refpose_prior_sigmas;
  refpose_prior_sigmas << 0.0001, 0.0001, 0.0001, // rotation
                          0.00001, 0.00001, 0.00001; // translation, 1mm
  gtsam::noiseModel::Gaussian::shared_ptr pose_prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(refpose_prior_sigmas);
  Transformation T_world_refimu = frames_ref_->at(0)->T_world_imu();
  gtsam::Pose3 pose_ref(
        gtsam::Rot3(T_world_refimu.getRotationMatrix()),
        gtsam::Point3(T_world_refimu.getPosition()));
  graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(
                    gtsam::Symbol('x', 0), pose_ref, pose_prior_noise)); // add directly to graph

  Matrix3d eye = Matrix3d::Identity();
  gtsam::Pose3 T_prior(gtsam::Rot3(eye), Vector3d::Zero());
  std::cout << "Prior = " << T_prior << std::endl;
  Matrix<double, 6, 6> relpose_prior_info;
  relpose_prior_info.setZero();
  relpose_prior_info.block<3,3>(0,0) = Matrix3d::Identity()*1e8;
  gtsam::noiseModel::Gaussian::shared_ptr T_prior_noise =
      gtsam::noiseModel::Diagonal::Information(relpose_prior_info); // translation

  // Add prior on relative pose between first two frames
  graph.push_back(gtsam::BetweenFactor<gtsam::Pose3>(
                    gtsam::Symbol('x', 0), gtsam::Symbol('x', 1),
                    T_prior, T_prior_noise));

  // Add values for reference and current frame. init at same pose.
  initial_estimate.insert(gtsam::Symbol('x', 0), pose_ref);
  initial_estimate.insert(gtsam::Symbol('x', 1), pose_ref);

  /* TODO(cfo)
   *
  // Add values for 3D points. Initialize with random depth.
  int point_id = 0;
  for(const FramePtr& frame : frames_ref_->frames_)
  {
    for(const FeaturePtr& ftr : frame->fts_)
    {
      double depth = depth_at_current_frame_
          +vk::Sample::gaussian(depth_at_current_frame_/3.0);
      Vector3d xyz = frame->T_world_cam()*(ftr->f*depth);
      initial_estimate.insert(gtsam::Symbol('l', point_id++), gtsam::Point3(xyz));
    }
  }

  // Add reprojection factors.
  int point_id_ref = 0;
  typedef gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2> ProjectionFactor;
  for(const FramePtr& frame : frames_ref_->frames_)
  {
    gtsam::Pose3 T_imu_cam(frame->T_imu_cam().getRotationMatrix(),
                           frame->T_imu_cam().getPosition());
    for(const FeaturePtr& ftr : frame->fts_)
    {
      graph.push_back(
            ProjectionFactor(
              gtsam::Point2(vk::project2(ftr->f)), pixel_noise, gtsam::Symbol('x', 0),
              gtsam::Symbol('l', point_id_ref++), K, T_imu_cam));
    }
  }
  int point_id_cur = 0;
  for(const FramePtr& frame : frames_cur->frames_)
  {
    gtsam::Pose3 T_imu_cam(frame->T_imu_cam().getRotationMatrix(),
                           frame->T_imu_cam().getPosition());
    for(const FeaturePtr& ftr : frame->fts_)
    {
      graph.push_back(
            ProjectionFactor(
              gtsam::Point2(vk::project2(ftr->f)), pixel_noise, gtsam::Symbol('x', 1),
              gtsam::Symbol('l', point_id_cur++), K, T_imu_cam));
    }
  }
  SVO_ASSERT(point_id_cur == point_id_ref,
             "Initialization: Cur and Ref Bundles don't have same amount of features.");
  SVO_ASSERT(point_id_cur == point_id,
             "Initialization: Cur Bundle and Values don't have same amount of features.");

  // solve
  gtsam::LevenbergMarquardtParams params;
  params.verbosity = gtsam::LevenbergMarquardtParams::ERROR;
  gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial_estimate, params).optimize();
  //result.print("Final results:\n");
  std::cout << "initial error = " << graph.error(initial_estimate) << std::endl;
  std::cout << "final error = " << graph.error(result) << std::endl;

  gtsam::Pose3 T_ref = result.at<gtsam::Pose3>(gtsam::Symbol('x',0));
  gtsam::Pose3 T_cur = result.at<gtsam::Pose3>(gtsam::Symbol('x',1));

  T_world_refimu = Transformation(T_ref.rotation().toQuaternion(), T_ref.translation().vector());
  Transformation T_world_curimu(T_cur.rotation().toQuaternion(), T_cur.translation().vector());

  Transformation T_ref_cur = T_world_refimu.inverse()*T_world_curimu;
  std::cout << T_ref_cur.inverse() << std::endl;

  for(size_t i=0; i<frames_cur->size(); ++i)
  {
    initialization_utils::displayFeatureTracks(frames_cur->at(i), frames_ref_->at(i));
  }
  */
  return InitResult::kSuccess;

#else
  SVO_ERROR_STREAM("You need to compile SVO with GTSAM to use ArrayInitOptimization!");
  return InitResult::kFailure;
#endif
}


namespace initialization_utils {

bool triangulateAndInitializePoints(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref,
    const Transformation& T_cur_ref,
    const double reprojection_threshold,
    const double depth_at_current_frame,
    const size_t min_inliers_threshold,
    AbstractInitialization::FeatureMatches& matches_cur_ref)
{
  Positions points_in_cur;
  initialization_utils::triangulatePoints(
        *frame_cur, *frame_ref, T_cur_ref, reprojection_threshold, matches_cur_ref, points_in_cur);

  if(matches_cur_ref.size() < min_inliers_threshold)
  {
    LOG(WARNING) << "Init WARNING: " <<min_inliers_threshold <<" inliers minimum required. "
                 << "Have only " << matches_cur_ref.size();
    return false;
  }

  // Scale 3D points to given scene depth and initialize Points
  initialization_utils::rescaleAndInitializePoints(
        frame_cur, frame_ref, matches_cur_ref, points_in_cur, T_cur_ref, depth_at_current_frame);

  return true;
}

void triangulatePoints(
    const Frame& frame_cur,
    const Frame& frame_ref,
    const Transformation& T_cur_ref,
    const double reprojection_threshold,
    AbstractInitialization::FeatureMatches& matches_cur_ref,
    Positions& points_in_cur)
{
  points_in_cur.resize(Eigen::NoChange, frame_cur.num_features_);
  const Transformation T_ref_cur = T_cur_ref.inverse();
  std::vector<size_t> outliers;
  size_t num_inliers = 0;
  const double angle_threshold =
      frame_cur.getAngleError(reprojection_threshold);
  for(size_t i = 0; i < matches_cur_ref.size(); ++i)
  {
    BearingVector f_cur = frame_cur.f_vec_.col(matches_cur_ref[i].first);
    BearingVector f_ref = frame_ref.f_vec_.col(matches_cur_ref[i].second);
    // TODO(cfo): should take reference to eigen
    const Position xyz_in_cur =
        vk::triangulateFeatureNonLin(
          T_cur_ref.getRotationMatrix(), T_cur_ref.getPosition(), f_cur, f_ref);

    const double e1 = angleError(f_cur, xyz_in_cur);
    const double e2 = angleError(f_ref, T_ref_cur * xyz_in_cur);
    if(e1 > angle_threshold || e2 > angle_threshold ||
       (frame_cur.cam()->getType() == Camera::Type::kPinhole &&
           xyz_in_cur.z() < 0.0))
    {
      outliers.push_back(i);
    }
    else
    {
      points_in_cur.col(num_inliers) = xyz_in_cur;
      ++num_inliers;
    }
  }
  if(!outliers.empty())
  {
    svo::common::container_helpers::eraseIndicesFromVector(
        outliers, &matches_cur_ref);
  }
}

void rescaleAndInitializePoints(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref,
    const AbstractInitialization::FeatureMatches& matches_cur_ref,
    const Positions& points_in_cur,
    const Transformation& T_cur_ref,
    const double depth_at_current_frame)
{
  // compute scale factor
  std::vector<double> depth_vec;
  for(size_t i = 0; i < matches_cur_ref.size(); ++i)
  {
    depth_vec.push_back(points_in_cur.col(i).norm());
  }
  CHECK_GT(depth_vec.size(), 1u);
  const double scene_depth_median = vk::getMedian(depth_vec);
  const double scale = depth_at_current_frame / scene_depth_median;

  // reset pose of current frame to have right scale
  frame_cur->T_f_w_ = T_cur_ref * frame_ref->T_f_w_;
  frame_cur->T_f_w_.getPosition() =
      - frame_cur->T_f_w_.getRotation().rotate(
        frame_ref->pos() + scale * (frame_cur->pos() - frame_ref->pos()));

  // Rescale 3D points and add to features
  Transformation T_world_cur = frame_cur->T_f_w_.inverse();
  for(size_t i = 0; i < matches_cur_ref.size(); ++i)
  {
    const Vector3d xyz_in_world = T_world_cur * (points_in_cur.col(i) * scale);
    const int point_id_cur = frame_cur->track_id_vec_(matches_cur_ref[i].first);
    const int point_id_ref = frame_ref->track_id_vec_(matches_cur_ref[i].second);
    CHECK_EQ(point_id_cur, point_id_ref);
    PointPtr new_point(new Point(point_id_cur, xyz_in_world));
    frame_cur->landmark_vec_.at(matches_cur_ref[i].first) = new_point;
    frame_ref->landmark_vec_.at(matches_cur_ref[i].second) = new_point;
    new_point->addObservation(frame_ref, matches_cur_ref[i].second);
    new_point->addObservation(frame_cur, matches_cur_ref[i].first);
  }

  SVO_INFO_STREAM("Init: Triangulated " << matches_cur_ref.size() << " points");
}

void displayFeatureTracks(
    const FramePtr& frame_cur,
    const FramePtr& frame_ref)
{
  cv::Mat img_rgb(frame_cur->img().size(), CV_8UC3);
  cv::cvtColor(frame_cur->img(), img_rgb, cv::COLOR_GRAY2RGB);

  /* TODO(cfo)
  for(size_t i=0; i<frame_cur->fts_.size(); ++i)
  {
    const FeaturePtr& ftr_cur = frame_cur->fts_.at(i);
    const FeaturePtr& ftr_ref = frame_ref->fts_.at(i);
    cv::line(img_rgb,
             cv::Point2f(ftr_cur->px[0], ftr_cur->px[1]),
             cv::Point2f(ftr_ref->px[0], ftr_ref->px[1]),
             cv::Scalar(0,0,255), 2);

  }
  */
  cv::imshow(frame_cur->cam_->getLabel().c_str(), img_rgb);
  cv::waitKey(0);
}

AbstractInitialization::UniquePtr makeInitializer(
    const InitializationOptions& init_options,
    const FeatureTrackerOptions& tracker_options,
    const DetectorOptions& detector_options,
    const CameraBundle::Ptr& cams)
{
  AbstractInitialization::UniquePtr initializer;
  switch(init_options.init_type)
  {
    case InitializerType::kHomography:
      initializer.reset(new HomographyInit(init_options, tracker_options, detector_options, cams));
      break;
    case InitializerType::kTwoPoint:
      initializer.reset(new TwoPointInit(init_options, tracker_options, detector_options, cams));
      break;
    case InitializerType::kFivePoint:
      initializer.reset(new FivePointInit(init_options, tracker_options, detector_options, cams));
      break;
    case InitializerType::kOneShot:
      initializer.reset(new OneShotInit(init_options, tracker_options, detector_options, cams));
      break;
    case InitializerType::kStereo:
      initializer.reset(new StereoInit(init_options, tracker_options, detector_options, cams));
      break;
    case InitializerType::kArrayGeometric:
      initializer.reset(new ArrayInitGeometric(init_options, tracker_options, detector_options, cams));
      break;
    case InitializerType::kArrayOptimization:
      initializer.reset(new ArrayInitOptimization(init_options, tracker_options, detector_options, cams));
      break;
    default:
      LOG(FATAL) << "Initializer type not known.";
  }
  return initializer;
}

void copyBearingVectors(
    const Frame& frame_cur,
    const Frame& frame_ref,
    const AbstractInitialization::FeatureMatches& matches_cur_ref,
    AbstractInitialization::BearingVectors* f_cur,
    AbstractInitialization::BearingVectors* f_ref)
{
  CHECK_NOTNULL(f_cur);
  CHECK_NOTNULL(f_ref);
  f_cur->reserve(matches_cur_ref.size());
  f_ref->reserve(matches_cur_ref.size());
  for(size_t i = 0; i < matches_cur_ref.size(); ++i)
  {
    f_cur->push_back(frame_cur.f_vec_.col(matches_cur_ref[i].first));
    f_ref->push_back(frame_ref.f_vec_.col(matches_cur_ref[i].second));
  }
}

} // namespace initialization_utils

} // namespace svo
