// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/depth_filter.h>

#include <algorithm>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <svo/common/camera.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/logging.h>
#include <svo/common/seed.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>


namespace svo {

DepthFilter::DepthFilter(
    const DepthFilterOptions& options,
    const DetectorOptions& detector_options,
    const CameraBundle::Ptr& cams)
  : DepthFilter(options)
{
  // TODO: make a detector for every camera!
  feature_detector_ =
      feature_detection_utils::makeDetector(detector_options, cams->getCameraShared(0));
  sec_feature_detector_.reset();
  if (options_.extra_map_points)
  {
    DetectorOptions sec_detector_options = detector_options;
    sec_detector_options.detector_type = DetectorType::kShiTomasi;
    sec_feature_detector_ =
        feature_detection_utils::makeDetector(
          sec_detector_options, cams->getCameraShared(0));
  }
}

DepthFilter::DepthFilter(
    const DepthFilterOptions& options)
  : options_(options)
  , matcher_(new Matcher())
{
  SVO_INFO_STREAM("DepthFilter: created.");
  matcher_->options_.scan_on_unit_sphere = options.scan_epi_unit_sphere;
  matcher_->options_.affine_est_offset_ = options.affine_est_offset;
  matcher_->options_.affine_est_gain_ = options.affine_est_gain;
  if(options_.use_threaded_depthfilter)
    startThread();
}

DepthFilter::~DepthFilter()
{
  stopThread();
  SVO_INFO_STREAM("DepthFilter: destructed.");
}

void DepthFilter::startThread()
{
  if(thread_)
  {
    SVO_ERROR_STREAM("DepthFilter: Thread already started!");
    return;
  }
  SVO_INFO_STREAM("DepthFilter: Start thread.");
  thread_.reset(new std::thread(&DepthFilter::updateSeedsLoop, this));
}

void DepthFilter::stopThread()
{
  SVO_DEBUG_STREAM("DepthFilter: stop thread invoked.");
  if(thread_ != nullptr)
  {
    SVO_DEBUG_STREAM("DepthFilter: interrupt and join thread... ");
    quit_thread_ = true;
    jobs_condvar_.notify_all();
    thread_->join();
    thread_.reset();
  }
}

void DepthFilter::addKeyframe(
    const FramePtr& frame,
    const double depth_mean,
    const double depth_min,
    const double depth_max)
{
  // allocate memory for new features.
  frame->resizeFeatureStorage(
      frame->num_features_ + feature_detector_->grid_.size() +
      (sec_feature_detector_ ?
           sec_feature_detector_->closeness_check_grid_.size() :
           0u));

  if(thread_ == nullptr)
  {
    ulock_t lock(feature_detector_mut_);
    depth_filter_utils::initializeSeeds(
          frame, feature_detector_, options_.max_n_seeds_per_frame,
          depth_min, depth_max, depth_mean);
    if (options_.extra_map_points)
    {
      for (size_t idx = 0; idx < frame->numFeatures(); idx++)
      {
        const FeatureType& type = frame->type_vec_[idx];
        if (!isMapPoint(type) && type != FeatureType::kOutlier)
        {
          sec_feature_detector_->closeness_check_grid_.fillWithKeypoints(
                frame->px_vec_.col(static_cast<int>(idx)));
        }
      }
      depth_filter_utils::initializeSeeds(
            frame, sec_feature_detector_,
            options_.max_n_seeds_per_frame + options_.max_map_seeds_per_frame,
            depth_min, depth_max, depth_mean);
    }
  }
  else
  {
    ulock_t lock(jobs_mut_);

    // clear all other jobs, this one has priority
    while(!jobs_.empty())
      jobs_.pop();
    jobs_.push(Job(frame, depth_min, depth_max, depth_mean));
    jobs_condvar_.notify_all();
  }
}

void DepthFilter::reset()
{
  ulock_t lock(jobs_mut_);
  while(!jobs_.empty())
    jobs_.pop();
  SVO_INFO_STREAM("DepthFilter: RESET.");
}

void DepthFilter::updateSeedsLoop()
{
  while(true)
  {
    // wait for new jobs
    Job job;
    {
      ulock_t lock(jobs_mut_);
      while(jobs_.empty() && !quit_thread_)
        jobs_condvar_.wait(lock);

      if(quit_thread_)
        return;

      job = jobs_.front();
      jobs_.pop();
    } // release lock

    // process jobs
    if(job.type == Job::SEED_INIT)
    {
      ulock_t lock(feature_detector_mut_);
      depth_filter_utils::initializeSeeds(
            job.cur_frame, feature_detector_,
            options_.max_n_seeds_per_frame,
            job.min_depth, job.max_depth, job.mean_depth);
      if (options_.extra_map_points)
      {
        for (size_t idx = 0; idx < job.cur_frame->numFeatures(); idx++)
        {
          const FeatureType& type = job.cur_frame->type_vec_[idx];
          if (!isMapPoint(type) && type != FeatureType::kOutlier)
          {
            sec_feature_detector_->closeness_check_grid_.fillWithKeypoints(
                  job.cur_frame->px_vec_.col(static_cast<int>(idx)));
          }
        }
        depth_filter_utils::initializeSeeds(
              job.cur_frame, sec_feature_detector_,
              options_.max_n_seeds_per_frame + options_.max_map_seeds_per_frame,
              job.min_depth, job.max_depth, job.mean_depth);
      }
    }
    else if(job.type == Job::UPDATE)
    {
      // We get higher precision (10x in the synthetic blender dataset)
      // when we keep updating seeds even though they are converged until
      // the frame handler selects a new keyframe.
      depth_filter_utils::updateSeed(
            *job.cur_frame, *job.ref_frame, job.ref_frame_seed_index, *matcher_,
            options_.seed_convergence_sigma2_thresh, true, false);
    }
  }
}

size_t DepthFilter::updateSeeds(
    const std::vector<FramePtr>& ref_frames_with_seeds,
    const FramePtr& cur_frame)
{
  size_t n_success = 0;
  if(thread_ == nullptr)
  {
    for(const FramePtr& ref_frame : ref_frames_with_seeds)
    {
      for(size_t i = 0; i < ref_frame->num_features_; ++i)
      {
        const FeatureType& type = ref_frame->type_vec_[i];
        if(isSeed(type))
        {
          double cur_thresh = options_.seed_convergence_sigma2_thresh;
          // we use a different threshold for map points to get better accuracy
          if (type == FeatureType::kMapPointSeed ||
              type == FeatureType::kMapPointSeedConverged)
          {
            cur_thresh = options_.mappoint_convergence_sigma2_thresh;
          }
          // We get higher precision (10x in the synthetic blender dataset)
          // when we keep updating seeds even though they are converged until
          // the frame handler selects a new keyframe.
          if(depth_filter_utils::updateSeed(
               *cur_frame, *ref_frame, i, *matcher_, cur_thresh, true, false))
          {
            ++n_success;
          }
        }
      }
    }
    SVO_DEBUG_STREAM("DepthFilter: " << cur_frame->cam()->getLabel() << " updated "
                     << n_success << " Seeds successfully.");
  }
  else
  {
    ulock_t lock(jobs_mut_);
    for(const FramePtr& ref_frame : ref_frames_with_seeds)
    {
      for(size_t i = 0; i < ref_frame->num_features_; ++i)
      {
        if(isSeed(ref_frame->type_vec_[i]))
        {
          jobs_.push(Job(cur_frame, ref_frame, i));
        }
      }
    }
    jobs_condvar_.notify_all();
  }
  return n_success;
}

namespace depth_filter_utils {

void initializeSeeds(
    const FramePtr& frame,
    const AbstractDetector::Ptr& feature_detector,
    const size_t max_n_seeds,
    const float depth_min,
    const float depth_max,
    const float depth_mean)
{
  // Detect new features.
  Keypoints new_px;
  Scores new_scores;
  Levels new_levels;
  Gradients new_grads;
  FeatureTypes new_types;
  Bearings new_f;

  //! @todo (MWE) FIXME - When we (ab)use the init seeds to initialize the first
  //! seeds the check px_vec_.cols()<(n_new+n_old) will fail but in case we don't
  //! have any old features we don't have concurrency issues and we just do it;
  //! still we should resolve this at some point and have a clean init / update procedure
  //!
  //! maybe the detector should take over more of these things as the detectors
  //! know what's going on -> but take care to be thread-safe
  //!
  bool no_features_in_frame = (frame->numFeatures() == 0);

  const int max_n_features = max_n_seeds - frame->numFeatures();
  if(max_n_features <= 0)
  {
    VLOG(3) << "Skip seed initialization. Have already enough features.";
    return;
  }
  if (no_features_in_frame)
  {
    ///TODO remove
    frame->clearFeatureStorage();
    CHECK_EQ(frame->px_vec_.size(), 0);

    feature_detector->detect(
          frame->img_pyr_, frame->getMask(), max_n_features, frame->px_vec_,
          frame->score_vec_, frame->level_vec_, frame->grad_vec_, frame->type_vec_);

    frame->num_features_ = frame->px_vec_.cols();
    frame->invmu_sigma2_a_b_vec_.resize(Eigen::NoChange, frame->numFeatures());
    frame->landmark_vec_.resize(frame->px_vec_.cols(), nullptr);
    frame->seed_ref_vec_.resize(frame->px_vec_.cols());

    // compute and normalize bearing vectors
    frame_utils::computeNormalizedBearingVectors(frame->px_vec_,
                                                 *frame->cam(), &frame->f_vec_);
    for(size_t i = 0; i<frame->num_features_ ; ++i)
    {
      if(frame->type_vec_[i] == FeatureType::kCorner)
        frame->type_vec_[i] = FeatureType::kCornerSeed;
      else if(frame->type_vec_[i] == FeatureType::kEdgelet)
        frame->type_vec_[i] = FeatureType::kEdgeletSeed;
      else if(frame->type_vec_[i] == FeatureType::kMapPoint)
        frame->type_vec_[i] = FeatureType::kMapPointSeed;
      else
        LOG(FATAL) << "Unknown feature types.";
    }
  }
  else
  {
    feature_detector->detect(
          frame->img_pyr_, frame->getMask(), max_n_features, new_px, new_scores,
          new_levels, new_grads, new_types);
    frame_utils::computeNormalizedBearingVectors(new_px, *frame->cam(), &new_f);
  }

  // Add features to frame.
  const size_t n_old = frame->num_features_;
  const size_t n_new = new_px.cols();

  CHECK_GE(frame->px_vec_.cols(), static_cast<int>(n_new+n_old));
  frame->px_vec_.middleCols(n_old, n_new) = new_px;
  frame->f_vec_.middleCols(n_old, n_new) = new_f;
  frame->grad_vec_.middleCols(n_old, n_new) = new_grads;
  frame->score_vec_.segment(n_old, n_new) = new_scores;
  frame->level_vec_.segment(n_old, n_new) = new_levels;
  for(size_t i = 0, j = n_old; i < n_new; ++i, ++j)
  {
    if(new_types[i] == FeatureType::kCorner)
      frame->type_vec_[j] = FeatureType::kCornerSeed;
    else if(new_types[i] == FeatureType::kEdgelet)
      frame->type_vec_[j] = FeatureType::kEdgeletSeed;
    else if(new_types[i] == FeatureType::kMapPoint)
      frame->type_vec_[j] = FeatureType::kMapPointSeed;
    else
      LOG(FATAL) << "Unknown feature types.";
  }
  frame->num_features_ = n_old+n_new;

  // initialize seeds
  frame->seed_mu_range_ = seed::getMeanRangeFromDepthMinMax(depth_min, depth_max);
  if (no_features_in_frame)
  {
    frame->invmu_sigma2_a_b_vec_.block(0, 0, 1, n_old).setConstant(seed::getMeanFromDepth(depth_mean));
    frame->invmu_sigma2_a_b_vec_.block(1, 0, 1, n_old).setConstant(seed::getInitSigma2FromMuRange(frame->seed_mu_range_));
    frame->invmu_sigma2_a_b_vec_.block(2, 0, 2, n_old).setConstant(10.0);
  }
  else
  {
    frame->invmu_sigma2_a_b_vec_.block(0, n_old, 1, n_new).setConstant(seed::getMeanFromDepth(depth_mean));
    frame->invmu_sigma2_a_b_vec_.block(1, n_old, 1, n_new).setConstant(seed::getInitSigma2FromMuRange(frame->seed_mu_range_));
    frame->invmu_sigma2_a_b_vec_.block(2, n_old, 2, n_new).setConstant(10.0);
  }

  SVO_DEBUG_STREAM("DepthFilter: "<< frame->cam()->getLabel() <<
                   " Initialized "<< n_new <<" new seeds");
}

bool updateSeed(
    const Frame& cur_frame,
    Frame& ref_frame,
    const size_t& seed_index,
    Matcher& matcher,
    const FloatType sigma2_convergence_threshold,
    const bool check_visibility,
    const bool check_convergence,
    const bool use_vogiatzis_update)
{
  if(cur_frame.id() == ref_frame.id())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "update seed with ref frame");
    return false;
  }

  constexpr double px_noise = 1.0;
  static double px_error_angle = cur_frame.getAngleError(px_noise);

  // check if seed is diverged
  const FeatureType type = ref_frame.type_vec_[seed_index];
  if(type == FeatureType::kOutlier)
  {
    return false;
  }

  // check if already converged
  if((type == FeatureType::kCornerSeedConverged ||
      type == FeatureType::kEdgeletSeedConverged ||
      type == FeatureType::kMapPointSeedConverged) && check_convergence)
  {
    return false;
  }

  // Create wrappers
  FeatureWrapper ref_ftr = ref_frame.getFeatureWrapper(seed_index);
  Eigen::Ref<SeedState> state = ref_frame.invmu_sigma2_a_b_vec_.col(seed_index);

  // check if point is visible in the current image
  Transformation T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse();
  if(check_visibility)
  {
    const Eigen::Vector3d xyz_f(T_cur_ref*(seed::getDepth(state) * ref_ftr.f) );
    Eigen::Vector2d px;
    if(!cur_frame.cam()->project3(xyz_f, &px).isKeypointVisible())
      return false;

    // check margin
    const Eigen::Vector2i pxi = px.cast<int>();
    const int boundary = 9;

    if(!cur_frame.cam()->isKeypointVisibleWithMargin(pxi, boundary))
      return false;
  }

  // set matcher options
  if(ref_ftr.type == FeatureType::kEdgeletSeed
     || ref_ftr.type == FeatureType::kEdgeletSeedConverged)
    matcher.options_.align_1d = true;
  else
    matcher.options_.align_1d = false;

  // sanity checks
  if(std::isnan(seed::mu(state)))
    SVO_ERROR_STREAM("seed is nan!");

  if(std::isnan(std::sqrt(seed::sigma2(state))))
    LOG(WARNING) << "seed sigma is nan!" << seed::sigma2(state) <<
                     ", sq" << std::sqrt(seed::sigma2(state)) <<
                     ", check-convergence = " << check_convergence;

  // search epipolar line, find match, and triangulate to find new depth z
  double depth;
  Matcher::MatchResult res =
      matcher.findEpipolarMatchDirect(
        ref_frame, cur_frame, T_cur_ref, ref_ftr, seed::getInvDepth(state),
        seed::getInvMinDepth(state), seed::getInvMaxDepth(state), depth);

  if(res != Matcher::MatchResult::kSuccess)
  {
    if(!matcher.reject_)
    {
      seed::increaseOutlierProbability(state);
    }
    if(matcher.options_.verbose)
    {
      std::cout << "filter fail = " << Matcher::getResultString(res) << std::endl;
    }
    return false;
  }

  // compute tau
  const FloatType depth_sigma = computeTau(T_cur_ref.inverse(), ref_ftr.f, depth, px_error_angle);

  // update the estimate
  if(use_vogiatzis_update)
  {
    if(!updateFilterVogiatzis(
         seed::getMeanFromDepth(depth),
         seed::getSigma2FromDepthSigma(depth, depth_sigma),
         ref_frame.seed_mu_range_,
         state))
    {
      ref_ftr.type = FeatureType::kOutlier;
      return false;
    }
  }
  else
  {
    if(!updateFilterGaussian(
         seed::getMeanFromDepth(depth),
         seed::getSigma2FromDepthSigma(depth, depth_sigma),
         state))
    {
      ref_ftr.type = FeatureType::kOutlier;
      return false;
    }
  }

  // check if converged
  if(seed::isConverged(state,
                       ref_frame.seed_mu_range_,
                       sigma2_convergence_threshold))
  {
    if(ref_ftr.type == FeatureType::kCornerSeed)
      ref_ftr.type = FeatureType::kCornerSeedConverged;
    else if(ref_ftr.type == FeatureType::kEdgeletSeed)
      ref_ftr.type = FeatureType::kEdgeletSeedConverged;
    else if(ref_ftr.type == FeatureType::kMapPointSeed)
      ref_ftr.type = FeatureType::kMapPointSeedConverged;
  }
  return true;
}

bool updateFilterVogiatzis(
    const FloatType z, // Measurement
    const FloatType tau2,
    const FloatType mu_range,
    Eigen::Ref<SeedState>& mu_sigma2_a_b)
{
  FloatType& mu = mu_sigma2_a_b(0);
  FloatType& sigma2 = mu_sigma2_a_b(1);
  FloatType& a = mu_sigma2_a_b(2);
  FloatType& b = mu_sigma2_a_b(3);

  const FloatType norm_scale = std::sqrt(sigma2 + tau2);
  if(std::isnan(norm_scale))
  {
    LOG(WARNING) << "Update Seed: Sigma2+Tau2 is NaN";
    return false;
  }

  const FloatType oldsigma2 = sigma2;
  const FloatType s2 = 1.0/(1.0/sigma2 + 1.0/tau2);
  const FloatType m = s2*(mu/sigma2 + z/tau2);
  const FloatType uniform_x = 1.0/mu_range;
  FloatType C1 = a/(a+b) * vk::normPdf<FloatType>(z, mu, norm_scale);
  FloatType C2 = b/(a+b) * uniform_x;
  const FloatType normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  const FloatType f = C1*(a+1.0)/(a+b+1.0) + C2*a/(a+b+1.0);
  const FloatType e = C1*(a+1.0)*(a+2.0)/((a+b+1.0)*(a+b+2.0))
                    + C2*a*(a+1.0)/((a+b+1.0)*(a+b+2.0));

  // update parameters
  const FloatType mu_new = C1*m+C2*mu;
  sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
  mu = mu_new;
  a = (e - f) / (f - e / f);
  b = a * (1.0 - f) / f;

  // TODO: This happens sometimes.
  if(sigma2 < 0.0)
  {
    LOG(WARNING) << "Seed sigma2 is negative!";
    sigma2 = oldsigma2;
  }
  if(mu < 0.0)
  {
    LOG(WARNING) << "Seed diverged! mu is negative!!";
    mu = 1.0;
    return false;
  }
  return true;
}

bool updateFilterGaussian(
    const FloatType z, // Measurement
    const FloatType tau2,
    Eigen::Ref<SeedState>& mu_sigma2_a_b)
{
  FloatType& mu = mu_sigma2_a_b(0);
  FloatType& sigma2 = mu_sigma2_a_b(1);
  FloatType& a = mu_sigma2_a_b(2);
  FloatType& b = mu_sigma2_a_b(3);

  const FloatType norm_scale = std::sqrt(sigma2 + tau2);
  if(std::isnan(norm_scale))
  {
    LOG(WARNING) << "Update Seed: Sigma2+Tau2 is NaN";
    return false;
  }

  const FloatType denom = (sigma2 + tau2);
  mu = (sigma2 * z + tau2 * mu) / denom;
  sigma2 = sigma2 * tau2 / denom;

  CHECK_GE(sigma2, 0.0);
  CHECK_GE(mu, 0.0);
  return true;
}

double computeTau(
      const Transformation& T_ref_cur,
      const BearingVector& f,
      const FloatType z,
      const FloatType px_error_angle)
{
  const BearingVector& t = T_ref_cur.getPosition();
  const BearingVector a = f*z-t;
  FloatType t_norm = t.norm();
  FloatType a_norm = a.norm();
  FloatType alpha = std::acos(f.dot(t)/t_norm); // dot product
  FloatType beta = std::acos(a.dot(-t)/(t_norm*a_norm)); // dot product
  FloatType beta_plus = beta + px_error_angle;
  FloatType gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
  FloatType z_plus = t_norm*std::sin(beta_plus)/std::sin(gamma_plus); // law of sines
  return (z_plus - z); // tau
}

double computeEpiGradAngle(
    const Transformation& T_cur_ref,
    const BearingVector& f_ref,
    const GradientVector& grad_ref,
    const FloatType depth_estimate)
{
  // compute epipolar line in current image
  const BearingVector& e_hom = T_cur_ref.getPosition();
  const BearingVector u_infty_hom(T_cur_ref.getRotation().rotate(f_ref));
  const BearingVector l(e_hom.cross(u_infty_hom)); // epipolar line in homogeneous coordinates

  const BearingVector f_ref_plus(f_ref + BearingVector(0.1*grad_ref[0], 0.1*grad_ref[1], f_ref[2]));

  const BearingVector f_cur = T_cur_ref*(f_ref*depth_estimate);
  const BearingVector f_cur_plus = T_cur_ref*(f_ref_plus*depth_estimate);

  GradientVector grad_cur(vk::project2(f_cur_plus)-vk::project2(f_cur));
  GradientVector l_dir(l[1],-l[0]);

  grad_cur.normalize();
  l_dir.normalize();

  return std::fabs(l_dir.dot(grad_cur));
}

#ifdef SVO_USE_PHOTOMETRIC_DISPARITY_ERROR
bool setSeedCovariance(
    const int halfpatch_size,
    const double image_noise2,
    SeedImplementation::Ptr seed)
{
  FramePtr frame = seed->ftr_->frame.lock();
  if(!frame)
  {
    SVO_ERROR_STREAM("Could not lock weak_ptr<Frame> in DepthFilter::setSeedCovariance");
    return false;
  }
  Eigen::Matrix2d H; H.setZero();
  const int patch_size = 2*halfpatch_size;
  const int L = seed->ftr_->level;
  const cv::Mat& img = frame->img_pyr_[L];
  const int step = img.step;
  const int u = seed->ftr_->px[0]/(1<<L);
  const int v = seed->ftr_->px[1]/(1<<L);

  if(u-halfpatch_size < 0 || u+halfpatch_size >= img.cols
      || v-halfpatch_size < 0 || v+halfpatch_size >= img.rows)
    return false;

  for(int y=0; y<patch_size; ++y)
  {
    uint8_t* p = img.data + (v-halfpatch_size+y)*step + (u-halfpatch_size);
    for(int x=0; x<patch_size; ++x, ++p)
    {
      const Eigen::Vector2d J((p[1] - p[-1]), (p[step] - p[-step]));
      H += J*J.transpose();
    }
  }
  H /= 260100.0*patch_size*patch_size; // 260100 = 2^2 * 255^2 for the missing 0.5 of the derivative and the 255 of uint8_t image
  seed->patch_cov_ = 2.0*image_noise2*H.inverse();

  if((bool) std::isnan((double)seed->patch_cov_(0,0)))
  {
    SVO_WARN_STREAM("Seed Patch Covariance is NaN");
      return false;
  }
  return true;
}

double getSeedDisparityUncertainty(
    const SeedImplementation::Ptr& seed,
    const Transformation& T_cur_ref)
{
  // compute epipolar line in current image
  // we use the essential matrix since we will only need the angle of the line
  // which should not be affected by the K-matrix. (distortion yes, but hopefully
  // neglectible).
  Eigen::Matrix3d E_cur_ref(vk::skew(T_cur_ref.getPosition())*T_cur_ref.getRotationMatrix());
  Eigen::Vector3d l_cur(E_cur_ref*seed->ftr_->f);

  // TODO apply rotation to l_cur to get l_ref

  // find uncertainty in direction of epipolar line, marginalize bivariate distribution
  double epi_angle_cur = atan(-l_cur[0]/l_cur[1]);
  double s = sin(-epi_angle_cur);
  double c = cos(-epi_angle_cur);
  double disp_sigma2 =
      seed->patch_cov_.determinant() / (seed->patch_cov_(1,1)*c*c + 2*seed->patch_cov_(0,1)*s*c + seed->patch_cov_(0,0)*s*s);
  return fmax(disp_sigma2, 1.0); // we don't want less than 1.0px uncertainty. because e.g. sampling errors
}
#endif

} // namespace depth_filter_utils
} // namespace svo
