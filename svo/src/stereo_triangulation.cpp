// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#include <numeric>
#include <svo/direct/matcher.h>
#include <svo/common/point.h>
#include <svo/common/frame.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/tracker/feature_tracker.h>

namespace svo {

StereoTriangulation::StereoTriangulation(
    const StereoTriangulationOptions& options,
    const AbstractDetector::Ptr& feature_detector)
  : options_(options)
  , feature_detector_(feature_detector)
{ ; }

void StereoTriangulation::compute(const FramePtr& frame0,
                                  const FramePtr& frame1)
{
  // Check if there is something to do
  if(frame0->numLandmarks() >= options_.triangulate_n_features)
  {
    VLOG(5) << "Calling stereo triangulation with sufficient number of features"
        << " has no effect.";
    return;
  }

  // Detect new features.
  Keypoints new_px;
  Levels new_levels;
  Scores new_scores;
  Gradients new_grads;
  FeatureTypes new_types;
  const size_t max_n_features = feature_detector_->grid_.size();
  feature_detector_->detect(
        frame0->img_pyr_, frame0->getMask(), max_n_features, new_px,
        new_scores, new_levels, new_grads, new_types);
  if(new_px.cols() == 0)
  {
    SVO_ERROR_STREAM("Stereo Triangulation: No features detected.");
    return;
  }

  // Compute and normalize all bearing vectors.
  Bearings new_f;
  frame_utils::computeNormalizedBearingVectors(new_px, *frame0->cam(), &new_f);

  // Add features to first frame.
  const long n_old = static_cast<long>(frame0->numFeatures());
  const long n_new = new_px.cols();
  frame0->resizeFeatureStorage(
        frame0->num_features_ + static_cast<size_t>(n_new));
  frame0->px_vec_.middleCols(n_old, n_new) = new_px;
  frame0->f_vec_.middleCols(n_old, n_new) = new_f;
  frame0->grad_vec_.middleCols(n_old, n_new) = new_grads;
  frame0->score_vec_.segment(n_old, n_new) = new_scores;
  frame0->level_vec_.segment(n_old, n_new) = new_levels;
  frame0->num_features_ += static_cast<size_t>(n_new);
  frame0->type_vec_.insert(
        frame0->type_vec_.begin()+n_old, new_types.cbegin(), new_types.cend());

  // We only want a limited number of features. Therefore, we create a random
  // vector of indices that we will process.
  std::vector<size_t> indices(static_cast<size_t>(n_new));
  std::iota(indices.begin(), indices.end(), n_old);
  long n_corners = std::count_if(
        new_types.begin(), new_types.end(),
        [](const FeatureType& t) { return t==FeatureType::kCorner; });

  // shuffle twice before we prefer corners!
  std::random_shuffle(indices.begin(), indices.begin()+n_corners);
  std::random_shuffle(indices.begin()+n_corners, indices.end());

  // now for all maximum corners, initialize a new seed
  size_t n_succeded = 0, n_failed = 0;
  const size_t n_desired =
      options_.triangulate_n_features - frame0->numLandmarks();
  //note: we checked already at start that n_desired will be larger than 0

  // reserve space for features in second frame
  if(frame1->num_features_ + n_desired > frame1->landmark_vec_.size())
  {
    frame1->resizeFeatureStorage(frame1->num_features_ + n_desired);
  }

  Matcher matcher;
  matcher.options_.max_epi_search_steps = 500;
  matcher.options_.subpix_refinement = true;
  const Transformation T_f1f0 = frame1->T_cam_body_*frame0->T_body_cam_;
  for(const size_t &i_ref : indices)
  {
    matcher.options_.align_1d = isEdgelet(frame0->type_vec_[i_ref]); // TODO(cfo): check effect
    FloatType depth = 0.0;
    FeatureWrapper ref_ftr = frame0->getFeatureWrapper(i_ref);
    Matcher::MatchResult res =
            matcher.findEpipolarMatchDirect(
                *frame0, *frame1, T_f1f0, ref_ftr, options_.mean_depth_inv,
                options_.min_depth_inv, options_.max_depth_inv, depth);

    if(res == Matcher::MatchResult::kSuccess)
    {
      const Position xyz_world = frame0->T_world_cam()
          * (frame0->f_vec_.col(static_cast<int>(i_ref)) * depth);
      PointPtr new_point(new Point(xyz_world));
      frame0->landmark_vec_[i_ref] = new_point;
      frame0->track_id_vec_(static_cast<int>(i_ref)) = new_point->id();
      new_point->addObservation(frame0, i_ref);

      const int i_cur = static_cast<int>(frame1->num_features_);
      frame1->type_vec_[static_cast<size_t>(i_cur)] = ref_ftr.type;
      frame1->level_vec_[i_cur] = ref_ftr.level;
      frame1->px_vec_.col(i_cur) = matcher.px_cur_;
      frame1->f_vec_.col(i_cur) = matcher.f_cur_;
      frame1->score_vec_[i_cur] = ref_ftr.score;
      GradientVector g = matcher.A_cur_ref_*ref_ftr.grad;
      frame1->grad_vec_.col(i_cur) = g.normalized();
      frame1->landmark_vec_[static_cast<size_t>(i_cur)] = new_point;
      frame1->track_id_vec_(i_cur) = new_point->id();
      new_point->addObservation(frame1, static_cast<size_t>(i_cur));
      frame1->num_features_++;
      ++n_succeded;
    }
    else
    {
      ++n_failed;
    }
    if(n_succeded >= n_desired)
      break;
  }
  VLOG(20) << "Stereo: Triangulated " << n_succeded << " features,"
           << n_failed << " failed.";
}

} // namespace svo

