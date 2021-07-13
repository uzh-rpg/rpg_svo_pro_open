// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <set>
#include <algorithm>
#include <svo/map.h>
#include <svo/common/point.h>
#include <svo/common/frame.h>

namespace svo {

Map::Map()
  : last_added_kf_id_(-1)
{}

Map::~Map()
{
  reset();
  SVO_INFO_STREAM("Map destructed");
}

void Map::reset()
{
  keyframes_.clear();
  sorted_keyframe_ids_.clear();
  last_added_kf_id_ = -1;
  points_to_delete_.clear();
}

void Map::removeKeyframe(const int frame_id)
{
  auto it_kf = keyframes_.find(frame_id);
  if (it_kf == keyframes_.end())
  {
    LOG(WARNING) << "Cannot find the keyframe with id " << frame_id
                 << ", will not do anything.";
    return;
  }
  const FramePtr& frame = it_kf->second;
  last_removed_kf_ = frame;
  for(size_t i = 0; i < frame->num_features_; ++i)
  {
    if(frame->landmark_vec_[i])
    {
      // observation is used in frontend to find reference frame, etc
      // this frame is not going to be used in the front end
      frame->landmark_vec_[i]->removeObservation(frame_id);
      // Since the map is the first to remove keyframe,
      //  the frame may still be used by other module
      // frame->landmark_vec_[i] = nullptr;
    }
  }
  keyframes_.erase(it_kf);
}

void Map::removeOldestKeyframe()
{
  removeKeyframe(sorted_keyframe_ids_.front());
  sorted_keyframe_ids_.pop_front();
}

void Map::safeDeletePoint(const PointPtr& pt)
{
  // Delete references to mappoints in all keyframes
  for(const KeypointIdentifier& obs : pt->obs_)
  {
    if(const FramePtr& frame = obs.frame.lock())
    {
      frame->deleteLandmark(obs.keypoint_index_);
    }
    else
      SVO_ERROR_STREAM("could not lock weak_ptr<Frame> in Map::safeDeletePoint");
  }
  pt->obs_.clear();
}

void Map::addPointToTrash(const PointPtr& pt)
{
  std::lock_guard<std::mutex> lock(points_to_delete_mutex_);
  points_to_delete_.push_back(pt);
}

void Map::emptyPointsTrash()
{
  std::lock_guard<std::mutex> lock(points_to_delete_mutex_);
  SVO_DEBUG_STREAM("Deleting " << points_to_delete_.size() << " point from trash");
  for(auto& pt : points_to_delete_)
  {
    safeDeletePoint(pt);
  }
  points_to_delete_.clear();
}

void Map::addKeyframe(const FramePtr& new_keyframe, bool temporal_map)
{
  VLOG(100) << "Adding keyframe to map. Frame-Id = " << new_keyframe->id();

  keyframes_.insert(std::make_pair(new_keyframe->id(), new_keyframe));
  last_added_kf_id_ = new_keyframe->id();
  if(temporal_map)
  {
    sorted_keyframe_ids_.push_back(new_keyframe->id());
  }
}

void Map::getOverlapKeyframes(
    const FramePtr& frame,
    std::vector<std::pair<FramePtr, double>>* close_kfs) const
{
  CHECK_NOTNULL(close_kfs);
  for(const auto& kf : keyframes_)
  {
    // check first if Point is visible in the Keyframe, use therefore KeyPoints
    for(const auto& keypoint : kf.second->key_pts_)
    {
      if(keypoint.first == -1)
        continue;

      if(frame->isVisible(keypoint.second))
      {
        close_kfs->push_back(
              std::make_pair(kf.second,
                             (frame->T_f_w_.getPosition()-kf.second->T_f_w_.getPosition()).norm()));
        break; // this keyframe has an overlapping field of view -> add to close_kfs
      }
    }
  }
}

void Map::getClosestNKeyframesWithOverlap(
    const FramePtr& cur_frame,
    const size_t num_frames,
    std::vector<FramePtr>* close_kfs) const
{
  CHECK_NOTNULL(close_kfs);
  std::vector<std::pair<FramePtr, double>> overlap_kfs;
  getOverlapKeyframes(cur_frame, &overlap_kfs);
  if(overlap_kfs.empty())
    return;

  size_t N = std::min(num_frames, overlap_kfs.size());
  // Extract closest N frames.
  std::nth_element(
        overlap_kfs.begin(), overlap_kfs.begin()+N, overlap_kfs.end(),
        [](const std::pair<FramePtr, double>& lhs,
           const std::pair<FramePtr, double>& rhs)
  { return lhs.second < rhs.second; });
  overlap_kfs.resize(N);

  close_kfs->reserve(num_frames);
  std::transform(overlap_kfs.begin(), overlap_kfs.end(), std::back_inserter(*close_kfs),
                 [](const std::pair<FramePtr, double>& p) { return p.first; });
}

FramePtr Map::getClosestKeyframe(const FramePtr& frame) const
{
  CHECK_NOTNULL(frame.get());
  std::vector<std::pair<FramePtr, double>> close_kfs;
  getOverlapKeyframes(frame, &close_kfs);
  if(close_kfs.empty())
    return nullptr;
  std::sort(
        close_kfs.begin(), close_kfs.end(),
        [](const std::pair<FramePtr, double>& lhs,
           const std::pair<FramePtr, double>& rhs)
  { return lhs.second < rhs.second; });
  if(close_kfs.at(0).first != frame)
    return close_kfs.at(0).first;
  if(close_kfs.size() == 1)
    return nullptr;
  return close_kfs.at(1).first;
}

FramePtr Map::getOldsestKeyframe() const
{
  return getKeyFrameAt(0);
}

FramePtr Map::getFurthestKeyframe(const Vector3d& pos) const
{
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(const auto& kf : keyframes_)
  {
    double dist = (kf.second->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = kf.second;
    }
  }
  return furthest_kf;
}

FramePtr Map::getKeyframeById(const int id) const
{
  auto it_kf = keyframes_.find(id);
  if(it_kf == keyframes_.end())
    return nullptr;
  return it_kf->second;
}

void Map::getSortedKeyframes(std::vector<FramePtr>& kfs_sorted) const
{
  kfs_sorted.reserve(keyframes_.size());
  std::transform(
        keyframes_.begin(), keyframes_.end(), std::back_inserter(kfs_sorted),
        [](const std::pair<int, FramePtr>& val)
  { return val.second;} );
  std::sort(
        kfs_sorted.begin(), kfs_sorted.end(),
        [](const FramePtr &left, const FramePtr &right)
  { return left->id_ < right->id_; });
}

void Map::transform(const Matrix3d& R, const Vector3d& t, const double& s)
{
  for(const auto& kf : keyframes_)
  {
    Vector3d pos = s*R*kf.second->pos() + t;
    Matrix3d rot = R*kf.second->T_f_w_.getRotation().inverse().getRotationMatrix();
    kf.second->T_f_w_ = Transformation(Quaternion(rot), pos).inverse();
    CHECK(false);
    /* TODO(cfo)
    for(const auto& ftr : kf.second->fts_)
    {
      if(ftr->point->last_published_ts_ == 0)
        continue;
      ftr->point->last_published_ts_ = 0;
      ftr->point->pos_ = s*R*ftr->point->pos_ + t;
    }
    */
  }
}

void Map::checkDataConsistency() const
{
  for(const auto& kf : keyframes_)
  {
    const FramePtr& frame = kf.second;

    // check that feature-stuff has all same length
    CHECK_EQ(frame->px_vec_.cols(), frame->f_vec_.cols());
    CHECK_EQ(frame->px_vec_.cols(), frame->level_vec_.size());
    CHECK_EQ(frame->px_vec_.cols(), frame->grad_vec_.cols());
    CHECK_EQ(static_cast<size_t>(frame->px_vec_.cols()), frame->type_vec_.size());
    CHECK_EQ(static_cast<size_t>(frame->px_vec_.cols()), frame->landmark_vec_.size());
    CHECK_EQ(static_cast<size_t>(frame->px_vec_.cols()), frame->seed_ref_vec_.size());
    CHECK_EQ(frame->px_vec_.cols(), frame->invmu_sigma2_a_b_vec_.cols());
    CHECK_LE(frame->num_features_, static_cast<size_t>(frame->px_vec_.cols()));

    // check features
    for(size_t i = 0; i < frame->num_features_; ++i)
    {
      if(frame->landmark_vec_[i])
      {
        // make sure that the 3d point has only one reference back
        size_t ref_count = 0;
        for(const KeypointIdentifier& o : frame->landmark_vec_[i]->obs_)
        {
          if(o.frame_id == frame->id())
          {
            ++ref_count;
            const FramePtr obs_frame = o.frame.lock();
            CHECK_EQ(obs_frame.get(), frame.get());
          }
        }
        CHECK_EQ(ref_count, 1u);
      }

      CHECK((isSeed(frame->type_vec_[i]) && frame->landmark_vec_[i] == nullptr)
            || !isSeed(frame->type_vec_[i]));
      CHECK(frame->seed_ref_vec_[i].keyframe == nullptr);
      CHECK(frame->seed_ref_vec_[i].seed_id == -1);
    }
  }
}

void Map::getLastRemovedKF(FramePtr* f)
{
  (*f) = last_removed_kf_;
  if (last_removed_kf_)
  {
    last_removed_kf_.reset();
  }
}

} // namespace svo
