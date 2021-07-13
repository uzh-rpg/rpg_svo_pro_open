// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#pragma once

#include <unordered_map>
#include <deque>
#include <mutex>
#include <svo/global.h>

namespace svo {

/// Map object which saves all keyframes which are in a map.
class Map
{
public:
  typedef std::shared_ptr<Map> Ptr;
  typedef std::unordered_map<int, FramePtr> Keyframes; // Frame-Id & Pointer

  Keyframes keyframes_;                   //!< List of keyframes in the map.
  std::vector<PointPtr> points_to_delete_;
  std::mutex points_to_delete_mutex_;
  int last_added_kf_id_;
  std::deque<int> sorted_keyframe_ids_; //!< Used with ceres backend

  // cache the last removed keyframe for use of other modules
  FramePtr last_removed_kf_ = nullptr;

  Map();
  ~Map();

  Map(const Map&) = delete;             // no copy
  Map& operator=(const Map&) = delete;  // no copy

  /// Reset the map. Delete all keyframes and reset the frame and point counters.
  void reset();

  /// Add a new keyframe to the map.
  void addKeyframe(const FramePtr& new_keyframe, bool temporal_map);


  /// Moves the frame to the trash queue which is cleaned now and then.
  void removeKeyframe(const int frame_id);

  /// Remove oldest keyframe (used with temporal map)
  void removeOldestKeyframe();

  /// Safely remove a point from a map
  void safeDeletePoint(const PointPtr& pt);

  /// Moves the point to the trash bin, cleared after all reprojectors have done their job
  void addPointToTrash(const PointPtr& pt);

  /// Effectively delete points in the trash bin
  void emptyPointsTrash();

  /// Given a frame, return all keyframes which have an overlapping field of view.
  void getOverlapKeyframes(
      const FramePtr& frame,
      std::vector<std::pair<FramePtr,double>>* close_kfs) const;

  /// Given a frame, return N closest keyframes with overlapping FoV.
  void getClosestNKeyframesWithOverlap(
      const FramePtr& cur_frame,
      const size_t num_frames,
      std::vector<FramePtr>* visible_kfs) const;

  /// Return the keyframe which is spatially closest and has overlapping field of view.
  FramePtr getClosestKeyframe(const FramePtr& frame) const;

  /// Return the keyframe which is oldest
  FramePtr getOldsestKeyframe() const;

  /// Return the keyframe which is furthest apart from pos.
  FramePtr getFurthestKeyframe(const Vector3d& pos) const;

  /// Get Keyframe by Frame-Id. Used for relocalizer.
  FramePtr getKeyframeById(const int id) const;

  /// We use a hashtable to store the keyframes. This function provides a sorted
  /// list of keyframes.
  void getSortedKeyframes(std::vector<FramePtr>& kfs_sorted) const;

  /// Transform the whole map with rotation R, translation t and scale s.
  void transform(const Matrix3d& R, const Vector3d& t, const double& s);

  /// Return the number of keyframes in the map
  inline size_t size() const { return keyframes_.size(); }

  /// Returns the last added keyframe (or a nullptr if none has been added)
  inline FramePtr getLastKeyframe() { return this->getKeyframeById(last_added_kf_id_); }

  void checkDataConsistency() const;

  inline FramePtr getKeyFrameAt(const size_t i)
  {
    return this->getKeyframeById(sorted_keyframe_ids_[i]);
  }

  inline size_t numKeyframes() const
  {
    return keyframes_.size();
  }

  inline const FramePtr getKeyFrameAt(const size_t i) const
  {
    return this->getKeyframeById(sorted_keyframe_ids_[i]);
  }

  void getLastRemovedKF(FramePtr* f);
};

} // namespace svo
