#include "svo/motion_detector.hpp"

#include <svo/common/frame.h>

namespace svo
{
bool MotionDetector::isImageMoving(double &sigma) const
{
  last_check_valid_ = false;
  if(!last_frames_)
  {
    return true;
  }

  //pre-check if there are enough features
  if(last_frames_->numFeatures() < size_t(opt_.min_number_correspondences)
     || new_frames_->numFeatures() < size_t(opt_.min_number_correspondences))
  {
    return true;
  }

  // find correspondences between last and new frame
  Keypoints features_last(2, opt_.max_features_to_check);
  Keypoints features_new(2, opt_.max_features_to_check);
  int num_correspondences = findFeatureCorrespondences(features_last,
                                                               features_new);

  if(num_correspondences < opt_.min_number_correspondences)
  {
    return true;
  }

  //find the number of moving pixels
  last_check_valid_ = true;
  const int n_moving_pixels_threshold =
      static_cast<int>(opt_.ratio_moving_pixels_threshold*num_correspondences);
  int n_moving_pixels = 0;
  for(int i = 0; i < num_correspondences; ++i)
  {
    //compute pixel distance
    if((features_last.col(i)-features_new.col(i)).norm()
       > opt_.px_diff_threshold)
    {
      ++n_moving_pixels;
      if(n_moving_pixels > n_moving_pixels_threshold)
      {
        return true;
      }
    }
  }
  //! @todo more sophisticated assignment of sigma
  sigma = opt_.sigma;
  return false;
}

int MotionDetector::findFeatureCorrespondences(Keypoints &features_last,
                                               Keypoints &features_new) const
{
  int num_correspondences = 0;
  Keypoint px;
  for(const FramePtr &frame : *new_frames_)
  {
    size_t cam_index = static_cast<size_t>(frame->getNFrameIndex());
    const int num_features = static_cast<int>(frame->numFeatures());
    for(int i = 0; i<num_features; ++i)
    {
      //check if track ID is assigned (seed or landmark)
      if(frame->track_id_vec_[i] == -1)
      {
        continue;
      }

      if(findFeatureCorrespondence(cam_index, frame->track_id_vec_[i], &px))
      {
        features_last.col(num_correspondences) = px;
        features_new.col(num_correspondences) = frame->px_vec_.col(i);
        ++num_correspondences;
        if(num_correspondences == opt_.max_features_to_check)
        {
          return num_correspondences;
        }
      }
    }
  }
  return num_correspondences;
}

bool MotionDetector::findFeatureCorrespondence(const size_t cam_index,
                                               const int track_id,
                                               Keypoint *last_px) const
{
  //find the matching track ID in last frames
  const int num_features =
      static_cast<int>(last_frames_->at(cam_index)->numFeatures());
  for(int i = 0; i<num_features; ++i)
  {
    if(last_frames_->at(cam_index)->track_id_vec_[i] == track_id)
    {
      *last_px = last_frames_->at(cam_index)->px_vec_.col(i);
      return true;
    }
  }
  return false;
}

} //namespace svo
