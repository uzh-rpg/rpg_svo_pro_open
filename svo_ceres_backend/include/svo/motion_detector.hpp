#include <svo/common/types.h>

namespace svo\
{

struct MotionDetectorOptions
{
  double px_diff_threshold = 0.5; ///< maximum allowed pixel distance [px]
  /// (!) ratio of allowed moving correspondences (moving_corresp/total_corresp)
  double ratio_moving_pixels_threshold = 0.1;
  int min_number_correspondences = 5; ///< min number of correspondendces
  int max_features_to_check = 100; ///< max number of correspondences to check
  double sigma = 0.05; ///< standard deviation for return motion prior
};

class MotionDetector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<MotionDetector> Ptr;

  MotionDetector(const MotionDetectorOptions &options): opt_(options){}
  ~MotionDetector(){}

  const MotionDetectorOptions opt_;

  /**
   * @brief Check if there was pixel motion between the set frames
   * @param[out] sigma proposed uncertainty for velocity prior
   * @return true if image is moving
   */
  bool isImageMoving(double &sigma) const;

  /**
   * @brief Set the frames to be considerd
   * @param[in] last_frames older frames
   * @param[in] new_frames newer frames
   */
  void setFrames(const FrameBundlePtr &last_frames,
                 const FrameBundlePtr &new_frames)
  {
    last_frames_= last_frames;
    new_frames_ = new_frames;
  }

  inline bool lastCheckValid() const
  {
    return last_check_valid_;
  }

private:
  FrameBundlePtr last_frames_;
  FrameBundlePtr new_frames_;

  /**
   * @brief Get ordered correspondences between last and new frame
   * @param[out] features_last ordered pixel positions in the last frames of
   *             correspondending features
   * @param[out] features_new ordered pixel positions in the new frames of
   *             correspondending features
   * @return number of correspondences between the frames
   */
  int findFeatureCorrespondences(Keypoints &features_last,
                                 Keypoints &features_new) const;

  /**
   * @brief Find pixel in last frame corresponding to feature in new frame
   * @param[in] frame frame to search for the correspondence
   * @param[in] track_id track id of the feature
   * @param[out] last_px pixel position in last frame
   * @return true if correspondence is found
   */
  bool findFeatureCorrespondence(const size_t cam_index,
                                 const int track_id,
                                 Keypoint *last_px) const;
  mutable bool last_check_valid_ = false;
};
} //namespace svo
