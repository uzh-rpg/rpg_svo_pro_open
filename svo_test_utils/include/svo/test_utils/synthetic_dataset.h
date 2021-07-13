#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <vikit/blender_utils.h>
#include <svo/common/transformation.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/types.h>

namespace svo {
namespace test_utils {

/// Utility class to load synthetic datasets in tests.
class SyntheticDataset
{
public:

  SyntheticDataset(
      const std::string& dataset_dir,
      size_t cam_index,
      size_t first_frame_id,
      double sigma_img_noise = 0.0);

  ~SyntheticDataset() = default;

  const CameraPtr& cam() const { return cam_; }
  const CameraBundlePtr& ncam() const { return ncam_; }

  bool getNextFrame(
      size_t n_pyramid_levels,
      FramePtr& frame,
      cv::Mat* depthmap);

  bool skipNImages(size_t n);

private:

  void init();
  void skipFrames(size_t first_frame_id);

  // dataset dir
  std::string dataset_dir_;

  // which camera and frames to test
  size_t cam_index_;
  size_t first_frame_id_;

  // cameras
  CameraBundlePtr ncam_;
  CameraPtr cam_;

  // read image sequence
  std::ifstream img_fs_;
  std::ifstream gt_fs_;
  std::ifstream depth_fs_;

  // params
  double sigma_img_noise_;
};

} // namespace test_utils
} // namespace svo
