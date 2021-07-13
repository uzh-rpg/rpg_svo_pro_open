#include "vikit/cameras/camera_geometry_base.h"

#include <string>
#include <utility>

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>

#include <vikit/cameras/yaml/camera-yaml-serialization.h>
#include <aslam/common/yaml-serialization.h>
#include <vikit/path_utils.h>

namespace vk {
namespace cameras {

CameraGeometryBase::CameraGeometryBase(const int width, const int height)
  : width_(width)
  , height_(height)
{}

CameraGeometryBase::Ptr CameraGeometryBase::loadFromYaml(
    const std::string& yaml_file)
{
  try {
    YAML::Node doc = YAML::LoadFile(yaml_file.c_str());
    CameraGeometryBase::Ptr cam = doc.as<CameraGeometryBase::Ptr>();

    std::string basename = vk::path_utils::getBaseName(yaml_file);
    if(basename.empty())
    {
      return cam;
    }
    const YAML::Node& mask = doc["mask"];
    if(mask)
    {
      cam->loadMask(basename + "/" + mask.as<std::string>());
    }
    return cam;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Failed to load Camera from file " << yaml_file << " with the error: \n"
               << ex.what();
  }
  // Return nullptr in the failure case.
  return CameraGeometryBase::Ptr();
}

void CameraGeometryBase::backProject3(
      const Eigen::Ref<const Eigen::Matrix2Xd>& keypoints,
      Eigen::Matrix3Xd* out_bearing_vectors, std::vector<bool>* success) const
{
  const int num_keypoints = keypoints.cols();
  CHECK_NOTNULL(out_bearing_vectors)->resize(Eigen::NoChange, num_keypoints);
  CHECK_NOTNULL(success)->resize(num_keypoints);

  for (int i = 0; i < num_keypoints; ++i) {
    Eigen::Vector3d bearing_vector;
    (*success)[i] = backProject3(keypoints.col(i), &bearing_vector);
    out_bearing_vectors->col(i) = bearing_vector;
  }
}

void CameraGeometryBase::setMask(const cv::Mat& mask) {
  CHECK_EQ(height_, mask.rows);
  CHECK_EQ(width_,  mask.cols);
  CHECK_EQ(mask.type(), CV_8UC1);
  mask_ = mask;
}

void CameraGeometryBase::loadMask(const std::string& mask_file)
{
  cv::Mat mask(cv::imread(mask_file, 0));
  if(mask.data)
    setMask(mask);
  else
    LOG(FATAL) << "Unable to load mask file.";
}

bool CameraGeometryBase::isMasked(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint) const
{
  return keypoint[0] < 0.0 ||
         keypoint[0] >= static_cast<double>(width_) ||
         keypoint[1] < 0.0 ||
         keypoint[1] >= static_cast<double>(height_) ||
         (!mask_.empty() &&
           mask_.at<uint8_t>(static_cast<int>(keypoint[1]),
                             static_cast<int>(keypoint[0])) == 0);
}

Eigen::Vector2d CameraGeometryBase::createRandomKeypoint() const
{
  Eigen::Vector2d out;
  do
  {
    out.setRandom();
    out(0) = std::abs(out(0)) * imageWidth();
    out(1) = std::abs(out(1)) * imageHeight();
  }
  while (isMasked(out));

  return out;
}

} // namespace cameras
} // namespace vk
