#include "vikit/cameras/equidistant_fisheye_geometry.h"

namespace vk {
namespace cameras {

EquidistantFisheyeGeometry::EquidistantFisheyeGeometry(
      const int width, const int height, const double focal_length,
      const Eigen::Vector2d& principal_point, const double radius)
: EquidistantFisheyeGeometry(width, height, EquidistantFisheyeProjection(
    focal_length, principal_point), radius)
{}

EquidistantFisheyeGeometry::EquidistantFisheyeGeometry(
    const int width, const int height,
    const EquidistantFisheyeProjection& projection, const double radius)
: CameraGeometry<EquidistantFisheyeProjection>(width, height, projection)
{
  cv::Mat mask = cv::Mat(imageHeight(), imageWidth(), CV_8UC1);
  for(uint32_t i=0; i<imageHeight(); i++) {
    for(uint32_t j=0; j<imageWidth(); j++) {
      int64_t h_dist = static_cast<int64_t>(i) -
          static_cast<int64_t>(projection.principal_point()[1]);
      int64_t w_dist = static_cast<int64_t>(j) -
          static_cast<int64_t>(projection.principal_point()[0]);
      double dist_squared = std::pow(h_dist, 2) + std::pow(w_dist, 2);
      if (dist_squared > radius * radius)
        mask.at<uint8_t>(i, j) = 0;
      else
        mask.at<uint8_t>(i, j) = 1;
    }
  }
  setMask(mask);
}

}  // namespace cameras
}  // namespace vk
