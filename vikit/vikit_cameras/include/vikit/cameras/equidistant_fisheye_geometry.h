#pragma once

#include "vikit/cameras/camera_geometry.h"
#include "vikit/cameras/equidistant_fisheye_projection.h"

namespace vk {
namespace cameras {

// Equidistant fisheye geometry is strongly typed to enforce mask.
class EquidistantFisheyeGeometry :
    public CameraGeometry<EquidistantFisheyeProjection>
{
 public:
  EquidistantFisheyeGeometry(
      const int width, const int height, const double focal_length,
      const Eigen::Vector2d& principal_point, const double radius);

  EquidistantFisheyeGeometry(
      const int width, const int height,
      const EquidistantFisheyeProjection& projection, const double radius);

  virtual ~EquidistantFisheyeGeometry() = default;
};

}  // namespace cameras
}  // namespace vk
