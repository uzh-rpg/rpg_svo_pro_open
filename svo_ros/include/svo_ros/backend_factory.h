#pragma once

#include <memory>
#include <ros/ros.h>

namespace svo {

// forward declarations
class BackendInterface;

namespace backend_factory {

std::shared_ptr<BackendInterface> makeBackend(const ros::NodeHandle& pnh);

} // namespace vin_factory
} // namespace svo
