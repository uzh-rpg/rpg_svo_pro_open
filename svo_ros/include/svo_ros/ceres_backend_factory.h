#pragma once

#include <memory>
#include <ros/ros.h>
#include <svo/ceres_backend_interface.hpp>

namespace svo {

namespace ceres_backend_factory {

std::shared_ptr<CeresBackendInterface> makeBackend(
    const ros::NodeHandle& pnh, const CameraBundlePtr& camera_bundle);

} // namespace ceres_backend_factory
} // namespace svo
