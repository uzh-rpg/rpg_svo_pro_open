#pragma once

#include <memory>

namespace vk {
namespace cameras {
class CameraGeometryBase;
class NCamera;
}
}

namespace svo
{
using Camera = vk::cameras::CameraGeometryBase;
using CameraPtr = std::shared_ptr<Camera>;
using CameraConstPtr = std::shared_ptr<const Camera>;
using CameraBundle = vk::cameras::NCamera;
using CameraBundlePtr = std::shared_ptr<CameraBundle>;
} // namespace svo
