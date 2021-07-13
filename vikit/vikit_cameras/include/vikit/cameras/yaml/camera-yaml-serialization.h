#pragma once

#include <glog/logging.h>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop

namespace vk {
namespace cameras {
class CameraGeometryBase;
}  // namespace cameras
}  // namespace vk

namespace YAML {

template<>
struct convert<std::shared_ptr<vk::cameras::CameraGeometryBase>> {
  /// This function will attempt to parse a camera from the yaml node.
  /// By default, yaml-cpp will throw and exception if the parsing fails.
  /// This function was written to *not* throw exceptions. Hence, decode always
  /// returns true, but when it fails, the shared pointer will be null.
  static bool decode(const Node& node, std::shared_ptr<vk::cameras::CameraGeometryBase>& camera);
  static Node encode(const std::shared_ptr<vk::cameras::CameraGeometryBase>& camera);
};

template<>
struct convert<vk::cameras::CameraGeometryBase> {
  static bool decode(const Node& node, vk::cameras::CameraGeometryBase& camera);
  static Node encode(const vk::cameras::CameraGeometryBase& camera);
};

}  // namespace YAML
