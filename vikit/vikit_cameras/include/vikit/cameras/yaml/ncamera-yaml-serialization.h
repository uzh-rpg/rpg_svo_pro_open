#pragma once

#include <glog/logging.h>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <yaml-cpp/yaml.h>
#pragma diagnostic pop

namespace vk {
namespace cameras {
class NCamera;
}  // namespace cameras
}  // namespace vk

namespace YAML {

template<>
struct convert<std::shared_ptr<vk::cameras::NCamera>> {
  /// This function will attempt to parse an ncamera from the yaml node.
  /// By default, yaml-cpp will throw an exception if the parsing fails.
  /// This function was written to *not* throw exceptions. Hence, decode always
  /// returns true, but when it fails, the shared pointer will be null.
  static bool decode(const Node& node, std::shared_ptr<vk::cameras::NCamera>& ncamera);
  static Node encode(const std::shared_ptr<vk::cameras::NCamera>& ncamera);
};

template<>
struct convert<vk::cameras::NCamera> {
  static bool decode(const Node& node, vk::cameras::NCamera& ncamera);
  static Node encode(const vk::cameras::NCamera& ncamera);
};

}  // namespace YAML
