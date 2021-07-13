#pragma once

#include <string>
#include <iostream>

namespace vk {
namespace path_utils {

// usage: getFileDir(__FILE__)
inline std::string getBaseName(const std::string& filename)
{
  const std::string separator = "/";
  std::size_t last_separator = filename.find_last_of(separator);
  if(last_separator == std::string::npos)
  {
    return std::string();
  }
  return filename.substr(0, last_separator);
}

} // namespace path_utils
} // namespace vk
