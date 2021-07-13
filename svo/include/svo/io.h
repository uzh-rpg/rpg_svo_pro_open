#pragma once

#include <string>
#include <svo/map.h>

namespace svo {
namespace io {

bool saveMap(
    const Map::Ptr& map,
    const std::string& save_dir,
    const std::string& map_name);

bool loadMap(
    const std::string& save_dir,
    Map::Ptr& map);

} // namespace io
} // namespace svo
