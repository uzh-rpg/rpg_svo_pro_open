#pragma once

#include <svo/common/types.h>

namespace svo {

// Forward declarations
class FeatureTracker;

void visualizeTracks(
    const FeatureTracker& tracker, size_t frame_index, int sleep);

} // namespace svo
