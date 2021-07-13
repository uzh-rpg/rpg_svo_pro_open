#pragma once

#include <utility>
#include <svo/common/types.h>
#include <svo/tracker/feature_tracking_types.h>

namespace svo {

namespace feature_tracking_utils {

/// pivot_ration needs to be in range(0,1) and if 0.5 it returns the median.
double getTracksDisparityPercentile(
    const FeatureTracks& tracks,
    double pivot_ratio);

/// Loops through TrackIds and checks if two frames have tracks in common.
void getFeatureMatches(
    const Frame& frame1, const Frame& frame2,
    std::vector<std::pair<size_t, size_t>>* matches_12);

} // namespace feature_tracking_utils
} // namespace svo
