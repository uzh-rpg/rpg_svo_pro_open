#include <svo/tracker/feature_tracking_utils.h>

#include <svo/common/types.h>
#include <svo/common/frame.h>
#include <svo/tracker/feature_tracking_types.h>

namespace svo {
namespace feature_tracking_utils {

double getTracksDisparityPercentile(
    const FeatureTracks& tracks,
    double pivot_ratio)
{
  CHECK_GT(pivot_ratio, 0.0) << "pivot_ratio needs to be in (0,1)";
  CHECK_LT(pivot_ratio, 1.0) << "pivot_ratio needs to be in (0,1)";

  if(tracks.empty())
    return 0.0;

  // compute all disparities.
  std::vector<double> disparities;
  disparities.reserve(tracks.size());
  for(const FeatureTrack& track : tracks)
    disparities.push_back(track.getDisparity());

  // compute percentile.
  const size_t pivot = std::floor(pivot_ratio * disparities.size());
  CHECK_LT(pivot, disparities.size());
  std::nth_element(disparities.begin(), disparities.begin() + pivot, disparities.end(),
                   std::greater<double>());
  return disparities[pivot];
}

void getFeatureMatches(
    const Frame& frame1, const Frame& frame2,
    std::vector<std::pair<size_t, size_t>>* matches_12)
{
  CHECK_NOTNULL(matches_12);

  // Create lookup-table with track-ids from frame 1.
  std::unordered_map<int, size_t> trackid_slotid_map;
  for(size_t i = 0; i < frame1.num_features_; ++i)
  {
    int track_id_1 = frame1.track_id_vec_(i);
    if(track_id_1 >= 0)
      trackid_slotid_map[track_id_1] = i;
  }

  // Create list of matches.
  matches_12->reserve(frame2.num_features_);
  for(size_t i = 0; i < frame2.num_features_; ++i)
  {
    int track_id_2 = frame2.track_id_vec_(i);
    if(track_id_2 >= 0)
    {
      const auto it = trackid_slotid_map.find(track_id_2);
      if(it != trackid_slotid_map.end())
        matches_12->push_back(std::make_pair(it->second, i));
    }
  }
}

} // namespace feature_tracking_utils
} // namespace svo
