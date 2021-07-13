#include <svo/tracker/feature_tracking_viz.h>
#include <svo/tracker/feature_tracker.h>
#include <svo/common/frame.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace svo {

void visualizeTracks(
    const FeatureTracker& tracker, size_t frame_index, int sleep)
{
  const FeatureTracks tracks = tracker.getActiveTracks(frame_index);
  if(tracks.empty())
  {
    VLOG(1) << "No features to visualize.";
    return;
  }
  VLOG(5) << "Tracker: Visualize " << tracks.size() << " tracks.";

  cv::Mat img_8u = tracks.at(0).back().getFrame()->img();
  cv::Mat img_rgb(img_8u.size(), CV_8UC3);
  cv::cvtColor(img_8u, img_rgb, cv::COLOR_GRAY2RGB);
  int frame_id = tracks.at(0).back().getFrame()->id();
  for(size_t i = 0; i < tracks.size(); ++i)
  {
    const FeatureTrack& track = tracks.at(i);
    CHECK_EQ(frame_id, track.back().getFrame()->id());
    cv::line(
          img_rgb,
          cv::Point2f(track.front().getPx()(0), track.front().getPx()(1)),
          cv::Point2f(track.back().getPx()(0), track.back().getPx()(1)),
          cv::Scalar(0,255,0), 2);
  }
  cv::imshow("tracking result", img_rgb);
  cv::waitKey(sleep);
}

} // namespace svo
