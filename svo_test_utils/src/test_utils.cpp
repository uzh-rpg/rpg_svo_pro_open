#include <svo/test_utils/test_utils.h>

#include <numeric>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#ifdef SVO_USE_ROS
# include <ros/package.h>
# include <vikit/params_helper.h>
#endif

namespace svo {
namespace test_utils {

std::string getDatasetDir()
{
  const char* env_dir = std::getenv("SVO_DATASET_DIR");
#ifdef SVO_USE_ROS
  std::string dataset_dir(ros::package::getPath("svo")+"/test/data");
  if(env_dir != NULL)
    dataset_dir = std::string(env_dir);
  return dataset_dir;
#else
  return std::string(env_dir);
#endif
}

std::string getTestDataDir()
{
#ifdef SVO_USE_ROS
  return std::string(ros::package::getPath("svo_experiments")+"/data");
#else
  const char* env_dir = std::getenv("SVO_DATASET_DIR");
  return std::string(env_dir);
#endif
}

std::string getTraceDir()
{
#ifdef SVO_USE_ROS
  std::string default_dir(ros::package::getPath("svo_experiments")+"/results");
  return vk::getParam<std::string>("svo/trace_dir", default_dir);
#else
  return "/tmp";
#endif
}

VectorStats computeStats(std::vector<double>& v)
{
  VectorStats stats;
  const size_t size = v.size();
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / size;
  double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);

  std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
  stats.median = v[v.size()/2];

  std::nth_element(v.begin(), v.begin() + v.size()/10, v.end());
  stats.percentile10th = v[v.size()/10];

  std::nth_element(v.begin(), v.begin() + 9*v.size()/10, v.end());
  stats.percentile90th = v[9*v.size()/10];

  stats.mean = sum / v.size();
  stats.stdev = std::sqrt(sq_sum / v.size() - mean * mean);
  return stats;
}

FrameBundle::Ptr createFrameBundle(
    CameraPtr cam,
    const Transformation& T_w_f,
    const Transformation& T_f_b)
{
  FramePtr frame = std::make_shared<Frame>(0, 0, cam, T_w_f);
  frame->set_T_cam_imu(T_f_b);
  std::vector<FramePtr> frame_vec { frame };
  return std::make_shared<FrameBundle>(frame_vec);
}

Eigen::Vector3d generateRandomPoint(double max_depth, double min_depth)
{
  Eigen::Vector3d raw_point;
  // between -1 ~ +1
  raw_point[0] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)-0.5)*2;
  raw_point[1] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)-0.5)*2;
  raw_point[2] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)-0.5)*2;
  Eigen::Vector3d direction = raw_point / raw_point.norm();
  raw_point /= 1.733; // make sure the norm is below 1

  return min_depth*direction + (max_depth-min_depth)*raw_point;
}

void calcHist(const std::vector<double>& values, size_t bins, std::vector<size_t>* hist)
{
  CHECK(!values.empty() && bins >= 2);
  CHECK_NOTNULL(hist);
  hist->assign(bins, 0);
  double max = *std::max_element(values.begin(), values.end());
  double min = *std::min_element(values.begin(), values.end());
  if(max == min)
  {
    (*hist)[0] = values.size();
    return;
  }
  double interval = (max-min) / bins;

  for(auto ele : values)
  {
    size_t count = std::floor((ele-min)/interval);
    count = (count >= bins? bins-1:count); // max value
    (*hist)[count]++;
  }
}

// TODO(zzc): auto zoom for the text
cv::Mat drawHist(
    const std::vector<size_t>& hist,
    const std::vector<double>& bounds,
    int width, int height)
{
  CHECK(width>0 && height>0);
  CHECK(!hist.empty() && bounds.size() == hist.size()+1);

  size_t hist_bin_num = hist.size();
  cv::Mat histogram(height, width, CV_8UC1, cv::Scalar(255));
  size_t max_count = *std::max_element(hist.begin(), hist.end());
  double height_ratio = (height*1.0) / max_count;
  double bin_width = (width*1.0) / hist_bin_num;
  for(size_t i=0; i<hist_bin_num; i++)
  {
    std::stringstream ss;
    ss.precision(3);
    ss << 0.5 * (bounds[i]+bounds[i+1]);
    double bin_height = height_ratio * hist[i];
    double bin_bl = i*bin_width;
    cv::rectangle(
          histogram,
          cv::Point2f(bin_bl+0.1*bin_width, height),
          cv::Point2f(bin_bl+0.9*bin_width, height-bin_height),
          cv::Scalar(0), cv::FILLED);
    cv::putText(
          histogram,
          ss.str(),
          cv::Point2f(bin_bl+0.1*bin_width, height/2.0),
          cv::FONT_HERSHEY_PLAIN,
          1.2, cv::Scalar(120), 2
          );
    cv::putText(
          histogram,
          std::to_string(hist[i]),
          cv::Point2f(bin_bl+0.1*bin_width, height*0.8/2.0),
          cv::FONT_HERSHEY_PLAIN,
          1.2, cv::Scalar(120), 2
          );
  }
  return histogram;
}

} // namespace test_utils
} // namespace svo
