// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <cstdlib> // for getenv rand
#include <string>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <vikit/params_helper.h>
#include <svo/common/camera.h>

#include <svo/common/frame.h>


#define SVO_TEST_STREAM(x) {std::cerr<<"\033[0;0m[          ] * "<<x<<"\033[0;0m"<<std::endl; }

namespace svo {
namespace test_utils {
class Dataset
{
public:
  Dataset(const std::string& dataset_dir, size_t cam_index=0, size_t first_frame_id=0)
    : dataset_dir_(dataset_dir)
    , first_frame_id_(first_frame_id)
    , cam_index_(cam_index)
  { init(); }

  Camera::Ptr cam() { return cam_; }
  CameraBundle::Ptr ncam() { return ncam_; }

  svo::FramePtr getNextImage(size_t img_align_max_level, int user_id)
  {
    if(!img_fs_.good() || img_fs_.eof()) // end or bad
      return svo::FramePtr();

    // skip comment block
    while(img_fs_.peek() == '#')
      img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // load image
    size_t img_id;
    double stamp_seconds;
    std::string img_name;
    img_fs_ >> img_id >> stamp_seconds >> img_name;
    img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    CHECK(img_id >= first_frame_id_) << "image list is misordered.";
    uint64_t stamp = stamp_seconds*1e9;
    std::string img_filename(dataset_dir_ + "data/" + img_name);
    cv::Mat img(cv::imread(img_filename, 0));
    if(img.empty())
    {
      LOG(ERROR) << "fail to read image: " << img_filename;
      return svo::FramePtr();
    }

    // make frame
    svo::FramePtr frame = std::make_shared<svo::Frame>(cam_, img.clone(),
                                                       stamp, img_align_max_level+1,
                                                       user_id);
    frame->set_T_cam_imu(ncam_->get_T_C_B(cam_index_));

    return frame;
  }

private:
  void init()
  {
    // load camera
    ncam_ = CameraBundle::loadFromYaml(dataset_dir_+"calib.yaml");
    if(!ncam_)
      LOG(FATAL) << "fail to load camera.";
    cam_ = ncam_->getCameraShared(cam_index_);

    // initializing image file list
    std::string img_filename(dataset_dir_ + "data/images.txt");
    img_fs_.open(img_filename.c_str());
    if(!img_fs_.is_open())
      LOG(FATAL) << "fail to load image list.";

    skipFrames(first_frame_id_);
  }

  void skipFrames(size_t first_frame_id)
  {
    while(img_fs_.good() && !img_fs_.eof())
    {
      // skip comment block
      while(img_fs_.peek() == '#')
        img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

      if(first_frame_id == 0) return;

      size_t img_id;
      double stamp_seconds;
      std::string img_name;
      img_fs_ >> img_id >> stamp_seconds >> img_name;
      img_fs_.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      if(img_id < first_frame_id-1)
        continue;
      else
        return;
    }
    LOG(FATAL) << "image list corrupted or first_frame_id exceeds rang listed.";
  }

  // dataset dir
  std::string dataset_dir_;
  // which camera and frames to test
  size_t first_frame_id_;
  size_t cam_index_;
  // cameras
  CameraBundle::Ptr ncam_;
  Camera::Ptr cam_;
  // read image sequence
  std::ifstream img_fs_;
};

inline std::string getDatasetDir()
{
  const char* env_dir = std::getenv("SVO_DATASET_DIR");
  std::string dataset_dir(ros::package::getPath("svo")+"/test/data");
  if(env_dir != NULL)
    dataset_dir = std::string(env_dir);
  return dataset_dir;
}

inline std::string getTestDataDir()
{
#if 1
  return std::string(ros::package::getPath("svo")+"/test/data");
#else
  const char* env_dir = std::getenv("SVO_DATASET_DIR");
  return std::string(env_dir);
#endif
}

inline std::string getTraceDir()
{
#if 1
  std::string default_dir(ros::package::getPath("svo")+"/test/results");
  return vk::getParam<std::string>("svo/trace_dir", default_dir);
#else
  return "/tmp";
#endif
}

inline Eigen::Vector3d generateRandomPoint(double max_depth, double min_depth)
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

inline void calcHist(const std::vector<double>& values, size_t bins, std::vector<size_t>* hist)
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
inline cv::Mat drawHist(
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


#endif // TEST_UTILS_H_
