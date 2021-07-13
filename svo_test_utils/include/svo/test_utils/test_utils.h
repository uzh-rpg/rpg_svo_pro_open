// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#pragma once

#include <cstdlib> // for getenv rand
#include <string>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <vector>

# include <vikit/cameras/ncamera.h>

#include <svo/common/frame.h>


#define SVO_TEST_STREAM(x) {std::cerr<<"\033[0;0m[          ] * "<<x<<"\033[0;0m"<<std::endl; }

namespace svo {
namespace test_utils {

std::string getDatasetDir();
std::string getTestDataDir();
std::string getTraceDir();

struct VectorStats
{
  double mean;
  double stdev;
  double median;
  double percentile90th;
  double percentile10th;
};

VectorStats computeStats(std::vector<double>& v);

FrameBundle::Ptr createFrameBundle(
    CameraPtr cam,
    const Transformation& T_w_f,
    const Transformation& T_f_b);

Eigen::Vector3d generateRandomPoint(double max_depth, double min_depth);

void calcHist(const std::vector<double>& values, size_t bins, std::vector<size_t>* hist);

// TODO(zzc): auto zoom for the text
cv::Mat drawHist(
    const std::vector<size_t>& hist,
    const std::vector<double>& bounds,
    int width, int height);

} // namespace test_utils
} // namespace svo
