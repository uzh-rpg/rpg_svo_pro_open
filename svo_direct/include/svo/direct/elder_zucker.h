#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

namespace svo {
namespace elder_zucker {

void detectEdges(
    const std::vector<cv::Mat>& img_pyr,
    const double sigma,
    cv::Mat& edge_map,
    cv::Mat& level_map);

void getCovarEntries(
    const cv::Mat& src,
    cv::Mat& dxdx,
    cv::Mat& dydy,
    cv::Mat& dxdy);

void filterGauss3by316S(
    const cv::Mat& src,
    cv::Mat& dst);

} // namespace elder_zucker
} // namespace svo
