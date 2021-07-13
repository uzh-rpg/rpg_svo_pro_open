/*
 * blender_utils.h
 *
 *  Created on: Feb 13, 2014
 *      Author: cforster
 */

#ifndef VIKIT_BLENDER_UTILS_H_
#define VIKIT_BLENDER_UTILS_H_

#include <list>
#include <string>
#include <glog/logging.h>
#include <vikit/math_utils.h>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <Eigen/Core>

namespace vk {
namespace blender_utils {

void loadBlenderDepthmap(
    const std::string file_name,
    const int img_width,
    const int img_height,
    cv::Mat& z_map)
{
  std::ifstream file_stream(file_name.c_str());
  CHECK(file_stream.is_open()) << "file '" << file_name << "' could not be opened.";
  z_map = cv::Mat(img_height, img_width, CV_32FC1);
  float * img_ptr = z_map.ptr<float>();
  float depth;
  for(int y=0; y<img_height; ++y)
  {
    for(int x=0; x<img_width; ++x, ++img_ptr)
    {
      file_stream >> depth;

      // blender:
      *img_ptr = depth;

      // povray
      // *img_ptr = depth/100.0; // depth is in [cm], we want [m]

      if(file_stream.peek() == '\n' && x != img_width-1 && y != img_height-1)
        printf("WARNING: did not read the full depthmap!\n");
    }
  }
}

} // namespace blender_utils
} // namespace vk

#endif // VIKIT_BLENDER_UTILS_H_
