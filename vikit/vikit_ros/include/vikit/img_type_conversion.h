#pragma once

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

namespace vk {
namespace img_type_conversion
{
int sensorMsgsEncodingToOpenCVType(const std::string& encoding);

std::string openCVTypeToSensorMsgsEncoding(const int opencv_type);
} // namespace img_type_conversion
} // namespace vk
