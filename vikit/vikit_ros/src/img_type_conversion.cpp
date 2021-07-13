#include <vikit/img_type_conversion.h>

namespace vk {
namespace img_type_conversion{

int sensorMsgsEncodingToOpenCVType(const std::string &encoding)
{
  if((encoding == sensor_msgs::image_encodings::MONO8) ||
     (encoding == sensor_msgs::image_encodings::TYPE_8UC1))
  {
    return CV_8UC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_8UC2)
  {
    return CV_8UC2;
  }
  if((encoding == sensor_msgs::image_encodings::BGR8) ||
     (encoding == sensor_msgs::image_encodings::TYPE_8UC3) ||
     (encoding == sensor_msgs::image_encodings::RGB8))
  {
    return CV_8UC3;
  }
  if((encoding == sensor_msgs::image_encodings::BGRA8) ||
     (encoding == sensor_msgs::image_encodings::TYPE_8UC4) ||
     (encoding == sensor_msgs::image_encodings::RGBA8))
  {
    return CV_8UC4;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_8SC1)
  {
    return CV_8SC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_8SC2)
  {
    return CV_8SC2;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_8SC3)
  {
    return CV_8SC3;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_8SC4)
  {
    return CV_8SC4;
  }
  if((encoding == sensor_msgs::image_encodings::MONO16) ||
     (encoding == sensor_msgs::image_encodings::TYPE_16UC1))
  {
    return CV_16UC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_16UC2)
  {
    return CV_16UC2;
  }
  if((encoding == sensor_msgs::image_encodings::BGR16) ||
     (encoding == sensor_msgs::image_encodings::TYPE_16UC3) ||
     (encoding == sensor_msgs::image_encodings::RGB16))
  {
    return CV_16UC3;
  }
  if((encoding == sensor_msgs::image_encodings::BGRA16) ||
     (encoding == sensor_msgs::image_encodings::TYPE_16UC4) ||
     (encoding == sensor_msgs::image_encodings::RGBA16))
  {
    return CV_16UC4;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_16SC1)
  {
    return CV_16SC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_16SC2)
  {
    return CV_16SC2;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_16SC3)
  {
    return CV_16SC3;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_16SC4)
  {
    return CV_16SC4;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32SC1)
  {
    return CV_32SC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32SC2)
  {
    return CV_32SC2;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32SC3)
  {
    return CV_32SC3;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32SC4)
  {
    return CV_32SC4;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    return CV_32FC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32FC2)
  {
    return CV_32FC2;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32FC3)
  {
    return CV_32FC3;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_32FC4)
  {
    return CV_32FC4;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_64FC1)
  {
    return CV_64FC1;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_64FC2)
  {
    return CV_64FC2;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_64FC3)
  {
    return CV_64FC3;
  }
  if(encoding == sensor_msgs::image_encodings::TYPE_64FC4)
  {
    return CV_64FC4;
  }

  // invalid sensor_msgs encoding was passed
  ROS_ERROR("[sensorMsgsEncodingToOpenCVType] Invalid sensor_msgs encoding was passed");
  return -1;
}

std::string openCVTypeToSensorMsgsEncoding(const int opencv_type)
{
  if(opencv_type == CV_8UC1)
  {
    return sensor_msgs::image_encodings::MONO8;
  }
  if(opencv_type == CV_8UC2)
  {
    return sensor_msgs::image_encodings::TYPE_8UC2;
  }
  if(opencv_type == CV_8UC3)
  {
    return sensor_msgs::image_encodings::BGR8;
  }
  if(opencv_type == CV_8UC4)
  {
    return sensor_msgs::image_encodings::BGRA8;
  }
  if(opencv_type == CV_8SC1)
  {
    return sensor_msgs::image_encodings::TYPE_8SC1;
  }
  if(opencv_type == CV_8SC2)
  {
    return sensor_msgs::image_encodings::TYPE_8SC2;
  }
  if(opencv_type == CV_8SC3)
  {
    return sensor_msgs::image_encodings::TYPE_8SC3;
  }
  if(opencv_type == CV_8SC4)
  {
    return sensor_msgs::image_encodings::TYPE_8SC4;
  }
  if(opencv_type == CV_16UC1)
  {
    return sensor_msgs::image_encodings::MONO16;
  }
  if(opencv_type == CV_16UC2)
  {
    return sensor_msgs::image_encodings::TYPE_16UC2;
  }
  if(opencv_type == CV_16UC3)
  {
    return sensor_msgs::image_encodings::BGR16;
  }
  if(opencv_type == CV_16UC4)
  {
    return sensor_msgs::image_encodings::BGRA16;
  }
  if(opencv_type == CV_16SC1)
  {
    return sensor_msgs::image_encodings::TYPE_16SC1;
  }
  if(opencv_type == CV_16SC2)
  {
    return sensor_msgs::image_encodings::TYPE_16SC2;
  }
  if(opencv_type == CV_16SC3)
  {
    return sensor_msgs::image_encodings::TYPE_16SC3;
  }
  if(opencv_type == CV_16SC4)
  {
    return sensor_msgs::image_encodings::TYPE_16SC4;
  }
  if(opencv_type == CV_32SC1)
  {
    return sensor_msgs::image_encodings::TYPE_32SC1;
  }
  if(opencv_type == CV_32SC2)
  {
    return sensor_msgs::image_encodings::TYPE_32SC2;
  }
  if(opencv_type == CV_32SC3)
  {
    return sensor_msgs::image_encodings::TYPE_32SC3;
  }
  if(opencv_type == CV_32SC4)
  {
    return sensor_msgs::image_encodings::TYPE_32SC4;
  }
  if(opencv_type == CV_32FC1)
  {
    return sensor_msgs::image_encodings::TYPE_32FC1;
  }
  if(opencv_type == CV_32FC2)
  {
    return sensor_msgs::image_encodings::TYPE_32FC2;
  }
  if(opencv_type == CV_32FC3)
  {
    return sensor_msgs::image_encodings::TYPE_32FC3;
  }
  if(opencv_type == CV_32FC4)
  {
    return sensor_msgs::image_encodings::TYPE_32FC4;
  }
  if(opencv_type == CV_64FC1)
  {
    return sensor_msgs::image_encodings::TYPE_64FC1;
  }
  if(opencv_type == CV_64FC2)
  {
    return sensor_msgs::image_encodings::TYPE_64FC2;
  }
  if(opencv_type == CV_64FC3)
  {
    return sensor_msgs::image_encodings::TYPE_64FC3;
  }
  if(opencv_type == CV_64FC4)
  {
    return sensor_msgs::image_encodings::TYPE_64FC4;
  }

  // invalid opencv type was passed
  ROS_ERROR("[openCVTypeToSensorMsgsEncoding] Invalid OpenCV type was passed");
  return "";
}

} // namespace img_type_conversion
} // namespace vk
