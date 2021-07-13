/*
 * ros_params_helper.h
 *
 *  Created on: Feb 22, 2013
 *      Author: cforster
 *
 * from libpointmatcher_ros
 */

#ifndef ROS_PARAMS_HELPER_H_
#define ROS_PARAMS_HELPER_H_

#include <string>
#include <ros/ros.h>

namespace vk {

inline
bool hasParam(const std::string& name)
{
  return ros::param::has(name);
}

template<typename T>
T getParam(const std::string& name, const T& defaultValue)
{
  T v;
  if(ros::param::get(name, v))
  {
    ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
    return v;
  }
  else
    ROS_WARN_STREAM("Cannot find value for parameter: " << name << ", assigning default: " << defaultValue);
  return defaultValue;
}

template<typename T>
T getParam(const std::string& name)
{
  T v;
  if(ros::param::get(name, v))
  {
    ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
    return v;
  }
  else
    ROS_ERROR_STREAM("Cannot find value for parameter: " << name);
  return T();
}

template<typename T>
T param(const ros::NodeHandle& nh, const std::string& name, const T& defaultValue,
        const bool silent=false)
{
  if(nh.hasParam(name))
  {
    T v;
    nh.param<T>(name, v, defaultValue);
    if (!silent)
    {
      ROS_INFO_STREAM("Found parameter: " << name << ", value: " << v);
    }
    return v;
  }
  if (!silent)
  {
    ROS_WARN_STREAM("Cannot find value for parameter: " << name << ", assigning default: " << defaultValue);
  }
  return defaultValue;
}

} // namespace vk

#endif // ROS_PARAMS_HELPER_H_
