#pragma once

#ifdef LUCY_LOGGING
# include <oal/logfile/baselogger.hpp>
# define PREFIX "tracking"
# define LOG_TAG "svo"
# include <oal/logfile/enable_logging_macros.hpp>
#endif

#ifdef SVO_USE_ROS
# include <ros/console.h>
# define SVO_DEBUG_STREAM(x) ROS_DEBUG_STREAM(x)
# define SVO_INFO_STREAM(x) ROS_INFO_STREAM(x)
# define SVO_WARN_STREAM(x) ROS_WARN_STREAM(x)
# define SVO_WARN_STREAM_THROTTLE(rate, x) ROS_WARN_STREAM_THROTTLE(rate, x)
# define SVO_ERROR_STREAM(x) ROS_ERROR_STREAM(x)
#else
# ifdef LUCY_LOGGING
#   define SVO_INFO_STREAM(x) {std::stringstream ss; ss << x; LOGI("%s", ss.str().c_str()); }
#   define SVO_WARN_STREAM(x) {std::stringstream ss; ss << x; LOGW("%s", ss.str().c_str()); }
#   define SVO_DEBUG_STREAM(x) {std::stringstream ss; ss << x; LOGD("%s", ss.str().c_str()); }
#   define SVO_ERROR_STREAM(x) {std::stringstream ss; ss << x; LOGE("%s", ss.str().c_str()); }
# else
#   define SVO_INFO_STREAM(x) {std::cerr<<"\033[0;0m[INFO] SVO: "<<x<<"\033[0;0m"<<std::endl; }
#   define SVO_DEBUG_STREAM(x) {std::cerr<<"\033[0;0m[DEBUG] SVO: "<<x<<"\033[0;0m"<<std::endl; }
#   define SVO_WARN_STREAM(x) {std::cerr<<"\033[0;33m[WARN] SVO: "<<x<<"\033[0;0m"<<std::endl; }
#   define SVO_ERROR_STREAM(x) {std::cerr<<"\033[1;31m[ERROR] SVO: "<<x<<"\033[0;0m"<<std::endl; }
#endif
# include <chrono> // Adapted from rosconsole. Copyright (c) 2008, Willow Garage, Inc.
# define SVO_WARN_STREAM_THROTTLE(rate, x) \
    do { \
      static double __log_stream_throttle__last_hit__ = 0.0; \
      std::chrono::time_point<std::chrono::system_clock> __log_stream_throttle__now__ = \
      std::chrono::system_clock::now(); \
      if (__log_stream_throttle__last_hit__ + rate <= \
          std::chrono::duration_cast<std::chrono::seconds>( \
          __log_stream_throttle__now__.time_since_epoch()).count()) { \
        __log_stream_throttle__last_hit__ = \
        std::chrono::duration_cast<std::chrono::seconds>( \
        __log_stream_throttle__now__.time_since_epoch()).count(); \
        SVO_WARN_STREAM(x); \
      } \
    } while(0)
#endif
