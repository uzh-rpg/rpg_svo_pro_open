#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip> // std::setw
#include <ctime>   // std::localtime,

namespace rpg_common
{

class Timer
{
public:
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::time_point<Clock> TimePoint;
  typedef std::chrono::nanoseconds Nanoseconds;
  typedef std::chrono::seconds Seconds;

  /// The constructor directly starts the timer.
  Timer()
    : start_time_(Clock::now())
    , duration_(Nanoseconds::zero())
    , accumulated_(Nanoseconds::zero())
  {}

  /// Starts the timer
  inline void start()
  {
    start_time_ = Clock::now();
  }

  /// Resumes the timer. Total time can be obtained with getAccumulated().
  inline void resume()
  {
    start_time_ = Clock::now();
  }

  /// Returns duration in seconds
  inline double stop()
  {
    const TimePoint end_time(Clock::now());
    duration_ = std::chrono::duration_cast<Nanoseconds>(end_time - start_time_);
    accumulated_ += duration_;
    return static_cast<double>(duration_.count())*1e-9;
  }

  /// Returns duration of last measurement in seconds
  inline double getTime() const
  {
    return static_cast<double>(duration_.count())*1e-9;
  }

  /// Returns duration of last measurement in milliseconds
  inline double getMilliseconds() const
  {
    return static_cast<double>(duration_.count())*1e-6;
  }

  /// Returns duration since the last reset or construction of the timer
  inline double getAccumulated() const
  {
    return static_cast<double>(accumulated_.count())*1e-9;
  }

  /// Reset the current timer and the accumulated
  inline void reset()
  {
    start_time_ = TimePoint();
    duration_ = Nanoseconds::zero();
    accumulated_ = Nanoseconds::zero();
  }

  /// Get seconds since 1.1.1970
  static double getCurrentTime()
  {
    return static_cast<double>(
          std::chrono::duration_cast<Nanoseconds>(Clock::now()-TimePoint())
          .count())*1e-9;
  }

  /// Get a formated string of the current time, hour, minute and second
  static std::string getCurrentTimeStr()
  {
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm* t = std::localtime(&now);
    if(t == NULL)
      return std::string("ERROR");
    std::ostringstream ss;
    ss << t->tm_year-100 << "-"
       << std::setw(2) << std::setfill('0') << t->tm_mon+1 << "-"
       << std::setw(2) << std::setfill('0') << t->tm_mday << "_"
       << std::setw(2) << std::setfill('0') << t->tm_hour << "-"
       << std::setw(2) << std::setfill('0') << t->tm_min << "-"
       << std::setw(2) << std::setfill('0') << t->tm_sec;
    return ss.str();
  }

private:
  TimePoint start_time_;
  Nanoseconds duration_;
  Nanoseconds accumulated_;
};

} // namespace rpg_common
namespace rpg = rpg_common;
