#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>      // std::setprecision
#include <vikit/timer.h>

int main(int argc, char **argv)
{
  vk::Timer t;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::cout << "1. sleep took " << t.stop() << " seconds" << std::endl;
  std::cout << "1. sleep took " << t.getTime() << " seconds" << std::endl;

  t.resume();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::cout << "2. sleep took " << t.stop() << " seconds" << std::endl;
  std::cout << "2. sleep took " << t.getTime() << " seconds" << std::endl;

  std::cout << "total sleep took " << t.getAccumulated() << " seconds" << std::endl;

  std::cout << "seconds since 1.1.1970 is "
            << std::setprecision(15)
            << vk::Timer::getCurrentTime() << std::endl;

  std::cout << "current date is " << vk::Timer::getCurrentTimeStr() << std::endl;
  return 0;
}
