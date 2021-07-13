#pragma once

#include <thread>

#include "rpg_common/threadsafe_queue.h"

namespace rpg_common {

template <typename DataType>
class RealtimeWorker
{
 public:
  RealtimeWorker() : thread_(&RealtimeWorker<DataType>::workLoop, this) {}

  virtual ~RealtimeWorker()
  {
    shutdown();
  }

  void addTask(const DataType& item)
  {
    queue_.push(item);
  }

  void shutdown()
  {
    if (!thread_.joinable())
    {
      LOG(WARNING) << "Redundant shutdown call of real-time worker!";
      return;
    }
    queue_.shutdown();
    thread_.join();
  }

  void printBacklogWarningsWithTag(const std::string& tag)
  {
    queue_.printBacklogWarningsWithTag(
        "Queue of worker with tag \"" + tag + "\"");
  }

 private:
  virtual void process(const DataType& item) = 0;

  void workLoop()
  {
    DataType item;
    while (queue_.skipToLatest(&item))
    {
      process(item);
    }
  }

  ThreadSafeQueue<DataType> queue_;
  std::thread thread_;
};

}  // namespace rpg_common
namespace rpg = rpg_common;
