#pragma once

#include <atomic>
#include <thread>

#include <glog/logging.h>

#include "rpg_common/threadsafe_queue.h"
#include "rpg_common/worker_base.h"

namespace rpg_common {

template <typename DataType>
class BatchWorker : public WorkerBase
{
 public:
  BatchWorker() : paused_(true)
 {
    resume();
 }

  virtual ~BatchWorker()
  {
    shutdown();
  }

  void addTask(const DataType& item)
  {
    queue_.push(item);
  }

  virtual void shutdown() override
  {
    if (!thread_.joinable())
    {
      return;
    }
    queue_.shutdown();
    thread_.join();
  }

  virtual void softShutdown() override
  {
    if (!thread_.joinable())
    {
      return;
    }
    queue_.waitUntilEmpty();
    queue_.shutdown();
    thread_.join();
  }

  virtual void pause() override
  {
    CHECK(!paused_);
    CHECK(thread_.joinable());
    paused_ = true;
    thread_.join();
  }

  virtual bool isPaused() const override
  {
    return paused_;
  }

  virtual void resume() override
  {
    CHECK(paused_);
    CHECK(!thread_.joinable());
    paused_ = false;
    thread_ = std::thread(&BatchWorker<DataType>::workLoop, this);
  }

 private:
  virtual void process(const std::vector<DataType>& item) = 0;

  void workLoop()
  {
    std::vector<DataType> items;
    while (!paused_ && queue_.waitAndPopAvailable(&items))
    {
      process(items);
    }
  }

  ThreadSafeQueue<DataType> queue_;
  std::atomic<bool> paused_;
  std::thread thread_;
};

}  // namespace rpg_common
namespace rpg = rpg_common;
