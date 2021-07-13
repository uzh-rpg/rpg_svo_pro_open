#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include <glog/logging.h>

namespace rpg_common {

template<typename DataType>
class ThreadSafeQueue
{
 public:
  ThreadSafeQueue() : warn_if_has_backlog_(false), shutdown_(false) {}

  void push(const DataType& item)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      data_.push(item);
    }
    cv_push_.notify_all();
  }

  bool tryPopNow(DataType* item)
  {
    CHECK_NOTNULL(item);
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_.empty())
    {
      return false;
    }
    else
    {
      *item = data_.front();
      data_.pop();
      cv_pop_.notify_all();
      return true;
    }
  }

  // Returns false if queue has been shut down.
  bool waitAndPop(DataType* item)
  {
    CHECK_NOTNULL(item);
    std::unique_lock<std::mutex> lock(mutex_);
    if (!wait(&lock))
    {
      return false;
    }
    else
    {
      *item = data_.front();
      data_.pop();
      cv_pop_.notify_all();
      return true;
    }
  }

  // Returns false if queue has been shut down.
  bool skipToLatest(DataType* item)
  {
    CHECK_NOTNULL(item);
    std::unique_lock<std::mutex> lock(mutex_);
    if (!wait(&lock))
    {
      return false;
    }
    else
    {
      *item = data_.back();
      while (!data_.empty()) {
        data_.pop();
      }
      cv_pop_.notify_all();
      return true;
    }
  }

  // Returns false if queue has been shut down.
  bool waitAndPopAvailable(std::vector<DataType>* items)
  {
    CHECK_NOTNULL(items)->clear();
    std::unique_lock<std::mutex> lock(mutex_);
    if (!wait(&lock))
    {
      return false;
    }
    else
    {
      items->reserve(data_.size());
      while (!data_.empty()) {
        items->push_back(data_.front());
        data_.pop();
      }
      cv_pop_.notify_all();
      return true;
    }
  }

  void shutdown()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_push_.notify_all();
    cv_pop_.notify_all();
  }

  void printBacklogWarningsWithTag(const std::string& tag)
  {
    warn_if_has_backlog_ = true;
    backlog_warn_tag_ = tag;
  }

  // Returns false if shutdown has been requested.
  bool waitUntilEmpty()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_pop_.wait(
        lock, [this] { return data_.empty() || shutdown_; });
    if (shutdown_)
    {
      return false;
    }
    return true;
  }

 private:
  // Returns false if shutdown has been requested.
  bool wait(std::unique_lock<std::mutex>* lock)
  {
    CHECK_NOTNULL(lock);
    if (warn_if_has_backlog_ && !data_.empty())
    {
      LOG(WARNING) << "Backlog in queue \"" << backlog_warn_tag_ << "\"";
    }
    cv_push_.wait(
        *lock, [this] { return !data_.empty() || shutdown_; });
    if (shutdown_)
    {
      return false;
    }
    return true;
  }

  std::queue<DataType> data_;
  std::mutex mutex_;
  std::condition_variable cv_push_;
  std::condition_variable cv_pop_;

  bool warn_if_has_backlog_;
  std::string backlog_warn_tag_;
  volatile bool shutdown_;
};

}  // namespace rpg_common
namespace rpg = rpg_common;
