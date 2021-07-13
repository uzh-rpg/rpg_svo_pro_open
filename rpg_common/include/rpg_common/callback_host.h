#pragma once

#include <functional>
#include <unordered_set>

#include <glog/logging.h>

#include "rpg_common/batch_worker.h"
#include "rpg_common/realtime_worker.h"

namespace rpg_common {

template <typename ... CallbackArgs>
class CallbackHost
{
 public:
  typedef std::function<void(CallbackArgs...)> Callback;

  void addCallback(const Callback& callback)
  {
    CHECK(callback) << "Callback isn't a valid function!";
    CHECK(callbacks_.emplace(callback).second)
    << "Duplicate callback addition!";
  }

  void linkBatchWorker(BatchWorker<CallbackArgs...>* worker)
  {
    CHECK_NOTNULL(worker);
    addCallback([worker](CallbackArgs... args){
      worker->addTask(args...);
    });
  }
  void linkRealtimeWorker(RealtimeWorker<CallbackArgs...>* worker)
  {
    CHECK_NOTNULL(worker);
    addCallback([worker](CallbackArgs... args){
      worker->addTask(args...);
    });
  }

 protected:
  void triggerCallbacks(CallbackArgs... args) const
  {
    for (const Callback& callback : callbacks_)
    {
      callback(args...);
    }
    VLOG(40) << callbacks_.size() << " callbacks processed.";
  }
 private:
  std::unordered_set<Callback> callbacks_;
};

}  // namespace rpg_common

namespace std {

template <typename ... CallbackArgs>
struct hash<std::function<void(CallbackArgs...)>>
{
  size_t operator()(const std::function<void(CallbackArgs...)>& x) const
  {
    CHECK(x);
    void (*const* pointer)(CallbackArgs...) =
        x.template target<void(*)(CallbackArgs...)>();
    if (pointer)  // Functor is a function pointer.
    {
      return std::hash<const void*>()(reinterpret_cast<const void*>(pointer));
    }
    else  // Functor is a lambda.
    {
      return std::hash<std::string>()(x.target_type().name());
    }
  }
};

template <typename ... CallbackArgs>
bool operator ==(
    const std::function<void(CallbackArgs...)>& a,
    const std::function<void(CallbackArgs...)>& b)
{
  CHECK(a);
  CHECK(b);
  void (*const* a_pointer)(CallbackArgs...) =
      a.template target<void(*)(CallbackArgs...)>();
  void (*const* b_pointer)(CallbackArgs...) =
      b.template target<void(*)(CallbackArgs...)>();
  if (a_pointer)  // Functor is a function pointer.
  {
    return a_pointer == b_pointer;
  }
  else  // Functor is a lambda.
  {
    return a.target_type().name() == b.target_type().name();
  }
}

}  // namespace std
namespace rpg = rpg_common;
