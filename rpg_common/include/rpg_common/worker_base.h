#pragma once

namespace rpg_common {

// Template-free interface to worker features that do not depend on the queue
// data type.
class WorkerBase
{
public:
  virtual ~WorkerBase() = default;

  // Shuts down once the current loop is finished.
  virtual void shutdown() = 0;
  // Shuts down once the queue is empty.
  virtual void softShutdown() = 0;

  virtual void pause() = 0;
  virtual bool isPaused() const = 0;
  virtual void resume() = 0;
};

}  // namespace rpg_common
namespace rpg = rpg_common;
