#pragma once

#include <memory>
#include <nodelet/nodelet.h>

namespace svo {

// forward declarations
class SvoInterface;

class SvoNodelet : public nodelet::Nodelet
{
public:
  SvoNodelet() = default;
  virtual ~SvoNodelet();

 private:
  virtual void onInit();

  std::unique_ptr<SvoInterface> svo_interface_;
};

} // namespace svo
