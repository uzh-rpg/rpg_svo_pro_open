#include "svo/ceres_backend/error_interface.hpp"

namespace svo
{
namespace ceres_backend
{
const std::map<ErrorType, std::string> kErrorToStr
{
  {ErrorType::kHomogeneousPointError, std::string("HomogeneousPointError") },
  {ErrorType::kReprojectionError, std::string("ReprojectionError") },
  {ErrorType::kSpeedAndBiasError, std::string("SpeedAndBiasError") },
  {ErrorType::kMarginalizationError, std::string("MarginalizationError") },
  {ErrorType::kPoseError, std::string("PoseError") },
  {ErrorType::kIMUError, std::string("IMUError") },
  {ErrorType::kRelativePoseError, std::string("RelativePoseError") },
};

}

}
