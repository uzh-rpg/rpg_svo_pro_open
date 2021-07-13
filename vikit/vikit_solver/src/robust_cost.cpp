#include "vikit/solver/robust_cost.h"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <glog/logging.h>

namespace vk {
namespace solver {

/* ************************************************************************* */
// Scale Estimators
/* ************************************************************************* */
float UnitScaleEstimator::compute(std::vector<float>& /*errors*/) const
{
  return 1.0f;
}

float MADScaleEstimator::compute(std::vector<float>& errors) const
{
  CHECK(!errors.empty()) << "Error vector is empty.";
  auto it = errors.begin()+std::floor(errors.size()/2);
  std::nth_element(errors.begin(), it, errors.end()); // compute median
  return 1.48f * (*it); // 1.48f / 0.6745
}

float NormalDistributionScaleEstimator::compute(std::vector<float>& errors) const
{
  const float mean = std::accumulate(errors.begin(), errors.end(), 0)/errors.size();
  float var = 0.0;
  for(const float d : errors)
    var += (d - mean) * (d - mean);
  return std::sqrt(var); // return standard deviation
}

/* ************************************************************************* */
// Weight Functions
/* ************************************************************************* */
float UnitWeightFunction::weight(const float& error) const
{
  return 1.0f;
}

TukeyWeightFunction::TukeyWeightFunction(const float b)
  : b_square_(b*b)
{}

float TukeyWeightFunction::weight(const float& error) const
{
  const float x_square = error * error;
  if(x_square <= b_square_)
  {
    const float tmp = 1.0f - x_square / b_square_;
    return tmp * tmp;
  }
  else
  {
    return 0.0f;
  }
}

HuberWeightFunction::HuberWeightFunction(const float k)
  : k_(k)
{}

float HuberWeightFunction::weight(const float& error) const
{
  const float abs_error = std::fabs(error);
  return (abs_error < k_) ? 1.0f : k_/abs_error;
}

} // namespace robust_cost
} // namespace vk


