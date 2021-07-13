#ifndef VIKIT_ROBUST_COST_H_
#define VIKIT_ROBUST_COST_H_

#include <vector>
#include <memory>

namespace vk {
namespace solver {

/// Scale Estimators to estimate standard deviation of a distribution of errors.
class ScaleEstimator
{
public:
  virtual ~ScaleEstimator() = default;

  /// Errors must be absolute values!
  virtual float compute(std::vector<float>& errors) const = 0;
};
typedef std::shared_ptr<ScaleEstimator> ScaleEstimatorPtr;

class UnitScaleEstimator : public ScaleEstimator
{
public:
  using ScaleEstimator::ScaleEstimator;
  virtual ~UnitScaleEstimator() = default;
  virtual float compute(std::vector<float>& errors) const;
};

// estimates scale by computing the median absolute deviation
class MADScaleEstimator : public ScaleEstimator
{
public:
  using ScaleEstimator::ScaleEstimator;
  virtual ~MADScaleEstimator() = default;
  virtual float compute(std::vector<float>& errors) const;
};

// estimates scale by computing the standard deviation
class NormalDistributionScaleEstimator : public ScaleEstimator
{
public:
  using ScaleEstimator::ScaleEstimator;
  virtual ~NormalDistributionScaleEstimator() = default;
  virtual float compute(std::vector<float>& errors) const;
private:
};

/// Weight-Functions for M-Estimators
/// http://research.microsoft.com/en-us/um/people/zhang/inria/publis/tutorial-estim/node24.html
class WeightFunction
{
public:
  WeightFunction() = default;
  virtual ~WeightFunction() = default;
  virtual float weight(const float& error) const = 0;
};
typedef std::shared_ptr<WeightFunction> WeightFunctionPtr;

class UnitWeightFunction : public WeightFunction
{
public:
  using WeightFunction::WeightFunction;
  virtual ~UnitWeightFunction() = default;
  virtual float weight(const float& error) const;
};

class TukeyWeightFunction : public WeightFunction
{
public:
  TukeyWeightFunction(const float b = 4.6851f);
  virtual ~TukeyWeightFunction() = default;
  virtual float weight(const float& error) const;
private:
  float b_square_;
};

class HuberWeightFunction : public WeightFunction
{
public:
  HuberWeightFunction(const float k = 1.345f);
  virtual ~HuberWeightFunction() = default;
  virtual float weight(const float& error) const;
private:
  float k_;
};

} // namespace robust_cost
} // namespace vk

#endif // VIKIT_ROBUST_COST_H_
