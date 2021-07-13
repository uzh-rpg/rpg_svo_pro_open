#pragma once

#include <svo/common/types.h>

namespace svo {
namespace test_utils {

/// Point Cloud Generators
class PCG
{
public:
  typedef std::shared_ptr<PCG> Ptr;
  PCG() {}
  virtual std::vector<Eigen::Vector3d> generatePointCloud(size_t num_points) = 0;
};

class UniformRandomPCG : public PCG
{
public:
  UniformRandomPCG(double min_depth, double max_depth)
    : PCG()
    , min_depth_(min_depth)
    , max_depth_(max_depth) {}

  virtual std::vector<Eigen::Vector3d> generatePointCloud(size_t num_points);

private:
  double min_depth_;
  double max_depth_;
};

class GaussianRandomPCG : public PCG
{
public:
  GaussianRandomPCG(double mean_depth, double std_depth)
    : PCG()
    , mean_depth_(mean_depth)
    , std_depth_(std_depth) {}

  virtual std::vector<Eigen::Vector3d> generatePointCloud(size_t num_points);

private:
  double mean_depth_;
  double std_depth_;
};

class FromFilePCG : public PCG
{
public:
  FromFilePCG(const std::string& filename);
  virtual std::vector<Eigen::Vector3d> generatePointCloud(size_t /*num_points*/);

private:
  std::vector<Eigen::Vector3d> points_;
};

}
}
