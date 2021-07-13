// to cope with the warning from the distribution generator
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59800
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#include <svo/test_utils/imu_simulation.h>

#include <chrono>
#include <random>
#include <glog/logging.h>

#include <svo/common/types.h>
#include <svo/common/imu_calibration.h>

namespace svo {
namespace simulation {

namespace internal {

void simulateBias(
    const size_t n_measurements,
    const double dt,
    const double sigma_continuous,
    const Eigen::Vector3d& bias_init,
    Eigen::Matrix3Xd* biases)
{
  CHECK_NOTNULL(biases);
  const uint8_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
  std::normal_distribution<double> normal_dist(0.0, 1.0);
  const double sqrt_dt = std::sqrt(dt);
  biases->resize(Eigen::NoChange, n_measurements);
  biases->col(0) = bias_init;
  for(size_t i = 1; i < n_measurements; ++i)
  {
    const Eigen::Vector3d N(normal_dist(gen), normal_dist(gen), normal_dist(gen));
    biases->col(i) = biases->col(i-1) + N * sigma_continuous * sqrt_dt;
  }
}

} // namespace internal

void simulateBiases(
    const size_t n_measurements,
    const double dt,
    const ImuCalibration& imu_calib,
    const Eigen::Vector3d& bias_gyr_init,
    const Eigen::Vector3d& bias_acc_init,
    Eigen::Matrix3Xd* bias_gyr,
    Eigen::Matrix3Xd* bias_acc)
{
  CHECK_NOTNULL(bias_gyr);
  CHECK_NOTNULL(bias_acc);
  internal::simulateBias(n_measurements, dt, imu_calib.gyro_bias_random_walk_sigma,
                         bias_gyr_init, bias_gyr);
  internal::simulateBias(n_measurements, dt, imu_calib.acc_bias_random_walk_sigma,
                         bias_acc_init, bias_acc);
}

} // namespace simulation
} // namespace svo
