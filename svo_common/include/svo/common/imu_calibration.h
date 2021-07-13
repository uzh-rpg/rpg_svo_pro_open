#pragma once

#include <deque>
#include <iostream>
#include <svo/common/types.h>

namespace svo {

/// \brief Settings for the IMU statistics
/// Check the following references for more information about the IMU parameters:
/// * https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model-and-Intrinsics
/// * http://www-users.cs.umn.edu/~trawny/Publications/Quaternions_3D.pdf (Sec. 2.1)
/// Default parameters are for ADIS16448 IMU.
class ImuCalibration
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<ImuCalibration> Ptr;

  /// Camera-IMU delay: delay_imu_cam = cam_ts - cam_ts_delay
  double delay_imu_cam = 0.0;

  /// Maximum delay camera-imu
  double max_imu_delta_t = 0.01;

  /// Gyro noise density (sigma). [rad/s*1/sqrt(Hz)]
  double gyro_noise_density = 0.00073088444;

  /// Accelerometer noise density (sigma). [m/s^2*1/sqrt(Hz)]
  double acc_noise_density = 0.01883649;

  /// IMU integration sigma (sigma). GTSAM preintegration option.
  double imu_integration_sigma = 0.0;

  /// Gyro bias random walk (sigma). [rad/s^2*1/sqrt(Hz)]
  double gyro_bias_random_walk_sigma = 0.00038765;

  /// Accelerometer bias random walk (sigma). [m/s^3*1/sqrt(Hz)]
  double acc_bias_random_walk_sigma = 0.012589254;

  /// Norm of the Gravitational acceleration. [m/s^2]
  double gravity_magnitude = 9.81007;

  /// Coriolis acceleration (earth rotation rate).
  Eigen::Vector3d omega_coriolis = Eigen::Vector3d::Zero();

  /// Accelerometer saturation. [m/s^2]
  double saturation_accel_max = 150;

  /// Gyroscope saturation. [rad/s]
  double saturation_omega_max = 7.8;

  /// Expected IMU Rate [1/s]
  double imu_rate = 20;

  ImuCalibration() = default;
  ~ImuCalibration() = default;

  inline void print(const std::string& s = "IMU Calibration: ") const
  {
    std::cout << s << std::endl
              << "  delay_imu_cam = " << delay_imu_cam << std::endl
              << "  sigma_omega_c = " << gyro_noise_density << std::endl
              << "  sigma_acc_c = " << acc_noise_density << std::endl
              << "  sigma_integration_c = " << imu_integration_sigma << std::endl
              << "  sigma_omega_bias_c = " << gyro_bias_random_walk_sigma << std::endl
              << "  sigma_acc_bias_c = " << acc_bias_random_walk_sigma << std::endl
              << "  g = " << gravity_magnitude << std::endl
              << "  coriolis = " << omega_coriolis.transpose() << std::endl;
  }
};

class ImuInitialization
{
public:
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  typedef std::shared_ptr<ImuInitialization> Ptr;

  /// Initial velocity, in world frame!
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

  /// Initial gyroscope bias
  Eigen::Vector3d omega_bias = Eigen::Vector3d::Zero();

  /// Initial accelerometer bias
  Eigen::Vector3d acc_bias = Eigen::Vector3d::Zero();

  /// Initial velocity uncertainty
  double velocity_sigma = 2.0;

  /// Initial gyroscope bias uncertainty
  double omega_bias_sigma = 0.01;

  /// Initial acceleromter bias uncertainty
  double acc_bias_sigma = 0.1;

  inline void print(const std::string& s = "IMU Initialization: ") const
  {
    std::cout << s << std::endl
              << "  velocity = " << velocity.transpose() << std::endl
              << "  omega_bias = " << omega_bias.transpose() << std::endl
              << "  acc_bias = " << acc_bias.transpose() << std::endl
              << "  velocity_sigma = " << velocity_sigma << std::endl
              << "  omega_bias_sigma = " << omega_bias_sigma << std::endl
              << "  acc_bias_sigma = " << acc_bias_sigma << std::endl;
  }
};

// TODO(cfo) Deprecate
struct ImuMeasurement
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp_; ///< In seconds.
  Eigen::Vector3d angular_velocity_;
  Eigen::Vector3d linear_acceleration_;
  ImuMeasurement() {}
  ImuMeasurement(
      const double timestamp,
      const Eigen::Vector3d& angular_velocity,
      const Eigen::Vector3d& linear_acceleration)
  : timestamp_(timestamp)
  , angular_velocity_(angular_velocity)
  , linear_acceleration_(linear_acceleration)
  {}
};
typedef std::deque<ImuMeasurement,
Eigen::aligned_allocator<ImuMeasurement> > ImuMeasurements;

} // namespace svo
