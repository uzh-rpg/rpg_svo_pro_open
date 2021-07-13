#pragma once

#include <svo/common/types.h>
#include <svo/common/transformation.h>

namespace svo {

/// The state of a ViNode (pose, velocity and bias). 
/// From SWE.
class ViNodeState {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ViNodeState(const Transformation& T_W_B)
      :  T_W_B_(T_W_B),
         W_v_B_(Eigen::Vector3d::Zero()),
         acc_bias_(Eigen::Vector3d::Zero()),
         gyro_bias_(Eigen::Vector3d::Zero()) {}

  ViNodeState()
      :  W_v_B_(Eigen::Vector3d::Zero()),
         acc_bias_(Eigen::Vector3d::Zero()),
         gyro_bias_(Eigen::Vector3d::Zero()) {}

  ViNodeState(const Transformation& T_W_B,
              const Eigen::Vector3d& W_v_B,
              const Eigen::Vector3d& accelerometer_bias,
              const Eigen::Vector3d& gyro_bias)
      : T_W_B_(T_W_B), W_v_B_(W_v_B),
        acc_bias_(accelerometer_bias), gyro_bias_(gyro_bias) {}

  virtual ~ViNodeState() {}

  inline const Transformation& get_T_W_B() const { return T_W_B_; }
  inline Transformation& get_T_W_B() { return T_W_B_; }
  inline const Eigen::Vector3d& get_W_v_B() const { return W_v_B_; }
  inline const Eigen::Vector3d& getAccBias() const { return acc_bias_; }
  inline const Eigen::Vector3d& getGyroBias() const { return gyro_bias_; }

  inline Eigen::Matrix<double, 6, 1> getImuBias() const {
    return (Eigen::Matrix<double, 6, 1>()
            << getAccBias(), getGyroBias()).finished();
  }

  inline void set_T_W_B(const Transformation& T_W_B) { T_W_B_ = T_W_B; }
  inline void set_W_v_B(const Eigen::Vector3d& W_v_B) { W_v_B_ = W_v_B; }
  inline void setAccBias(const Eigen::Vector3d& acc_bias)
  {
    acc_bias_ = acc_bias;
  }
  inline void setGyroBias(const Eigen::Vector3d& gyro_bias)
  {
    gyro_bias_ = gyro_bias;
  }

 private:
  /// The pose taking points from the body frame to the world frame.
  Transformation T_W_B_;
  /// The velocity (m/s).
  Eigen::Vector3d W_v_B_;
  /// The accelerometer bias (m/s^2).
  Eigen::Vector3d acc_bias_;
  /// The gyroscope bias (rad/s).
  Eigen::Vector3d gyro_bias_;
};

} // namespace svo
