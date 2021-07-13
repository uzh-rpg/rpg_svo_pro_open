// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#pragma once

#include <limits>
#include <vector>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <kindr/minimal/quat-transformation.h>
#pragma diagnostic pop

#include <svo/common/types.h>

#include "svo/vio_common/matrix.hpp"

namespace svo {

// Transformation utils

// Right Jacobian for Exponential map in SO(3)
inline Eigen::Matrix<FloatType,3,3> expmapDerivativeSO3(
    const Eigen::Matrix<FloatType,3,1>& omega)
{
  FloatType theta2 = omega.dot(omega);
  if (theta2 <= std::numeric_limits<FloatType>::epsilon())
  {
    return Eigen::Matrix<FloatType,3,3>::Identity();
  }
  FloatType theta = std::sqrt(theta2);  // rotation angle
  // element of Lie algebra so(3): X = omega^, normalized by normx
  const Eigen::Matrix<FloatType,3,3> Y = skewSymmetric(omega) / theta;
  return Eigen::Matrix<FloatType,3,3>::Identity()
      - ((FloatType{1} - std::cos(theta)) / (theta)) * Y
      + (FloatType{1} - std::sin(theta) / theta) * Y * Y;
}

// -----------------------------------------------------------------------------
// Quaternion utils

//! Plus matrix for a quaternion. q_AB x q_BC = plus(q_AB) * q_BC.coeffs().
inline Eigen::Matrix<FloatType,4,4> quaternionPlusMatrix(
    const Eigen::Quaternion<FloatType>& q_AB)
{
  const Eigen::Matrix<FloatType,4,1>& q = q_AB.coeffs();
  Eigen::Matrix<FloatType,4,4> Q;
  Q(0,0) =  q[3]; Q(0,1) = -q[2]; Q(0,2) =  q[1]; Q(0,3) =  q[0];
  Q(1,0) =  q[2]; Q(1,1) =  q[3]; Q(1,2) = -q[0]; Q(1,3) =  q[1];
  Q(2,0) = -q[1]; Q(2,1) =  q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

//! Opposite-Plus matrix for a quaternion q_AB x q_BC = oplus(q_BC) * q_AB.coeffs().
inline Eigen::Matrix<FloatType,4,4> quaternionOplusMatrix(
    const Eigen::Quaternion<FloatType>& q_BC)
{
  const   Eigen::Matrix<FloatType,4,1>& q = q_BC.coeffs();
  Eigen::Matrix<FloatType,4,4> Q;
  Q(0,0) =  q[3]; Q(0,1) =  q[2]; Q(0,2) = -q[1]; Q(0,3) =  q[0];
  Q(1,0) = -q[2]; Q(1,1) =  q[3]; Q(1,2) =  q[0]; Q(1,3) =  q[1];
  Q(2,0) =  q[1]; Q(2,1) = -q[0]; Q(2,2) =  q[3]; Q(2,3) =  q[2];
  Q(3,0) = -q[0]; Q(3,1) = -q[1]; Q(3,2) = -q[2]; Q(3,3) =  q[3];
  return Q;
}

} // namespace svo
