/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2016, ETH Zurich, Wyss Zurich, Zurich Eye
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file PoseLocalParameterization.cpp
 * @brief Source file for the PoseLocalParameterization class.
 * @author Stefan Leutenegger
 */

#include "svo/ceres_backend/pose_local_parameterization.hpp"

#include <svo/vio_common/logging.hpp>
#include <svo/vio_common/matrix_operations.hpp>
#include <svo/common/transformation.h>

namespace svo {
namespace ceres_backend {

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool PoseLocalParameterization::Plus(const double* x, const double* delta,
                                     double* x_plus_delta) const
{
  return plus(x, delta, x_plus_delta);
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool PoseLocalParameterization::Minus(const double* x,
                                      const double* x_plus_delta,
                                      double* delta) const
{
  return minus(x, x_plus_delta, delta);
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool PoseLocalParameterization::ComputeLiftJacobian(const double* x,
                                                    double* jacobian) const
{
  return liftJacobian(x, jacobian);
}

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool PoseLocalParameterization::plus(const double* x, const double* delta,
                                     double* x_plus_delta)
{

  Eigen::Map<const Eigen::Matrix<double, 6, 1> > delta_(delta);

  Quaternion q(x[6], x[3], x[4], x[5]);
  q = Quaternion::exp(delta_.tail<3>()) * q;

  // copy back
  x_plus_delta[0] = x[0] + delta[0];
  x_plus_delta[1] = x[1] + delta[1];
  x_plus_delta[2] = x[2] + delta[2];
  x_plus_delta[3] = q.x();
  x_plus_delta[4] = q.y();
  x_plus_delta[5] = q.z();
  x_plus_delta[6] = q.w();

  DEBUG_CHECK_NEAR(q.norm(), 1.0, 1e-13);

  return true;
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool PoseLocalParameterization::minus(const double* x,
                                      const double* x_plus_delta,
                                      double* delta)
{
  delta[0] = x_plus_delta[0] - x[0];
  delta[1] = x_plus_delta[1] - x[1];
  delta[2] = x_plus_delta[2] - x[2];
  Eigen::Map<const Eigen::Quaterniond> q_plus_delta_(&x_plus_delta[3]);
  Eigen::Map<const Eigen::Quaterniond> q_(&x[3]);
  Eigen::Map<Eigen::Vector3d> omega(&delta[3]);

  Eigen::Quaterniond q_diff = q_plus_delta_ * q_.inverse();

  // Quaternion implementation part copied from GTSAM.
  // define these compile time constants to avoid std::abs:
  static const double twoPi = 2.0 * M_PI, NearlyOne = 1.0 - 1e-10,
  NearlyNegativeOne = -1.0 + 1e-10;

  const double qw = q_diff.w();
  // See Quaternion-Logmap.nb in doc for Taylor expansions
  if (qw > NearlyOne)
  {
    // Taylor expansion of (angle / s) at 1
    // (2 + 2 * (1-qw) / 3) * q.vec();
    omega = ( 8. / 3. - 2. / 3. * qw) * q_diff.vec();
  }
  else if (qw < NearlyNegativeOne)
  {
    // Taylor expansion of (angle / s) at -1
    // (-2 - 2 * (1 + qw) / 3) * q.vec();
    omega = (-8. / 3. - 2. / 3. * qw) * q_diff.vec();
  }
  else
  {
    // Normal, away from zero case
    double angle = 2 * std::acos(qw), s = std::sqrt(1 - qw * qw);
    // Important:  convert to [-pi,pi] to keep error continuous
    if (angle > M_PI)
    {
      angle -= twoPi;
    }
    else if (angle < -M_PI)
    {
      angle += twoPi;
    }
    omega = (angle / s) * q_diff.vec();
  }
  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool PoseLocalParameterization::plusJacobian(const double* x,
                                             double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Jp.setZero();

  // Translational part:
  Jp.topLeftCorner<3, 3>().setIdentity();

  // Rotation:
  // exp(dalpha) x q \approx [dalpha/2; 1] x q
  // \approx ([0 0 0 1]^T + 0.5 * I_3x4 * dalpha) x q
  // \approx oplus(q) * ([0 0 0 1]^T + 0.5 * I_4x3 * dalpha)
  // => derivative wrt dalpha = oplus(q) * 0.5 * I_4x3
  Eigen::Map<const Eigen::Quaterniond> q(&x[3]);
  Jp.bottomRightCorner<4, 3>() = 0.5 * quaternionOplusMatrix(q) *
                                 Eigen::Matrix<double, 4, 3>::Identity();
  return true;
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool PoseLocalParameterization::liftJacobian(const double* x,
                                             double* jacobian)
{
  Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J_lift(jacobian);
  // Translational part.
  J_lift.setZero();
  J_lift.topLeftCorner<3, 3>().setIdentity();

  const Eigen::Quaterniond q_inv(x[6], -x[3], -x[4], -x[5]);
  Eigen::Matrix4d Qplus = quaternionOplusMatrix(q_inv);
  J_lift.bottomRightCorner<3, 4>() = 2.0 * Qplus.topLeftCorner<3, 4>();
  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool PoseLocalParameterization::ComputeJacobian(const double* x,
                                                double* jacobian) const
{
  return plusJacobian(x, jacobian);
}

bool PoseLocalParameterization::VerifyJacobianNumDiff(const double* x,
                                                      double* jacobian,
                                                      double* jacobianNumDiff)
{
  plusJacobian(x, jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jp(jacobian);
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > Jpn(jacobianNumDiff);
  double dx = 1e-9;
  Eigen::Matrix<double, 7, 1> xp;
  Eigen::Matrix<double, 7, 1> xm;
  for (size_t i = 0; i < 6; ++i)
  {
    Eigen::Matrix<double, 6, 1> delta;
    delta.setZero();
    delta[i] = dx;
    Plus(x, delta.data(), xp.data());
    delta[i] = -dx;
    Plus(x, delta.data(), xm.data());
    Jpn.col(i) = (xp - xm) / (2 * dx);
  }
  if ((Jp - Jpn).norm() < 1e-6)
    return true;
  else
    return false;
}

}  // namespace ceres_backend
}  // namespace svo
