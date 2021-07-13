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
 *  Created on: Sep 12, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file RelativePoseError.cpp
 * @brief Source file for the RelativePoseError class.
 * @author Stefan Leutenegger
 */

#include "svo/ceres_backend/relative_pose_error.hpp"

#include <svo/vio_common/matrix_operations.hpp>

#include "svo/ceres_backend/estimator_types.hpp"
#include "svo/ceres_backend/pose_local_parameterization.hpp"

namespace svo {
namespace ceres_backend {

// Construct with measurement and information matrix.
RelativePoseError::RelativePoseError(
    const Eigen::Matrix<double, 6, 6>& information)
{
  setInformation(information);
}

// Construct with measurement and variance.
RelativePoseError::RelativePoseError(double translationVariance,
                                     double rotationVariance)
{

  DEBUG_CHECK_GT(translationVariance, 0.0);
  DEBUG_CHECK_GT(rotationVariance, 0.0);
  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() *
                                      1.0 / translationVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() *
                                          1.0 / rotationVariance;
  setInformation(information);
}

// Set the information.
void RelativePoseError::setInformation(const information_t& information)
{
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  square_root_information_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool RelativePoseError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool RelativePoseError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobians_minimal) const
{
  // compute error
  Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]));
  Transformation T_WS_1(
      Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]),
      Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4],
                         parameters[1][5]));
  // delta pose
  Transformation dp = T_WS_1 * T_WS_0.inverse();
  // get the error
  Eigen::Matrix<double, 6, 1> error;
  const Eigen::Vector3d dtheta = 2 * dp.getRotation().imaginary();
  error.head<3>() = T_WS_1.getPosition() - T_WS_0.getPosition();
  error.tail<3>() = dtheta;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 6, 1> > weighted_error(residuals);
  weighted_error = square_root_information_ * error;

  // compute Jacobian...
  if (jacobians != nullptr)
  {
    if (jacobians[0] != nullptr)
    {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J0(jacobians[0]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal.setIdentity();
      J0_minimal *= -1.0;
      J0_minimal.block<3, 3>(3, 3) =
          -quaternionPlusMatrix(dp.getEigenQuaternion()).block<3, 3>(0, 0);
      J0_minimal = (square_root_information_ * J0_minimal).eval();

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[0], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      J0 = J0_minimal * J_lift;

      if (jacobians_minimal != nullptr && jacobians_minimal[0] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> >
            J0_minimal_mapped(jacobians_minimal[0]);
        J0_minimal_mapped = J0_minimal;
      }
    }
    if (jacobians[1] != nullptr)
    {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J1(jacobians[1]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J1_minimal;
      J1_minimal.setIdentity();
      J1_minimal.block<3, 3>(3, 3) =
          quaternionOplusMatrix(dp.getEigenQuaternion()).block<3, 3>(0, 0);
      J1_minimal = (square_root_information_ * J1_minimal).eval();

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[1], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      J1 = J1_minimal * J_lift;

      if (jacobians_minimal != nullptr && jacobians_minimal[1] != nullptr)
      {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> >
            J1_minimal_mapped(jacobians_minimal[1]);
        J1_minimal_mapped = J1_minimal;
      }
    }
  }

  return true;
}

}  // namespace ceres_backend
}  // namespace svo

