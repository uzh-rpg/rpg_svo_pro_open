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
 *  Created on: Sep 10, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file SpeedAndBiasError.cpp
 * @brief Source file for the SpeedAndBiasError class.
 * @author Stefan Leutenegger
 */

#include "svo/ceres_backend/speed_and_bias_error.hpp"

namespace svo {
namespace ceres_backend {

// Construct with measurement and information matrix
SpeedAndBiasError::SpeedAndBiasError(const SpeedAndBias& measurement,
                                     const information_t& information)
{
  setMeasurement(measurement);
  setInformation(information);
}

// Construct with measurement and variance.
SpeedAndBiasError::SpeedAndBiasError(const SpeedAndBias& measurement,
                                     double speed_variance,
                                     double gyr_bias_variance,
                                     double acc_bias_variance)
{
  setMeasurement(measurement);

  DEBUG_CHECK_NE(speed_variance, 0.0);
  DEBUG_CHECK_NE(gyr_bias_variance, 0.0);
  DEBUG_CHECK_NE(acc_bias_variance, 0.0);
  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() =
      Eigen::Matrix3d::Identity() * 1.0 / speed_variance;
  information.block<3, 3>(3, 3) =
      Eigen::Matrix3d::Identity() * 1.0 / gyr_bias_variance;
  information.bottomRightCorner<3, 3>() =
      Eigen::Matrix3d::Identity() * 1.0 / acc_bias_variance;
    setInformation(information);
}

// Set the information.
void SpeedAndBiasError::setInformation(const information_t& information)
{
  information_ = information;
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  square_root_information_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool SpeedAndBiasError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, nullptr);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool SpeedAndBiasError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobians_minimal) const
{
  // compute error
  Eigen::Map<const SpeedAndBias> estimate(parameters[0]);
  SpeedAndBias error = measurement_ - estimate;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 9, 1> > weighted_error(residuals);
  weighted_error = square_root_information_ * error;

  // compute Jacobian - this is rather trivial in this case...
  if (jacobians != nullptr && jacobians[0] != nullptr)
  {
    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> > J0(jacobians[0]);
    J0 = -square_root_information_ * Eigen::Matrix<double, 9, 9>::Identity();
  }
  if (jacobians_minimal != nullptr && jacobians_minimal[0] != nullptr)
  {
    Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor> >
        J0min(jacobians_minimal[0]);
    J0min = -square_root_information_ * Eigen::Matrix<double, 9, 9>::Identity();
  }

  return true;
}

}  // namespace ceres_backend
}  // namespace svo
