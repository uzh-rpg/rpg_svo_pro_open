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
 * @file ErrorInterface.hpp
 * @brief Header file for the ErrorInterface class. A simple interface class that
          other error classes should inherit from.
 * @author Stefan Leutenegger
 */

#pragma once

#include <Eigen/Core>
#include <map>

namespace svo {

namespace ceres_backend {

enum class ErrorType: uint8_t
{
  kHomogeneousPointError,
  kReprojectionError,
  kSpeedAndBiasError,
  kMarginalizationError,
  kPoseError,
  kIMUError,
  kRelativePoseError
};

extern const std::map<ErrorType, std::string> kErrorToStr;

/// @brief Simple interface class the errors implemented here should inherit from.
class ErrorInterface
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// @brief Constructor
  ErrorInterface() = default;
  /// @brief Destructor (does nothing).
  virtual ~ErrorInterface() = default;

  /// @name Sizes
  /// @{

  /// @brief Get dimension of residuals.
  /// @return The residual dimension.
  virtual size_t residualDim() const = 0;

  /// @brief Get the number of parameter blocks this is connected to.
  /// @return The number of parameter blocks.
  virtual size_t parameterBlocks() const = 0;

  /**
   * @brief get the dimension of a parameter block this is connected to.
   * @param parameter_block_idx The index of the parameter block of interest.
   * @return Its dimension.
   */
  virtual size_t parameterBlockDim(size_t parameter_block_idx) const = 0;

  /// @}
  // Error and Jacobian computation
  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobians_minimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(
      double const* const * parameters, double* residuals, double** jacobians,
      double** jacobians_minimal) const = 0;

  /// @brief Residual block type as string
  virtual ErrorType typeInfo() const = 0;
};

}  // namespace ceres_backend
}  // namespace svo
