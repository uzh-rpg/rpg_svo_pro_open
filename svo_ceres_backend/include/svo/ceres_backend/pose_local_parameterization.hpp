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
 * @file PoseLocalParameterization.hpp
 * @brief Header file for the PoseLocalParemerization class.
 * @author Stefan Leutenegger
 */

#pragma once

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <ceres/ceres.h>
#pragma diagnostic pop

#include "svo/ceres_backend/local_parameterization_additional_interfaces.hpp"

namespace svo {
namespace ceres_backend {

/// \brief Pose local parameterisation, i.e. for orientation dq(dalpha) x q_bar.
/// Order:
/// x[0]-x[2] translation
/// x[3]-x[6] quaternion xyzw
/// Perturbation:
/// dx[0] - dx[2] translation
/// dx[3] - dx[5] axis angle perturbation
class PoseLocalParameterization :
    public ceres::LocalParameterization,
    public LocalParamizationAdditionalInterfaces
{
 public:

  /// \brief Trivial destructor.
  virtual ~PoseLocalParameterization() = default;

  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x Variable.
  /// @param[in] delta Perturbation.
  /// @param[out] x_plus_delta Perturbed x.
  virtual bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const;

  /// \brief Computes the minimal difference between a variable x and a
  ///        perturbed variable x_plus_delta.
  /// @param[in] x Variable.
  /// @param[in] x_plus_delta Perturbed variable.
  /// @param[out] delta minimal difference.
  /// \return True on success.
  virtual bool Minus(const double* x, const double* x_plus_delta,
                     double* delta) const;

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x Variable.
  /// @param[out] jacobian The Jacobian.
  virtual bool ComputeJacobian(const double* x, double* jacobian) const;

  /// \brief Computes the Jacobian from minimal space to naively overparameterised
  /// space as used by ceres. It is the inverse of the plusJacobian.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  virtual bool ComputeLiftJacobian(const double* x, double* jacobian) const;

  // provide these as static for easy use elsewhere:

  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x Variable.
  /// @param[in] delta Perturbation.
  /// @param[out] x_plus_delta Perturbed x.
  static bool plus(const double* x, const double* delta, double* x_plus_delta);

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x Variable.
  /// @param[out] jacobian The Jacobian.
  static bool plusJacobian(const double* x, double* jacobian);

  /// \brief Computes the minimal difference between a variable x and a
  ///        perturbed variable x_plus_delta
  /// @param[in] x Variable.
  /// @param[in] x_plus_delta Perturbed variable.
  /// @param[out] delta minimal difference.
  /// \return True on success.
  static bool minus(const double* x, const double* x_plus_delta, double* delta);

  /// \brief Computes the Jacobian from minimal space to naively
  ///        overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double* x, double* jacobian);

  /// \brief The parameter block dimension.
  virtual int GlobalSize() const
  {
    return 7;
  }

  /// \brief The parameter block local dimension.
  virtual int LocalSize() const
  {
    return 6;
  }

  // added convenient check
  bool VerifyJacobianNumDiff(const double* x, double* jacobian,
                             double* jacobianNumDiff);
};

}  // namespace ceres_backend
}  // namespace svo
