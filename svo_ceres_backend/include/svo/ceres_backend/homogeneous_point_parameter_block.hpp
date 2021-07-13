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
 * @file HomogeneousPointParameterBlock.hpp
 * @brief Header file for the HomogeneousPointParameterBlock class.
 * @author Stefan Leutenegger
 */

#pragma once

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
#include <Eigen/Core>
#pragma diagnostic pop

#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/parameter_block.hpp"

namespace svo {
namespace ceres_backend {

/// \brief Wraps the parameter block for a speed / IMU biases estimate
class HomogeneousPointParameterBlock : public ParameterBlock
{
 public:

  /// \brief The estimate type (4D vector).
  typedef Eigen::Vector4d estimate_t;

  static constexpr size_t c_dimension = 4;
  static constexpr size_t c_minimal_dimension = 3;

  /// \brief Default constructor (assumes not fixed).
  HomogeneousPointParameterBlock();

  /// \brief Constructor with estimate and time.
  /// @param[in] point The homogeneous point estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  HomogeneousPointParameterBlock(const Eigen::Vector4d& point, uint64_t id,
                                 bool initialized = true);

  /// \brief Constructor with estimate and time.
  /// @param[in] point The homogeneous point estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  HomogeneousPointParameterBlock(const Eigen::Vector3d& point, uint64_t id,
                                 bool initialized = true);

  virtual ~HomogeneousPointParameterBlock() {}


  // ---------------------------------------------------------------------------
  // Setters.

  virtual void setEstimate(const Eigen::Vector4d& point) { estimate_ = point; }

  /// \brief Set initialisaiton status.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  void setInitialized(bool initialized)
  {
    initialized_ = initialized;
  }

  // ---------------------------------------------------------------------------
  // Getters

  virtual const Eigen::Vector4d& estimate() const { return estimate_; }

  /// \brief Get initialisaiton status.
  /// \return Whether or not the 3d position is considered initialised.
  bool initialized() const { return initialized_; }

  virtual double* parameters() { return estimate_.data(); }

  virtual const double* parameters() const { return estimate_.data(); }

  virtual size_t dimension() const { return c_dimension; }

  virtual size_t minimalDimension() const { return c_minimal_dimension; }

  // minimal internal parameterization
  // x0_plus_Delta=Delta_Chi[+]x0
  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x0 Variable.
  /// @param[in] Delta_Chi Perturbation.
  /// @param[out] x0_plus_Delta Perturbed x.
  virtual void plus(const double* x0, const double* Delta_Chi,
                    double* x0_plus_Delta) const
  {
    HomogeneousPointLocalParameterization::plus(x0, Delta_Chi, x0_plus_Delta);
  }

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian The Jacobian.
  virtual void plusJacobian(const double* x0, double* jacobian) const
  {
    HomogeneousPointLocalParameterization::plusJacobian(x0, jacobian);
  }

  // Delta_Chi=x0_plus_Delta[-]x0
  /// \brief Computes the minimal difference between a variable x and a
  ///        perturbed variable x_plus_delta
  /// @param[in] x0 Variable.
  /// @param[in] x0_plus_Delta Perturbed variable.
  /// @param[out] Delta_Chi Minimal difference.
  /// \return True on success.
  virtual void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const
  {
    HomogeneousPointLocalParameterization::minus(x0, x0_plus_Delta, Delta_Chi);
  }

  /// \brief Computes the Jacobian from minimal space to naively
  ///        overparameterised space as used by ceres.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  virtual void liftJacobian(const double* x0, double* jacobian) const
  {
    HomogeneousPointLocalParameterization::liftJacobian(x0, jacobian);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const
  {
    return "HomogeneousPointParameterBlock";
  }

 private:
  Eigen::Vector4d estimate_;
  bool initialized_;  ///< Whether or not the 3d position is considered initialised.
};

}  // namespace ceres_backend
}  // namespace svo
