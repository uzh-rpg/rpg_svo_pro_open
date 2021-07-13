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
 * @file ceres/MarginalizationError.hpp
 * @brief Header file for the MarginalizationError class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <memory>
#include <vector>

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <ceres/ceres.h>
#include <Eigen/Core>
#pragma diagnostic pop

#include <svo/vio_common/logging.hpp>

#include "svo/ceres_backend/error_interface.hpp"
#include "svo/ceres_backend/homogeneous_point_parameter_block.hpp"
#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"
#include "svo/ceres_backend/speed_and_bias_parameter_block.hpp"

namespace svo {
namespace ceres_backend {

// not sized, in order to be flexible.
class MarginalizationError : public ceres::CostFunction, public ErrorInterface
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base class type.
  typedef ceres::CostFunction base_t;

  /// \brief Trivial destructor.
  virtual ~MarginalizationError() = default;

  /// \brief Default constructor. Initialises a new ceres_backend::Map.
  MarginalizationError();

  /// \brief Constructor from ceres_backend::Map.
  /// @param[in] map The underlying ceres_backend::Map.
  MarginalizationError(Map& map);

  /// \brief Constructor from ceres_backend::Map and directly add some residuals
  /// @param[in] map The underlying ceres_backend::Map.
  /// @param[in] residual_block_ids Residual block IDs to be added directly
  ///            (\see ceres_backend::addResidualBlocks)
  MarginalizationError(
      Map& map, std::vector< ceres::ResidualBlockId >& residual_block_ids);

  // initialization
  /// \brief Set the underlying ceres_backend::Map.
  /// @param[in] map The underlying ceres_backend::Map.
  void setMap(Map& map);

  /// \brief Add some residuals to this marginalisation error. This means, they
  ///        will get linearised.
  /// \warning Note that once added here, they will be removed from the
  ///          ceres_backend::Map and stay linerised at exactly the points passed here.
  /// @param[in] residual_block_ids Vector of residual block ids, the
  ///            corresponding terms of which will be added.
  /// @param[in] keep_residual_blocks Currently not in use.
  bool addResidualBlocks(
      const std::vector< ceres::ResidualBlockId > & residual_block_ids,
      const std::vector<bool> & keep_residual_blocks = std::vector<bool>());

  /// \brief Add one residual to this marginalisation error. This means, it will
  ///        get linearised.
  /// \warning Note that once added here, it will be removed from the
  ///          ceres_backend::Map and stay linearised at exactly the point passed here.
  /// @param[in] residual_block_id Residual block id, the corresponding term of
  ///            which will be added.
  /// @param[in] keep Currently not in use.
  bool addResidualBlock(ceres::ResidualBlockId residual_block_id,
                        bool keep = false);

  /// \brief Info: is this parameter block connected to this marginalization error?
  /// @param[in] parameter_block_id Parameter block id of interest.
  bool isParameterBlockConnected(uint64_t parameter_block_id);

  // marginalization

  /// \brief Marginalise out a set of parameter blocks.
  /// \warning The parameter blocks to be marginalised must have been added
  ///          before. Also: make sure to add all the needed residual blocks
  ///          beforehand for this to make sense.
  /// \return False if not all necessary residual blocks were added before.
  bool marginalizeOut(const std::vector<uint64_t>& parameter_block_ids,
                      const std::vector<bool>& keep_parameter_blocks =
                          std::vector<bool>());

  /// \brief This must be called before optimization after adding residual
  ///        blocks and/or marginalizing, since it performs all the lhs and rhs
  ///        computations on from a given _H and _b.
  void updateErrorComputation();

  /// \brief Call this in order to (re-)add this error term after whenever it
  ///             had been modified.
  /// @param[in] parameter_block_ptrs Parameter block pointers in question.
  void getParameterBlockPtrs(
      std::vector<std::shared_ptr<ceres_backend::ParameterBlock> >& parameter_block_ptrs);

  // error term and Jacobian implementation (inherited pure virtuals from ceres::CostFunction)
  /**
    * @brief This evaluates the error term and additionally computes the Jacobians.
    * @param parameters Pointer to the parameters (see ceres)
    * @param residuals Pointer to the residual vector (see ceres)
    * @param jacobians Pointer to the Jacobians (see ceres)
    * @return success of th evaluation.
    */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobians_minimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobians_minimal) const;

  // sizes (inherited pure virtuals from ::ceres_backend::ErrorInterface)
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return base_t::num_residuals();
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const
  {
    return base_t::parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameter_block_idx ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameter_block_idx) const
  {
    return base_t::parameter_block_sizes().at(parameter_block_idx);
  }

  /// @brief Return parameter block type as string
  virtual ErrorType typeInfo() const
  {
    return ErrorType::kMarginalizationError;
  }


  /**
   * @brief Pseudo inversion of a symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric
   *            positive semi-definite (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix
   * @param[out] result Output, i.e. pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @param[out] rank Optional rank.
   * @return
   */
  template<typename Derived>
  static bool pseudoInverseSymm(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(),
      int * rank = 0);

  /**
   * @brief Pseudo inversion and square root (Cholesky decomposition) of a
   *        symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric
   *            positive semi-definite (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix
   * @param[out] result Output, i.e. the Cholesky decomposition of a pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @param[out] rank The rank, if of interest.
   * @return
   */
  template<typename Derived>
  static bool pseudoInverseSymmSqrt(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(),
      int* rank = NULL);

  /**
   * @brief Block-wise pseudo inversion of a symmetric matrix with non-zero
   *        diagonal blocks.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric
   *            positive semi-definite (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @tparam blockDim The block size of the diagonal blocks.
   * @param[in] M_in Input Matrix
   * @param[out] M_out Output, i.e. thepseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @return
   */
  template<typename Derived, int blockDim>
  static void blockPinverse(
      const Eigen::MatrixBase<Derived>& M_in,
      const Eigen::MatrixBase<Derived>& M_out, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon());


  /**
   * @brief Block-wise pseudo inversion and square root (Cholesky decomposition)
   *        of a symmetric matrix with non-zero diagonal blocks.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric
   *            positive semi-definite (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @tparam blockDim The block size of the diagonal blocks.
   * @param[in] M_in Input Matrix
   * @param[out] M_out Output, i.e. the Cholesky decomposition of a pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @return
   */
  template<typename Derived, int blockDim>
  static void blockPinverseSqrt(
      const Eigen::MatrixBase<Derived>& M_in,
      const Eigen::MatrixBase<Derived>& M_out, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon());

  /**
   * @brief check if parameter block with id is in marginalization term and
   *        hence check if it can be marginalized.
   * @param param_block_id the ID of the parameter block
   * @return true if the parameter block is contained, false otherwise
   */
  bool isInMarginalizationTerm(uint64_t param_block_id)
  {
    return parameter_block_id_to_parameter_block_info_idx_.find(param_block_id)
        != parameter_block_id_to_parameter_block_info_idx_.end();
  }

 protected:
  Map* map_ptr_; ///< The underlying map.
  ceres::ResidualBlockId residual_block_id_; ///< The residual block id of this.

  /// \brief Checks the internal datastructure (debug)
  void check();

  /// \brief Computes the linearized deviation from the references (linearization points)
  bool computeDeltaChi(Eigen::VectorXd& DeltaChi) const;  // use the stored estimates

  /// \brief Computes the linearized deviation from the references (linearization points)
  bool computeDeltaChi(double const* const * parameters,
                       Eigen::VectorXd& DeltaChi) const;  // use the provided estimates

  /// \brief Split for Schur complement op.
  template<typename Derived_A, typename Derived_U, typename Derived_W,
      typename Derived_V>
  static void splitSymmetricMatrix(
      const std::vector<std::pair<int, int> >& marginalization_start_idx_and_length_pairs,
      const Eigen::MatrixBase<Derived_A>& A,  // input
      const Eigen::MatrixBase<Derived_U>& U,  // output
      const Eigen::MatrixBase<Derived_W>& W,  // output
      const Eigen::MatrixBase<Derived_V>& V);  // output

  /// \brief Split for Schur complement op.
  template<typename Derived_b, typename Derived_b_a, typename Derived_b_b>
  static void splitVector(
      const std::vector<std::pair<int, int> >& marginalization_start_idx_and_length_pairs,
      const Eigen::MatrixBase<Derived_b>& b,  // input
      const Eigen::MatrixBase<Derived_b_a>& b_a,  // output
      const Eigen::MatrixBase<Derived_b_b>& b_b);  // output

  /// @name The internal storage of the linearised system.
  /// lhs and rhs:
  /// H_*delta_Chi = _b - H_*Delta_Chi .
  /// the lhs Hessian matrix is decomposed as _H = J^T*J = _U*S*_U^T ,
  /// the rhs is decomposed as
  /// _b - _H*Delta_Chi = -J^T * (-pinv(J^T) * _b + J*Delta_Chi) ,
  /// i.e. we have the ceres standard form with weighted Jacobians _J,
  /// an identity information matrix, and an error
  /// _e = -pinv(J^T) * _b + J*Delta_Chi .
  /// _e = _e0 + J*Delta_Chi .
  /// @{
  Eigen::MatrixXd H_;  ///< lhs - Hessian
  Eigen::VectorXd b0_;  ///<  rhs constant part
  Eigen::VectorXd e0_;  ///<  _e0 := pinv(J^T) * _b0
  Eigen::MatrixXd J_;  ///<  Jacobian such that _J^T * J == _H
  Eigen::MatrixXd U_;  ///<  H_ = _U*_S*_U^T lhs Eigen decomposition
  //! @todo are the following members even used anywhere?
  Eigen::VectorXd S_;  ///<  singular values
  Eigen::VectorXd S_sqrt_;
  ///<  cwise sqrt of _S, i.e. _S_sqrt*_S_sqrt=_S; _J=_U^T*_S_sqrt
  Eigen::VectorXd S_pinv_;  ///<  pseudo inverse of _S
  Eigen::VectorXd S_pinv_sqrt_;
  ///<  cwise sqrt of _S_pinv, i.e. pinv(J^T)=_U^T*_S_pinv_sqrt
  Eigen::VectorXd p_;
  Eigen::VectorXd p_inv_;
  volatile bool error_computation_valid_;
  ///<  adding residual blocks will invalidate this. before optimizing,
  ///   call updateErrorComputation()

  /// \brief Book-keeping of the ordering.
  struct ParameterBlockInfo
  {
    uint64_t parameter_block_id;
    std::shared_ptr<ParameterBlock> parameter_block_ptr;
    size_t ordering_idx;
    size_t dimension;
    size_t minimal_dimension;
    size_t local_dimension;
    std::shared_ptr<double> linearization_point;
    bool is_landmark;
    ParameterBlockInfo()
        : parameter_block_id(0),
          parameter_block_ptr(std::shared_ptr<ParameterBlock>()),
          ordering_idx(0),
          dimension(0),
          minimal_dimension(0),
          local_dimension(0),
          is_landmark(false)
    {
    }
    ParameterBlockInfo(uint64_t parameter_block_id,
                       std::shared_ptr<ParameterBlock> parameter_block_ptr,
                       size_t orderingIdx, bool is_landmark)
        : parameter_block_id(parameter_block_id),
          parameter_block_ptr(parameter_block_ptr),
          ordering_idx(orderingIdx),
          is_landmark(is_landmark)
    {
      dimension = parameter_block_ptr->dimension();
      minimal_dimension = parameter_block_ptr->minimalDimension();
      if (parameter_block_ptr->localParameterizationPtr())
      {
        local_dimension = parameter_block_ptr->localParameterizationPtr()
            ->LocalSize();
      }
      else
      {
        local_dimension = minimal_dimension;
      }
      if (parameter_block_ptr->fixed())
      {
        minimal_dimension = 0;
        local_dimension = 0;
      }
      linearization_point.reset(new double[dimension],
                               std::default_delete<double[]>());
      memcpy(linearization_point.get(), parameter_block_ptr->parameters(),
             dimension * sizeof(double));
    }

    /// \brief Reset the linearisation point. Use with caution.
    void resetLinearizationPoint(
        std::shared_ptr<ParameterBlock> parameter_block_ptr)
    {
      DEBUG_CHECK(dimension==parameter_block_ptr->dimension())
          << "not initialised.";
      memcpy(linearization_point.get(), parameter_block_ptr->parameters(),
             dimension * sizeof(double));
    }
  };

  std::vector<ParameterBlockInfo> parameter_block_infos_;  ///< Book keeper.
  std::map<uint64_t, size_t> parameter_block_id_to_parameter_block_info_idx_;
  ///< Maps parameter block Ids to index in _parameterBlockInfos
  size_t dense_indices_;
  ///< Keep track of the size of the dense part of the equation system

  /// @}

};

}  // namespace ceres_backend
}  // namespace svo

#include <svo/ceres_backend/marginalization_error_impl.hpp>
