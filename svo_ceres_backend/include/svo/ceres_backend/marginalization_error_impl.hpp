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
 *  Created on: Sep 11, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file implementation/MarginalizationError.hpp
 * @brief Header implementation file for the MarginalizationError class.
 * @author Stefan Leutenegger
 */

#pragma once

#include "svo/ceres_backend/marginalization_error.hpp"

namespace svo {
namespace ceres_backend {

// Split for Schur complement op.
template<typename Derived_A, typename Derived_U, typename Derived_W,
    typename Derived_V>
void MarginalizationError::splitSymmetricMatrix(
    const std::vector<std::pair<int, int> >& marginalization_start_idx_and_length_pairs,
    const Eigen::MatrixBase<Derived_A>& A,  // input
    const Eigen::MatrixBase<Derived_U>& U,  // output
    const Eigen::MatrixBase<Derived_W>& W,  // output
    const Eigen::MatrixBase<Derived_V>& V)  // output
{
  // sanity check
  const int size = A.cols();
  DEBUG_CHECK(size == A.rows()) << "matrix not symmetric";
  DEBUG_CHECK(V.cols() == V.rows()) << "matrix not symmetric";
  DEBUG_CHECK(V.cols() == W.cols()) << "matrix not symmetric";
  DEBUG_CHECK(U.rows() == W.rows()) << "matrix not symmetric";
  DEBUG_CHECK(V.rows() + U.rows() == size)
      << "matrices supplied to be split into do not form exact upper triangular "
      << "blocks of the original one";
  DEBUG_CHECK(U.cols() == U.rows()) << "matrix not symmetric";

  //! @todo i'm sure this copy can be avoided as well..
  std::vector<std::pair<int, int> > marginalization_start_idx_and_length_pairs2 =
      marginalization_start_idx_and_length_pairs;
  marginalization_start_idx_and_length_pairs2.push_back(
      std::pair<int, int>(size, 0));

  const size_t length = marginalization_start_idx_and_length_pairs2.size();

  int last_idx_row = 0;
  int start_a_i = 0;
  int start_b_i = 0;
  for (size_t i = 0; i < length; ++i)
  {
    int last_idx_col = 0;
    int start_a_j = 0;
    int start_b_j = 0;
    int this_idx_row = marginalization_start_idx_and_length_pairs2[i].first;
    const int size_b_i = marginalization_start_idx_and_length_pairs2[i].second;  // kept
    const int size_a_i = this_idx_row - last_idx_row;  // kept
    for (size_t j = 0; j < length; ++j)
    {
      int this_idx_col = marginalization_start_idx_and_length_pairs2[j].first;
      const int size_a_j = this_idx_col - last_idx_col;  // kept
      const int size_b_j = marginalization_start_idx_and_length_pairs2[j].second;  // kept

      // the kept part - only access and copy if non-zero
      if (size_a_j > 0 && size_a_i > 0)
      {
        const_cast<Eigen::MatrixBase<Derived_U>&>(U).block(start_a_i, start_a_j,
                                                           size_a_i, size_a_j) =
            A.block(last_idx_row, last_idx_col, size_a_i, size_a_j);
      }

      // now the mixed part
      if (size_b_j > 0 && size_a_i > 0)
      {
        const_cast<Eigen::MatrixBase<Derived_W>&>(W).block(start_a_i, start_b_j,
                                                           size_a_i, size_b_j) =
            A.block(last_idx_row, this_idx_col, size_a_i, size_b_j);
      }

      // and finally the marginalized part
      if (size_b_j > 0 && size_b_i > 0)
      {
        const_cast<Eigen::MatrixBase<Derived_V>&>(V).block(start_b_i, start_b_j,
                                                           size_b_i, size_b_j) =
            A.block(this_idx_row, this_idx_col, size_b_i, size_b_j);
      }

      last_idx_col = this_idx_col + size_b_j;  // remember
      start_a_j += size_a_j;
      start_b_j += size_b_j;
    }
    last_idx_row = this_idx_row + size_b_i;  // remember
    start_a_i += size_a_i;
    start_b_i += size_b_i;
  }
}

// Split for Schur complement op.
template<typename Derived_b, typename Derived_b_a, typename Derived_b_b>
void MarginalizationError::splitVector(
    const std::vector<std::pair<int, int> >& marginalization_start_idx_and_length_pairs,
    const Eigen::MatrixBase<Derived_b>& b,  // input
    const Eigen::MatrixBase<Derived_b_a>& b_a,  // output
    const Eigen::MatrixBase<Derived_b_b>& b_b) // output
{
  const int size = b.rows();
  // sanity check
  DEBUG_CHECK(b.cols() == 1) << "supplied vector not x-by-1";
  DEBUG_CHECK(b_a.cols() == 1) << "supplied vector not x-by-1";
  DEBUG_CHECK(b_b.cols() == 1) << "supplied vector not x-by-1";
  DEBUG_CHECK(b_a.rows() + b_b.rows() == size)
      << "vector supplied to be split into cannot be concatenated to the original one";

  //! @todo remove this copy. If the last n entries of b are to be kept deal
  //! with this some other way. E.g. after loop. if (lastidx != size) ...
  std::vector<std::pair<int, int> > marginalization_start_idx_and_length_pairs2 =
      marginalization_start_idx_and_length_pairs;
  marginalization_start_idx_and_length_pairs2.push_back(
      std::pair<int, int>(size, 0));

  const size_t length = marginalization_start_idx_and_length_pairs2.size();

  int last_idx_row = 0;
  int start_a_i = 0;
  int start_b_i = 0;
  for (size_t i = 0; i < length; ++i)
  {
    int this_idx_row = marginalization_start_idx_and_length_pairs2[i].first;
    const int size_b_i = marginalization_start_idx_and_length_pairs2[i].second;  // marginalized
    const int size_a_i = this_idx_row - last_idx_row;  // kept

    // the kept part - only access and copy if non-zero
    if (size_a_i > 0)
    {
      const_cast<Eigen::MatrixBase<Derived_b_a>&>(b_a).segment(start_a_i,
                                                               size_a_i) = b
          .segment(last_idx_row, size_a_i);
    }

    // and finally the marginalized part
    if (size_b_i > 0)
    {
      const_cast<Eigen::MatrixBase<Derived_b_b>&>(b_b).segment(start_b_i,
                                                               size_b_i) = b
          .segment(this_idx_row, size_b_i);
    }

    last_idx_row = this_idx_row + size_b_i;  // remember
    start_a_i += size_a_i;
    start_b_i += size_b_i;
  }
}

// Pseudo inversion of a symmetric matrix.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric
// positive semi-definite (negative Eigenvalues are set to zero)
template<typename Derived>
bool MarginalizationError::pseudoInverseSymm(
    const Eigen::MatrixBase<Derived>&a, const Eigen::MatrixBase<Derived>&result,
    double epsilon, int * rank)
{
  DEBUG_CHECK(a.rows() == a.cols()) << "matrix supplied is not quadratic";

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
      * Eigen::VectorXd(
          (saes.eigenvalues().array() > tolerance).select(
              saes.eigenvalues().array().inverse(), 0)).asDiagonal()
      * (saes.eigenvectors().transpose());

  if (rank)
  {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i)
    {
      if (saes.eigenvalues()[i] > tolerance)
      {
        (*rank)++;
      }
    }
  }

  return true;
}

// Pseudo inversion and square root (Cholesky decomposition) of a symmetric matrix.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric
// positive semi-definite (negative Eigenvalues are set to zero)
template<typename Derived>
bool MarginalizationError::pseudoInverseSymmSqrt(
    const Eigen::MatrixBase<Derived>&a, const Eigen::MatrixBase<Derived>&result,
    double epsilon, int * rank)
{

  DEBUG_CHECK(a.rows() == a.cols()) << "matrix supplied is not quadratic";

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
      * Eigen::VectorXd(
          Eigen::VectorXd(
              (saes.eigenvalues().array() > tolerance).select(
                  saes.eigenvalues().array().inverse(), 0)).array().sqrt())
          .asDiagonal();

  if (rank)
  {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i)
    {
      if (saes.eigenvalues()[i] > tolerance)
      {
        (*rank)++;
      }
    }
  }

  return true;
}

// Block-wise pseudo inversion of a symmetric matrix with non-zero diagonal blocks.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric
// positive semi-definite (negative Eigenvalues are set to zero)
template<typename Derived, int blockDim>
void MarginalizationError::blockPinverse(
    const Eigen::MatrixBase<Derived>& M_in,
    const Eigen::MatrixBase<Derived>& M_out, double epsilon)
{
  DEBUG_CHECK(M_in.rows() == M_in.cols())
      << "matrix supplied is not quadratic";

  const_cast<Eigen::MatrixBase<Derived>&>(M_out).resize(M_in.rows(),
                                                        M_in.rows());
  const_cast<Eigen::MatrixBase<Derived>&>(M_out).setZero();
  for (int i = 0; i < M_in.cols(); i += blockDim)
  {
    Eigen::Matrix<double, blockDim, blockDim> inv;
    const Eigen::Matrix<double, blockDim, blockDim> in = M_in
        .template block<blockDim, blockDim>(i, i);
    //const Eigen::Matrix<double,blockDim,blockDim> in1=0.5*(in+in.transpose());
    pseudoInverseSymm(in, inv, epsilon);
    const_cast<Eigen::MatrixBase<Derived>&>(M_out)
        .template block<blockDim, blockDim>(i, i) = inv;
  }
}

// Block-wise pseudo inversion and square root (Cholesky decomposition)
// of a symmetric matrix with non-zero diagonal blocks.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric
// positive semi-definite (negative Eigenvalues are set to zero)
template<typename Derived, int blockDim>
void MarginalizationError::blockPinverseSqrt(
    const Eigen::MatrixBase<Derived>& M_in,
    const Eigen::MatrixBase<Derived>& M_out, double epsilon)
{
  DEBUG_CHECK(M_in.rows() == M_in.cols())
      << "matrix supplied is not quadratic";

  const_cast<Eigen::MatrixBase<Derived>&>(M_out).resize(M_in.rows(),
                                                        M_in.rows());
  const_cast<Eigen::MatrixBase<Derived>&>(M_out).setZero();
  for (int i = 0; i < M_in.cols(); i += blockDim)
  {
    Eigen::Matrix<double, blockDim, blockDim> inv;
    const Eigen::Matrix<double, blockDim, blockDim> in = M_in
        .template block<blockDim, blockDim>(i, i);
    //const Eigen::Matrix<double,blockDim,blockDim> in1=0.5*(in+in.transpose());
    pseudoInverseSymmSqrt(in, inv, epsilon);
    const_cast<Eigen::MatrixBase<Derived>&>(M_out)
        .template block<blockDim, blockDim>(i, i) = inv;
  }
}

} // namespace ceres_backend
} // namespace svo
