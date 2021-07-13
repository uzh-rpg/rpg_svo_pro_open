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
 * @file MarginalizationError.cpp
 * @brief Source file for the MarginalizationError class.
 * @author Stefan Leutenegger
 */

#include "svo/ceres_backend/marginalization_error.hpp"

#include <functional>

#include <svo/vio_common/logging.hpp>

#include "svo/ceres_backend/local_parameterization_additional_interfaces.hpp"

//#define USE_NEW_LINEARIZATION_POINT

namespace svo {
namespace ceres_backend {

inline void conservativeResize(Eigen::MatrixXd& matrixXd, int rows, int cols)
{
  Eigen::MatrixXd tmp(rows, cols);
  const int common_rows = std::min(rows, (int) matrixXd.rows());
  const int common_cols = std::min(cols, (int) matrixXd.cols());
  tmp.topLeftCorner(common_rows, common_cols) = matrixXd.topLeftCorner(
      common_rows, common_cols);
  matrixXd.swap(tmp);
}

inline void conservativeResize(Eigen::VectorXd& vectorXd, int size)
{
  if (vectorXd.rows() == 1)
  {
    Eigen::VectorXd tmp(size);  //Eigen::VectorXd tmp = Eigen::VectorXd::Zero(size,Eigen::RowMajor);
    const int common_size = std::min((int) vectorXd.cols(), size);
    tmp.head(common_size) = vectorXd.head(common_size);
    vectorXd.swap(tmp);
  }
  else
  {
    Eigen::VectorXd tmp(size);  //Eigen::VectorXd tmp = Eigen::VectorXd::Zero(size);
    const int common_size = std::min((int) vectorXd.rows(), size);
    tmp.head(common_size) = vectorXd.head(common_size);
    vectorXd.swap(tmp);
  }
}

// Default constructor. Initialises a new ceres_backend::Map.
MarginalizationError::MarginalizationError()
{
  map_ptr_ = 0;
  dense_indices_ = 0;
  residual_block_id_ = 0;
  error_computation_valid_ = false;
}

// Default constructor from ceres_backend::Map.
MarginalizationError::MarginalizationError(Map& map)
{
  setMap(map);
  dense_indices_ = 0;
  residual_block_id_ = 0;
  error_computation_valid_ = false;
}

MarginalizationError::MarginalizationError(
    Map& map, std::vector< ceres::ResidualBlockId> & residual_block_ids)
{
  setMap(map);
  dense_indices_ = 0;
  residual_block_id_ = 0;
  error_computation_valid_ = false;
  bool success = addResidualBlocks(residual_block_ids);
  CHECK(success)
      << "residual blocks supplied or their connected parameter blocks were not properly added to the map";
}

// Set the underlying ceres_backend::Map.
void MarginalizationError::setMap(Map& map)
{
  map_ptr_ = &map;
  residual_block_id_ = 0;  // reset.
}

// Add some residuals to this marginalisation error. This means, they will get linearised.
bool MarginalizationError::addResidualBlocks(
    const std::vector< ceres::ResidualBlockId> & residual_block_ids,
    const std::vector<bool>& keep_residual_blocks)
{
  // add one block after the other
  for (size_t i = 0; i < residual_block_ids.size(); ++i)
  {
    bool keep = false;
    if (keep_residual_blocks.size() == residual_block_ids.size())
    {
      keep = keep_residual_blocks[i];
    }
    if (!addResidualBlock(residual_block_ids[i], keep))
    {
      return false;
    }
  }
  return true;
}

// Add some residuals to this marginalisation error. This means, they will get linearised.
bool MarginalizationError::addResidualBlock(
    ceres::ResidualBlockId residual_block_id, bool keep)
{
  // get the residual block & check
  std::shared_ptr<ErrorInterface> error_interface_ptr =
      map_ptr_->errorInterfacePtr(residual_block_id);
  DEBUG_CHECK(error_interface_ptr)
      << "residual block id does not exist.";
  if (error_interface_ptr == nullptr)
  {
    return false;
  }

  error_computation_valid_ = false;  // flag that the error computation is invalid

  // get the parameter blocks
  Map::ParameterBlockCollection parameters = map_ptr_->parameters(residual_block_id);

  // insert into parameter block ordering book-keeping
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    Map::ParameterBlockSpec parameter_block_spec = parameters[i];

    // does it already exist as a parameter block connected?
    ParameterBlockInfo info;
    std::map<uint64_t, size_t>::iterator it =
        parameter_block_id_to_parameter_block_info_idx_.find(parameter_block_spec.first);
    if (it == parameter_block_id_to_parameter_block_info_idx_.end())
    {
      // not found. add it.
      // let's see, if it is actually a landmark, because then it will go into the sparse part
      bool is_landmark = false;
      if (std::dynamic_pointer_cast<HomogeneousPointParameterBlock>(
          parameter_block_spec.second) != 0)
      {
        is_landmark = true;
      }

      // resize equation system
      const size_t orig_size = H_.cols();
      size_t additionalSize = 0;
      if (!parameter_block_spec.second->fixed()) ////////DEBUG
        additionalSize = parameter_block_spec.second->minimalDimension();
      size_t denseSize = 0;

      if (dense_indices_ > 0)
        denseSize =
            parameter_block_infos_.at(dense_indices_ - 1).ordering_idx
            + parameter_block_infos_.at(dense_indices_ - 1).minimal_dimension;

      if(additionalSize > 0)
      {
        if (!is_landmark)
        {
          // insert
          // lhs
          Eigen::MatrixXd H01 = H_.topRightCorner(denseSize,
                                                  orig_size - denseSize);
          Eigen::MatrixXd H10 = H_.bottomLeftCorner(orig_size - denseSize,
                                                    denseSize);
          Eigen::MatrixXd H11 = H_.bottomRightCorner(orig_size - denseSize,
                                                     orig_size - denseSize);
          // rhs
          Eigen::VectorXd b1 = b0_.tail(orig_size - denseSize);

          conservativeResize(H_, orig_size + additionalSize,
                             orig_size + additionalSize);  // lhs
          conservativeResize(b0_, orig_size + additionalSize);  // rhs

          H_.topRightCorner(denseSize, orig_size - denseSize) = H01;
          H_.bottomLeftCorner(orig_size - denseSize, denseSize) = H10;
          H_.bottomRightCorner(orig_size - denseSize, orig_size - denseSize) = H11;
          H_.block(0, denseSize, H_.rows(), additionalSize).setZero();
          H_.block(denseSize, 0, additionalSize, H_.rows()).setZero();

          b0_.tail(orig_size - denseSize) = b1;
          b0_.segment(denseSize, additionalSize).setZero();
        }
        else
        {
          conservativeResize(H_, orig_size + additionalSize,
                             orig_size + additionalSize);  // lhs
          conservativeResize(b0_, orig_size + additionalSize);  // rhs
          // just append
          b0_.tail(additionalSize).setZero();
          H_.bottomRightCorner(H_.rows(), additionalSize).setZero();
          H_.bottomRightCorner(additionalSize, H_.rows()).setZero();
        }
      }

      // update book-keeping
      if (!is_landmark)
      {
        info = ParameterBlockInfo(parameter_block_spec.first,
                                  parameter_block_spec.second, denseSize,
                                  is_landmark);
        parameter_block_infos_.insert(
            parameter_block_infos_.begin() + dense_indices_, info);

        parameter_block_id_to_parameter_block_info_idx_.insert(
            std::pair<uint64_t, size_t>(parameter_block_spec.first,
                                        dense_indices_));

        //  update base_t book-keeping
        base_t::mutable_parameter_block_sizes()->insert(
            base_t::mutable_parameter_block_sizes()->begin() + dense_indices_,
            info.dimension);

        dense_indices_++;  // remember we increased the dense part of the problem

        // also increase the rest
        for (size_t j = dense_indices_; j < parameter_block_infos_.size(); ++j)
        {
          parameter_block_infos_.at(j).ordering_idx += additionalSize;
          parameter_block_id_to_parameter_block_info_idx_[parameter_block_infos_.at(j)
              .parameter_block_ptr->id()] += 1;
        }
      }
      else
      {
        // just add at the end
        info = ParameterBlockInfo(
            parameter_block_spec.first,
            parameter_block_spec.second,
            parameter_block_infos_.back().ordering_idx
                + parameter_block_infos_.back().minimal_dimension,
            is_landmark);
        parameter_block_infos_.push_back(info);
        parameter_block_id_to_parameter_block_info_idx_.insert(
            std::pair<uint64_t, size_t>(parameter_block_spec.first,
                                        parameter_block_infos_.size() - 1));

        //  update base_t book-keeping
        base_t::mutable_parameter_block_sizes()->push_back(info.dimension);
      }
      assert(
          parameter_block_infos_[
            parameter_block_id_to_parameter_block_info_idx_[parameter_block_spec
              .first]].parameter_block_id == parameter_block_spec.first);
    }
    else
    {

#ifdef USE_NEW_LINEARIZATION_POINT
      // switch linearization point - easy to do on the linearized part...
      size_t i = it->second;
      Eigen::VectorXd Delta_Chi_i(parameterBlockInfos_[i].minimal_dimension);
      parameterBlockInfos_[i].parameter_block_ptr->minus(
          parameterBlockInfos_[i].linearization_point.get(),
          parameterBlockInfos_[i].parameter_block_ptr->parameters(),
          Delta_Chi_i.data());
      b0_ -=
      H_.block(0,parameterBlockInfos_[i].ordering_idx,H_.rows(),
               parameterBlockInfos_[i].minimal_dimension)*
      Delta_Chi_i;
      parameterBlockInfos_[i].resetLinearizationPoint(
            parameterBlockInfos_[i].parameter_block_ptr);
#endif
      info = parameter_block_infos_.at(it->second);
    }
  }

  // update base_t book-keeping on residuals
  base_t::set_num_residuals(H_.cols());

  double** parameters_raw = new double*[parameters.size()];
  Eigen::VectorXd residuals_eigen(error_interface_ptr->residualDim());
  double* residuals_raw = residuals_eigen.data();

  double** jacobians_raw = new double*[parameters.size()];
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > >
      jacobiansEigen(parameters.size());

  double** jacobians_minimal_raw = new double*[parameters.size()];
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > >
      jacobians_minimal_eigen(parameters.size());

  for (size_t i = 0; i < parameters.size(); ++i)
  {
    DEBUG_CHECK(isParameterBlockConnected(parameters[i].first))
        << "ze bug: no linearization point, since not connected.";
    parameters_raw[i] =
        parameter_block_infos_
        [parameter_block_id_to_parameter_block_info_idx_[parameters[i].first]]
        .linearization_point.get();  // first estimate Jacobian!!

    jacobiansEigen[i].resize(error_interface_ptr->residualDim(),
                             parameters[i].second->dimension());
    jacobians_raw[i] = jacobiansEigen[i].data();
    jacobians_minimal_eigen[i].resize(error_interface_ptr->residualDim(),
                                    parameters[i].second->minimalDimension());
    jacobians_minimal_raw[i] = jacobians_minimal_eigen[i].data();
  }

  // evaluate residual block and get Jacobians.
  error_interface_ptr->EvaluateWithMinimalJacobians(parameters_raw, residuals_raw,
                                                  jacobians_raw,
                                                  jacobians_minimal_raw);


  // correct for loss function if applicable

  ceres::LossFunction* lossFunction = map_ptr_
      ->residualBlockIdToResidualBlockSpecMap().find(residual_block_id)->second
      .loss_function_ptr;
  if (lossFunction)
  {
    DEBUG_CHECK(
        map_ptr_->residualBlockIdToResidualBlockSpecMap().find(residual_block_id)
        != map_ptr_->residualBlockIdToResidualBlockSpecMap().end());

    // following ceres in internal/ceres/corrector.cc
    const double sq_norm = residuals_eigen.transpose() * residuals_eigen;
    double rho[3];
    lossFunction->Evaluate(sq_norm, rho);
    const double sqrt_rho1 = sqrt(rho[1]);
    double residual_scaling;
    double alpha_sq_norm;
    if ((sq_norm == 0.0) || (rho[2] <= 0.0))
    {
      residual_scaling = sqrt_rho1;
      alpha_sq_norm = 0.0;

    }
    else
    {
      // Calculate the smaller of the two solutions to the equation
      //
      // 0.5 *  alpha^2 - alpha - rho'' / rho' *  z'z = 0.
      //
      // Start by calculating the discriminant D.
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

      // Since both rho[1] and rho[2] are guaranteed to be positive at
      // this point, we know that D > 1.0.

      const double alpha = 1.0 - sqrt(D);
      DEBUG_CHECK(!std::isnan(alpha));

      // Calculate the constants needed by the correction routines.
      residual_scaling = sqrt_rho1 / (1 - alpha);
      alpha_sq_norm = alpha / sq_norm;
    }

    // correct Jacobians (Equation 11 in BANS)
    for (size_t i = 0; i < parameters.size(); ++i)
    {
      jacobians_minimal_eigen[i] = sqrt_rho1
          * (jacobians_minimal_eigen[i]
              - alpha_sq_norm * residuals_eigen
                  * (residuals_eigen.transpose() * jacobians_minimal_eigen[i]));
    }

    // correct residuals (caution: must be after "correct Jacobians"):
    residuals_eigen *= residual_scaling;
  }

  // add blocks to lhs and rhs
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    ParameterBlockInfo parameterBlockInfo_i = parameter_block_infos_.at(
        parameter_block_id_to_parameter_block_info_idx_[parameters[i].first]);

    DEBUG_CHECK(
          parameterBlockInfo_i.parameter_block_id == parameters[i].second->id())
        << "ze bug: inconsistent ze ordering";

    if (parameterBlockInfo_i.minimal_dimension == 0)
    {
      continue;
    }

    DEBUG_CHECK(H_.allFinite());

    // Insert Hessian and rhs in diagonal.
    H_.block(parameterBlockInfo_i.ordering_idx, parameterBlockInfo_i.ordering_idx,
             parameterBlockInfo_i.minimal_dimension,
             parameterBlockInfo_i.minimal_dimension) +=
        jacobians_minimal_eigen.at(i).transpose().eval()
        * jacobians_minimal_eigen.at(i);
    b0_.segment(parameterBlockInfo_i.ordering_idx,
                parameterBlockInfo_i.minimal_dimension) -=
        jacobians_minimal_eigen.at(i).transpose().eval() * residuals_eigen;

    DEBUG_CHECK(H_.allFinite())
        << jacobians_minimal_eigen.at(i).transpose().eval()
           * jacobians_minimal_eigen.at(i);

    for (size_t j = 0; j < i; ++j)
    {
      // Now the parts not in the diagonal
      ParameterBlockInfo parameterBlockInfo_j = parameter_block_infos_.at(
          parameter_block_id_to_parameter_block_info_idx_[parameters[j].first]);

      DEBUG_CHECK(
            parameterBlockInfo_j.parameter_block_id == parameters[j].second->id())
          << "ze bug: inconstistent ze ordering";

      if (parameterBlockInfo_j.minimal_dimension == 0)
      {
        continue;
      }

      // upper triangular:
      H_.block(parameterBlockInfo_i.ordering_idx,
               parameterBlockInfo_j.ordering_idx,
               parameterBlockInfo_i.minimal_dimension,
               parameterBlockInfo_j.minimal_dimension) +=
          jacobians_minimal_eigen.at(i).transpose().eval()
          * jacobians_minimal_eigen.at(j);
      // lower triangular:
      H_.block(parameterBlockInfo_j.ordering_idx,
               parameterBlockInfo_i.ordering_idx,
               parameterBlockInfo_j.minimal_dimension,
               parameterBlockInfo_i.minimal_dimension) +=
          jacobians_minimal_eigen.at(j).transpose().eval()
          * jacobians_minimal_eigen.at(i);
    }
  }

  // finally, we also have to delete the nonlinear residual block from the map:
  if (!keep)
  {
    map_ptr_->removeResidualBlock(residual_block_id);
  }

  // cleanup temporarily allocated stuff
  delete[] parameters_raw;
  delete[] jacobians_raw;
  delete[] jacobians_minimal_raw;

  check();

  return true;
}

// Info: is this parameter block connected to this marginalization error?
bool MarginalizationError::isParameterBlockConnected(
    uint64_t parameter_block_id) {
  DEBUG_CHECK(map_ptr_->parameterBlockExists(parameter_block_id))
      << "this parameter block does not even exist in the map...";
  std::map<uint64_t, size_t>::iterator it =
      parameter_block_id_to_parameter_block_info_idx_.find(parameter_block_id);
  if (it == parameter_block_id_to_parameter_block_info_idx_.end())
    return false;
  else
    return true;
}

// Checks the internal datastructure (debug)
void MarginalizationError::check() {
// check basic sizes
  DEBUG_CHECK(
        base_t::parameter_block_sizes().size()==parameter_block_infos_.size());
  DEBUG_CHECK(
        parameter_block_id_to_parameter_block_info_idx_.size()
        ==parameter_block_infos_.size());
  DEBUG_CHECK(base_t::num_residuals()==H_.cols());
  DEBUG_CHECK(base_t::num_residuals()==H_.rows());
  DEBUG_CHECK(base_t::num_residuals()==b0_.rows());
  DEBUG_CHECK(parameter_block_infos_.size()>=dense_indices_);
  int totalsize = 0;
  // check parameter block sizes
  for (size_t i = 0; i < parameter_block_infos_.size(); ++i) {
    totalsize += parameter_block_infos_[i].minimal_dimension;
    DEBUG_CHECK(parameter_block_infos_[i].dimension==
                size_t(base_t::parameter_block_sizes()[i]));
    DEBUG_CHECK(map_ptr_->parameterBlockExists(
                  parameter_block_infos_[i].parameter_block_id));
    DEBUG_CHECK(parameter_block_id_to_parameter_block_info_idx_[
                parameter_block_infos_[i].parameter_block_id]==i);
    if (i < dense_indices_)
    {
      DEBUG_CHECK(!parameter_block_infos_[i].is_landmark);
    }
    else
    {
      DEBUG_CHECK(parameter_block_infos_[i].is_landmark);
    }

  }
  // check contiguous
  for (size_t i = 1; i < parameter_block_infos_.size(); ++i) {
    DEBUG_CHECK(
        parameter_block_infos_[i-1].ordering_idx +
        parameter_block_infos_[i-1].minimal_dimension
        ==
        parameter_block_infos_[i].ordering_idx)
        << parameter_block_infos_[i-1].ordering_idx
        << "+" << parameter_block_infos_[i-1].minimal_dimension
        << "==" << parameter_block_infos_[i].ordering_idx;
  }
// check dimension again
  DEBUG_CHECK(base_t::num_residuals()==totalsize);
}

// Call this in order to (re-)add this error term after whenever it had been modified.
void MarginalizationError::getParameterBlockPtrs(
    std::vector<std::shared_ptr<ceres_backend::ParameterBlock> >& parameter_block_ptrs)
{
  DEBUG_CHECK(map_ptr_!=0) << "no Map object passed ever!";
  for (size_t i = 0; i < parameter_block_infos_.size(); ++i)
  {
    parameter_block_ptrs.push_back(parameter_block_infos_[i].parameter_block_ptr);
  }
}

// Marginalise out a set of parameter blocks.
bool MarginalizationError::marginalizeOut(
    const std::vector<uint64_t>& parameter_block_ids,
    const std::vector<bool>& keep_parameter_blocks) {
  if (parameter_block_ids.size() == 0)
  {
    return false;
  }
  //! @todo just remove the whole 'keepParameterBlocks'. It is not used...
  // copy so we can manipulate
  std::vector<uint64_t> parameter_block_ids_copy = parameter_block_ids;
  if (parameter_block_ids.size() != keep_parameter_blocks.size())
  {
    DEBUG_CHECK(keep_parameter_blocks.size() == 0)
        << "input vectors must either be of same size or omit optional parameter keepParameterBlocks: "
        << parameter_block_ids.size() << " vs " << keep_parameter_blocks.size();
  }
  std::map<uint64_t, bool> parameter_block_ptrs;
  for (size_t i = 0; i < parameter_block_ids_copy.size(); ++i)
  {
    bool keep = false;
    if (i < keep_parameter_blocks.size())
    {
      keep = keep_parameter_blocks.at(i);
    }
    parameter_block_ptrs.insert(
        std::pair<uint64_t, bool>(parameter_block_ids_copy.at(i), keep));
  }

  /* figure out which blocks need to be marginalized out */
  std::vector<std::pair<int, int> >
      marginalization_start_idx_and_length_pairs_landmarks;
  std::vector<std::pair<int, int> >
      marginalization_start_idx_and_length_pairs_dense;
  size_t marginalization_parameters_landmarks = 0;
  size_t marginalization_parameters_dense = 0;

  // make sure no duplications...
  std::sort(parameter_block_ids_copy.begin(), parameter_block_ids_copy.end());
  for (size_t i = 1; i < parameter_block_ids_copy.size(); ++i)
  {
    if (parameter_block_ids_copy[i] == parameter_block_ids_copy[i - 1])
    {
      parameter_block_ids_copy.erase(parameter_block_ids_copy.begin() + i);
      --i;
    }
  }
  for (size_t i = 0; i < parameter_block_ids_copy.size(); ++i)
  {
    std::map<uint64_t, size_t>::iterator it =
        parameter_block_id_to_parameter_block_info_idx_.find(
          parameter_block_ids_copy[i]);

    // sanity check - are we trying to marginalize stuff that is not connected to this error term?
    DEBUG_CHECK(it != parameter_block_id_to_parameter_block_info_idx_.end())
        << "trying to marginalize out unconnected parameter block id = "
        << parameter_block_ids_copy[i];
    if (it == parameter_block_id_to_parameter_block_info_idx_.end())
    {
      LOG(ERROR) << "trying to marginalize out unconnected parameter block id = "
                 << parameter_block_ids_copy[i];
      return false;
    }

    // distinguish dense and landmark (sparse) part for more efficient pseudo-inversion later on
    size_t start_idx = parameter_block_infos_.at(it->second).ordering_idx;
    size_t min_dim = parameter_block_infos_.at(it->second).minimal_dimension;
    if (parameter_block_infos_.at(it->second).is_landmark)
    {
      marginalization_start_idx_and_length_pairs_landmarks.push_back(
          std::pair<int, int>(start_idx, min_dim));
      marginalization_parameters_landmarks += min_dim;
    }
    else
    {
      marginalization_start_idx_and_length_pairs_dense.push_back(
          std::pair<int, int>(start_idx, min_dim));
      marginalization_parameters_dense += min_dim;
    }
  }

  // make sure the marginalization pairs are ordered
  std::sort(marginalization_start_idx_and_length_pairs_landmarks.begin(),
            marginalization_start_idx_and_length_pairs_landmarks.end(),
            [](std::pair<int,int> left, std::pair<int,int> right){
              return left.first < right.first;
            });
  std::sort(marginalization_start_idx_and_length_pairs_dense.begin(),
            marginalization_start_idx_and_length_pairs_dense.end(),
            [](std::pair<int,int> left, std::pair<int,int> right) {
              return left.first < right.first;
            });

  // Unify contiguous marginalization requests. I.e. if some of the parameters
  // are ordered right next to each other -> combine them.
  for (size_t m = 1;
       m < marginalization_start_idx_and_length_pairs_landmarks.size(); ++m)
  {
    if (marginalization_start_idx_and_length_pairs_landmarks.at(m - 1).first
        + marginalization_start_idx_and_length_pairs_landmarks.at(m - 1).second
        == marginalization_start_idx_and_length_pairs_landmarks.at(m).first)
    {
      marginalization_start_idx_and_length_pairs_landmarks.at(m - 1).second +=
          marginalization_start_idx_and_length_pairs_landmarks.at(m).second;
      marginalization_start_idx_and_length_pairs_landmarks.erase(
          marginalization_start_idx_and_length_pairs_landmarks.begin() + m);
      --m;
    }
  }
  for (size_t m = 1;
       m < marginalization_start_idx_and_length_pairs_dense.size(); ++m)
  {
    if (marginalization_start_idx_and_length_pairs_dense.at(m - 1).first
        + marginalization_start_idx_and_length_pairs_dense.at(m - 1).second
        == marginalization_start_idx_and_length_pairs_dense.at(m).first)
    {
      marginalization_start_idx_and_length_pairs_dense.at(m - 1).second +=
          marginalization_start_idx_and_length_pairs_dense.at(m).second;
      marginalization_start_idx_and_length_pairs_dense.erase(
          marginalization_start_idx_and_length_pairs_dense.begin() + m);
      --m;
    }
  }

  error_computation_valid_ = false;  // flag that the error computation is invalid

  // include in the fix rhs part deviations from linearization point of the parameter blocks to be marginalized
  // corrected: this is not necessary, will cancel itself

  /* landmark part (if existing) */
  if (marginalization_start_idx_and_length_pairs_landmarks.size() > 0)
  {
    // preconditioner
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(
          H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();

    // scale H and b
    H_ = p_inv.asDiagonal() * H_ * p_inv.asDiagonal();
    b0_ = p_inv.asDiagonal() * b0_;

    // U: Part to be kept, V: Marginalized part, W: Split.
    Eigen::MatrixXd U(H_.rows() - marginalization_parameters_landmarks,
                      H_.rows() - marginalization_parameters_landmarks);
    Eigen::MatrixXd V(marginalization_parameters_landmarks,
                      marginalization_parameters_landmarks);
    Eigen::MatrixXd W(H_.rows() - marginalization_parameters_landmarks,
                      marginalization_parameters_landmarks);
    // b_a kept, b_b marginalized.
    Eigen::VectorXd b_a(H_.rows() - marginalization_parameters_landmarks);
    Eigen::VectorXd b_b(marginalization_parameters_landmarks);

    // split preconditioner
    Eigen::VectorXd p_a(H_.rows() - marginalization_parameters_landmarks);
    Eigen::VectorXd p_b(marginalization_parameters_landmarks);
    splitVector(marginalization_start_idx_and_length_pairs_landmarks,
                p, p_a, p_b);  // output

    // split lhs
    splitSymmetricMatrix(marginalization_start_idx_and_length_pairs_landmarks,
                         H_, U, W, V);  // output

    // split rhs
    splitVector(marginalization_start_idx_and_length_pairs_landmarks,
                b0_, b_a, b_b);  // output


    // invert the marginalization block
    static const int sdim =
        ceres_backend::HomogeneousPointParameterBlock::c_minimal_dimension;
    b0_.resize(b_a.rows());
    b0_ = b_a;
    H_.resize(U.rows(), U.cols());
    H_ = U;
    const size_t numBlocks = V.cols() / sdim;
    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>>
        delta_H(numBlocks);
    std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>
        delta_b(numBlocks);
    Eigen::MatrixXd M1(W.rows(), W.cols());
    size_t idx = 0;
    // Calculate all the schur related things.
    for (size_t i = 0; static_cast<int>(i) < V.cols(); i += sdim)
    {
      Eigen::Matrix<double, sdim, sdim> V_inv_sqrt;
      Eigen::Matrix<double, sdim, sdim> V1 = V.block(i, i, sdim, sdim);
      MarginalizationError::pseudoInverseSymmSqrt(V1, V_inv_sqrt);
      Eigen::MatrixXd M = W.block(0, i, W.rows(), sdim) * V_inv_sqrt;
      Eigen::MatrixXd M1 = W.block(0, i, W.rows(), sdim)
          * V_inv_sqrt * V_inv_sqrt.transpose();
      // accumulate
      delta_H.at(idx).resize(U.rows(), U.cols());
      delta_b.at(idx).resize(b_a.rows());
      if (i == 0)
      {
        delta_H.at(idx) = M * M.transpose();
        delta_b.at(idx) = M1 * b_b.segment<sdim>(i);
      }
      else
      {
        delta_H.at(idx) = delta_H.at(idx - 1) + M * M.transpose();
        delta_b.at(idx) = delta_b.at(idx - 1) + M1 * b_b.segment<sdim>(i);
      }
      ++idx;
    }
    // Schur
    b0_ -= delta_b.at(idx - 1);
    H_ -= delta_H.at(idx - 1);

    // unscale
    H_ = p_a.asDiagonal() * H_ * p_a.asDiagonal();
    b0_ = p_a.asDiagonal() * b0_;
  }

  /* dense part (if existing) */
  if (marginalization_start_idx_and_length_pairs_dense.size() > 0)
  {
    // preconditioner
    Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(
          H_.diagonal().cwiseSqrt(),1.0e-3);
    Eigen::VectorXd p_inv = p.cwiseInverse();

    // scale H and b
    H_ = p_inv.asDiagonal() * H_ * p_inv.asDiagonal();
    b0_ = p_inv.asDiagonal() * b0_;

    Eigen::MatrixXd U(H_.rows() - marginalization_parameters_dense,
                      H_.rows() - marginalization_parameters_dense);
    Eigen::MatrixXd V(marginalization_parameters_dense,
                      marginalization_parameters_dense);
    Eigen::MatrixXd W(H_.rows() - marginalization_parameters_dense,
                      marginalization_parameters_dense);
    Eigen::VectorXd b_a(H_.rows() - marginalization_parameters_dense);
    Eigen::VectorXd b_b(marginalization_parameters_dense);

    // split preconditioner
    Eigen::VectorXd p_a(H_.rows() - marginalization_parameters_dense);
    Eigen::VectorXd p_b(marginalization_parameters_dense);
    splitVector(marginalization_start_idx_and_length_pairs_dense,
                p, p_a, p_b);  // output

    // split lhs
    splitSymmetricMatrix(marginalization_start_idx_and_length_pairs_dense,
                         H_, U, W, V);  // output

    // split rhs
    splitVector(marginalization_start_idx_and_length_pairs_dense, b0_,
                b_a, b_b);  // output

    // invert the marginalization block
    Eigen::MatrixXd V_inverse_sqrt(V.rows(), V.cols());
    Eigen::MatrixXd V1 = 0.5 * (V + V.transpose());
    pseudoInverseSymmSqrt(V1, V_inverse_sqrt);

    // Schur
    Eigen::MatrixXd M = W * V_inverse_sqrt;
    // rhs
    b0_.resize(b_a.rows());
    b0_ = (b_a - M * V_inverse_sqrt.transpose() * b_b);
    // lhs
    H_.resize(U.rows(), U.cols());

    H_ = (U - M * M.transpose());

    // unscale
    H_ = p_a.asDiagonal() * H_ * p_a.asDiagonal();
    b0_ = p_a.asDiagonal() * b0_;
  }

  // also adapt the ceres-internal size information
  base_t::set_num_residuals(base_t::num_residuals() -
                            marginalization_parameters_dense -
                            marginalization_parameters_landmarks);

  // delete all the book-keeping
  for (size_t i = 0; i < parameter_block_ids_copy.size(); ++i)
  {
    size_t idx = parameter_block_id_to_parameter_block_info_idx_.find(
        parameter_block_ids_copy[i])->second;
    int margSize = parameter_block_infos_.at(idx).minimal_dimension;
    parameter_block_infos_.erase(parameter_block_infos_.begin() + idx);

    for (size_t j = idx; j < parameter_block_infos_.size(); ++j)
    {
      parameter_block_infos_.at(j).ordering_idx -= margSize;
      parameter_block_id_to_parameter_block_info_idx_.at(
          parameter_block_infos_.at(j).parameter_block_id) -= 1;
    }

    parameter_block_id_to_parameter_block_info_idx_.erase(
          parameter_block_ids_copy[i]);

    // also adapt the ceres-internal book-keepin
    base_t::mutable_parameter_block_sizes()->erase(
        mutable_parameter_block_sizes()->begin() + idx);
  }

  // assume everything got dense
  // this is a conservative assumption, but true in particular when marginalizing
  // poses w/o landmarks
  dense_indices_ = parameter_block_infos_.size();
  for (size_t i = 0; i < parameter_block_infos_.size(); ++i)
  {
    if (parameter_block_infos_.at(i).is_landmark)
    {
      parameter_block_infos_.at(i).is_landmark = false;
    }
  }

  // check if the removal is safe
  for (size_t i = 0; i < parameter_block_ids_copy.size(); ++i)
  {
    Map::ResidualBlockCollection residuals = map_ptr_->residuals(
        parameter_block_ids_copy[i]);
    if (residuals.size() != 0
        && parameter_block_ptrs.at(parameter_block_ids_copy[i]) == false)
    {
      map_ptr_->printParameterBlockInfo(parameter_block_ids_copy[i]);
    }
    DEBUG_CHECK(residuals.size()==0 ||
                parameter_block_ptrs.at(parameter_block_ids_copy[i]) == true)
        << "trying to marginalize out a parameterBlock that is still connected "
        << "to other error terms. keep = "
        << int(parameter_block_ptrs.at(parameter_block_ids_copy[i]));
  }
  for (size_t i = 0; i < parameter_block_ids_copy.size(); ++i)
  {
    if (parameter_block_ptrs.at(parameter_block_ids_copy[i]))
    {
      LOG(FATAL) << "unmarginalizeLandmark(parameter_block_ids_copy[i]) "
                 << "not implemented.";
    }
    else
    {
      map_ptr_->removeParameterBlock(parameter_block_ids_copy[i]);
    }
  }

  check();

  return true;
}

// This must be called before optimization after adding residual blocks and/or
// marginalizing, since it performs all the lhs and rhs computations on from a
// given _H and _b.
void MarginalizationError::updateErrorComputation()
{
  if (error_computation_valid_)
  {
    return;  // already done.
  }

  // now we also know the error dimension:
  base_t::set_num_residuals(H_.cols());

  // preconditioner
  Eigen::VectorXd p = (H_.diagonal().array() > 1.0e-9).select(
        H_.diagonal().cwiseSqrt(),1.0e-3);
  Eigen::VectorXd p_inv = p.cwiseInverse();

  // H_lambda_lambda^star has to be positive semidefinit. Due to numeric issues
  // this migth not be the case => set eigenvalues below threshold to zero.
  // lhs SVD: _H = J^T*J = _U*S*_U^T
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(
      0.5  * p_inv.asDiagonal() * (H_ + H_.transpose())  * p_inv.asDiagonal() );

  static const double epsilon = std::numeric_limits<double>::epsilon();
  double tolerance = epsilon * H_.cols() * saes.eigenvalues().array().maxCoeff();
  S_ = Eigen::VectorXd(
      (saes.eigenvalues().array() > tolerance).select(
          saes.eigenvalues().array(), 0));
  S_pinv_ = Eigen::VectorXd(
      (saes.eigenvalues().array() > tolerance).select(
          saes.eigenvalues().array().inverse(), 0));

  S_sqrt_ = S_.cwiseSqrt();
  S_pinv_sqrt_ = S_pinv_.cwiseSqrt();

  // assign Jacobian
  J_ = (p.asDiagonal() * saes.eigenvectors()
        * (S_sqrt_.asDiagonal())).transpose();

  // constant error (residual) _e0 := (-pinv(J^T) * _b):
  Eigen::MatrixXd J_pinv_T =
      (S_pinv_sqrt_.asDiagonal()) * saes.eigenvectors().transpose()
      * p_inv.asDiagonal();
  e0_ = (-J_pinv_T * b0_);

  // reconstruct. TODO: check if this really improves quality --- doesn't seem so...
  //H_ = J_.transpose() * J_;
  //b0_ = -J_.transpose() * e0_;
  error_computation_valid_ = true;
}

// Computes the linearized deviation from the references (linearization points)
bool MarginalizationError::computeDeltaChi(Eigen::VectorXd& DeltaChi) const {
  DeltaChi.resize(H_.rows());
  for (size_t i = 0; i < parameter_block_infos_.size(); ++i)
  {
    // stack Delta_Chi vector
    if (!parameter_block_infos_[i].parameter_block_ptr->fixed())
    {
      Eigen::VectorXd Delta_Chi_i(parameter_block_infos_[i].minimal_dimension);
      parameter_block_infos_[i].parameter_block_ptr->minus(
          parameter_block_infos_[i].linearization_point.get(),
          parameter_block_infos_[i].parameter_block_ptr->parameters(),
          Delta_Chi_i.data());
      DeltaChi.segment(parameter_block_infos_[i].ordering_idx,
                       parameter_block_infos_[i].minimal_dimension) = Delta_Chi_i;
    }
  }
  return true;
}

// Computes the linearized deviation from the references (linearization points)
bool MarginalizationError::computeDeltaChi(double const* const * parameters,
                                           Eigen::VectorXd& DeltaChi) const
{
  DeltaChi.resize(H_.rows());
  for (size_t i = 0; i < parameter_block_infos_.size(); ++i)
  {
    // stack Delta_Chi vector
    if (!parameter_block_infos_[i].parameter_block_ptr->fixed())
    {
      Eigen::VectorXd Delta_Chi_i(parameter_block_infos_[i].minimal_dimension);
      parameter_block_infos_[i].parameter_block_ptr->
          minus(parameter_block_infos_[i].linearization_point.get(),
                parameters[i],
                Delta_Chi_i.data());
      DeltaChi.segment(parameter_block_infos_[i].ordering_idx,
                       parameter_block_infos_[i].minimal_dimension) = Delta_Chi_i;
    }
  }
  return true;
}

//This evaluates the error term and additionally computes the Jacobians.
bool MarginalizationError::Evaluate(double const* const * parameters,
                                    double* residuals,
                                    double** jacobians) const
{
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool MarginalizationError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobians_minimal) const
{
  DEBUG_CHECK(error_computation_valid_)
      << "trying to opmimize, but updateErrorComputation() was not called "
      << "after adding residual blocks/marginalizing";

  Eigen::VectorXd Delta_Chi;
  computeDeltaChi(parameters, Delta_Chi);

  for (size_t i = 0; i < parameter_block_infos_.size(); ++i)
  {
    // decompose the jacobians: minimal ones are easy
    if (jacobians_minimal != NULL)
    {
      if (jacobians_minimal[i] != NULL)
      {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor> >
            Jmin_i(jacobians_minimal[i],
                   e0_.rows(), parameter_block_infos_[i].minimal_dimension);
        Jmin_i = J_.block(0, parameter_block_infos_[i].ordering_idx, e0_.rows(),
                          parameter_block_infos_[i].minimal_dimension);
      }
    }

    // hallucinate the non-minimal Jacobians
    if (jacobians != NULL)
    {
      if (jacobians[i] != NULL)
      {
        Eigen::Map<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor> >
            J_i(jacobians[i], e0_.rows(), parameter_block_infos_[i].dimension);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            Jmin_i =
            J_.block(0, parameter_block_infos_[i].ordering_idx, e0_.rows(),
                     parameter_block_infos_[i].minimal_dimension);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            J_lift(
              parameter_block_infos_[i].parameter_block_ptr->minimalDimension(),
              parameter_block_infos_[i].parameter_block_ptr->dimension());
        parameter_block_infos_[i].parameter_block_ptr->liftJacobian(
            parameter_block_infos_[i].linearization_point.get(), J_lift.data());

        J_i = Jmin_i * J_lift;
      }
    }
  }

  // finally the error (residual) e = (-pinv(J^T) * _b + _J*Delta_Chi):
  Eigen::Map<Eigen::VectorXd> e(residuals, e0_.rows());
  e = e0_ + J_ * Delta_Chi;

  return true;
}

}  // namespace ceres_backend
}  // namespace svo

