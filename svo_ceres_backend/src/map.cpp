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
 *  Created on: Sep 8, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file Map.cpp
 * @brief Source file for the Map class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include "svo/ceres_backend/map.hpp"

#include <ceres/ordered_groups.h>

#include "svo/ceres_backend/homogeneous_point_parameter_block.hpp"
#include "svo/ceres_backend/marginalization_error.hpp"

namespace svo {
namespace ceres_backend {

// Constructor.
Map::Map()
    : residual_counter_(0)
{
  ceres::Problem::Options problemOptions;
  problemOptions.local_parameterization_ownership =
      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership =
      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership =
      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  //problemOptions.enable_fast_parameter_block_removal = true;
  problem_.reset(new ceres::Problem(problemOptions));
  //options.linear_solver_ordering = new ceres::ParameterBlockOrdering;
}

// Check whether a certain parameter block is part of the map.
bool Map::parameterBlockExists(uint64_t parameter_block_id) const
{
  if (id_to_parameter_block_map_.find(parameter_block_id)
      == id_to_parameter_block_map_.end())
  {
    return false;
  }
  return true;
}

// Log information on a parameter block.
void Map::printParameterBlockInfo(uint64_t parameter_block_id) const
{
  ResidualBlockCollection residualCollection = residuals(parameter_block_id);
  LOG(INFO) << "parameter info" << std::endl << "----------------------------"
            << std::endl << " - block Id: " << parameter_block_id << std::endl
            << " - type: " << parameterBlockPtr(parameter_block_id)->typeInfo()
            << std::endl << " - residuals (" << residualCollection.size()
            << "):";
  for (size_t i = 0; i < residualCollection.size(); ++i) {
    LOG(INFO)
        << "   - id: "
        << residualCollection.at(i).residual_block_id
        << std::endl
        << "   - type: "
        << kErrorToStr.at(errorInterfacePtr(residualCollection.at(i).residual_block_id)->typeInfo());
  }
  LOG(INFO) << "============================";
}

// Log information on a residual block.
void Map::printResidualBlockInfo(
    ceres::ResidualBlockId residual_block_id) const
{
  LOG(INFO) << "   - id: " << residual_block_id << std::endl << "   - type: "
            << kErrorToStr.at(errorInterfacePtr(residual_block_id)->typeInfo());
}

// Obtain the Hessian block for a specific parameter block.
void Map::getLhs(uint64_t parameter_block_id, Eigen::MatrixXd& H)
{
  DEBUG_CHECK(parameterBlockExists(parameter_block_id))
      << "parameter block not in map.";
  ResidualBlockCollection res = residuals(parameter_block_id);
  H.setZero();
  for (size_t i = 0; i < res.size(); ++i)
  {

    // parameters:
    ParameterBlockCollection pars = parameters(res[i].residual_block_id);

    double** parameters_raw = new double*[pars.size()];
    Eigen::VectorXd residuals_eigen(res[i].error_interface_ptr->residualDim());
    double* residuals_raw = residuals_eigen.data();

    double** jacobians_raw = new double*[pars.size()];
    std::vector<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::aligned_allocator<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::RowMajor> > > jacobiansEigen(pars.size());

    double** jacobians_minimal_raw = new double*[pars.size()];
    std::vector<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::aligned_allocator<
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::RowMajor> > > jacobians_minimal_eigen(pars.size());

    int J = -1;
    for (size_t j = 0; j < pars.size(); ++j)
    {
      // determine which is the relevant block
      if (pars[j].second->id() == parameter_block_id)
        J = j;
      parameters_raw[j] = pars[j].second->parameters();
      jacobiansEigen[j].resize(res[i].error_interface_ptr->residualDim(),
                               pars[j].second->dimension());
      jacobians_raw[j] = jacobiansEigen[j].data();
      jacobians_minimal_eigen[j].resize(res[i].error_interface_ptr->residualDim(),
                                      pars[j].second->minimalDimension());
      jacobians_minimal_raw[j] = jacobians_minimal_eigen[j].data();
    }

    // evaluate residual block
    res[i].error_interface_ptr->EvaluateWithMinimalJacobians(parameters_raw,
                                                           residuals_raw,
                                                           jacobians_raw,
                                                           jacobians_minimal_raw);

    // get block
    H += jacobians_minimal_eigen[J].transpose() * jacobians_minimal_eigen[J];

    // cleanup
    delete[] parameters_raw;
    delete[] jacobians_raw;
    delete[] jacobians_minimal_raw;
  }
}

// Check a Jacobian with numeric differences.
bool Map::isMinimalJacobianCorrect(ceres::ResidualBlockId residual_block_id,
                                   double relTol) const
{
  std::shared_ptr<const ceres_backend::ErrorInterface> error_interface_ptr =
      errorInterfacePtr(residual_block_id);
  ParameterBlockCollection parameter_blocks = parameters(residual_block_id);

  // set up data structures for storage
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > >
      J(parameter_blocks.size());
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > >
      J_min(parameter_blocks.size());
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, Eigen::Dynamic,
      Eigen::Dynamic, Eigen::RowMajor> > > J_min_numDiff(
      parameter_blocks.size());
  std::vector<double*> parameters(parameter_blocks.size());
  std::vector<double*> jacobians(parameter_blocks.size());
  std::vector<double*> jacobians_minimal(parameter_blocks.size());
  for (size_t i = 0; i < parameter_blocks.size(); ++i)
  {
    // fill in
    J[i].resize(error_interface_ptr->residualDim(),
                parameter_blocks[i].second->dimension());
    J_min[i].resize(error_interface_ptr->residualDim(),
                    parameter_blocks[i].second->minimalDimension());
    J_min_numDiff[i].resize(error_interface_ptr->residualDim(),
                            parameter_blocks[i].second->minimalDimension());
    parameters[i] = parameter_blocks[i].second->parameters();
    jacobians[i] = J[i].data();
    jacobians_minimal[i] = J_min[i].data();
  }

  // calculate num diff Jacobians
  const double delta = 1e-8;
  for (size_t i = 0; i < parameter_blocks.size(); ++i)
  {
    for (size_t j = 0; j < parameter_blocks[i].second->minimalDimension(); ++j)
    {
      Eigen::VectorXd residuals_p(error_interface_ptr->residualDim());
      Eigen::VectorXd residuals_m(error_interface_ptr->residualDim());

      // apply positive delta
      Eigen::VectorXd parameters_p(parameter_blocks[i].second->dimension());
      Eigen::VectorXd parameters_m(parameter_blocks[i].second->dimension());
      Eigen::VectorXd plus(parameter_blocks[i].second->minimalDimension());
      plus.setZero();
      plus[j] = delta;
      parameter_blocks[i].second->plus(parameters[i], plus.data(),
                                       parameters_p.data());
      parameters[i] = parameters_p.data();
      error_interface_ptr->EvaluateWithMinimalJacobians(parameters.data(),
                                                       residuals_p.data(),
                                                       nullptr,
                                                       nullptr);
      parameters[i] = parameter_blocks[i].second->parameters();  // reset
      // apply negative delta
      plus.setZero();
      plus[j] = -delta;
      parameter_blocks[i].second->plus(parameters[i], plus.data(),
                                       parameters_m.data());
      parameters[i] = parameters_m.data();
      error_interface_ptr->EvaluateWithMinimalJacobians(parameters.data(),
                                                       residuals_m.data(),
                                                       nullptr,
                                                       nullptr);
      parameters[i] = parameter_blocks[i].second->parameters();  // reset
      // calculate numeric difference
      J_min_numDiff[i].col(j) = (residuals_p - residuals_m) * 1.0 / (2.0 * delta);
    }
  }

  // calculate analytic Jacobians and compare
  bool isCorrect = true;
  Eigen::VectorXd residuals(error_interface_ptr->residualDim());
  for (size_t i = 0; i < parameter_blocks.size(); ++i)
  {
    // calc
    error_interface_ptr->EvaluateWithMinimalJacobians(parameters.data(),
                                                     residuals.data(),
                                                     jacobians.data(),
                                                     jacobians_minimal.data());
    // check
    double norm_minimal = J_min_numDiff[i].norm();
    Eigen::MatrixXd J_diff_minimal = J_min_numDiff[i] - J_min[i];
    double max_diff_minimal =
        std::max(-J_diff_minimal.minCoeff(), J_diff_minimal.maxCoeff());

    if (max_diff_minimal / norm_minimal > relTol)
    {
      LOG(INFO) << "Minimal Jacobian inconsistent: "
                << kErrorToStr.at(error_interface_ptr->typeInfo());
      LOG(INFO) << "num diff Jacobian[" << i << "]:\n" << J_min_numDiff[i];
      LOG(INFO) << "provided Jacobian[" << i << "]:\n" << J_min[i];
      LOG(INFO) << "relative error: " << max_diff_minimal / norm_minimal
                << ", relative tolerance: " << relTol;
      isCorrect = false;
    }
  }

  return isCorrect;
}

// Add a parameter block to the map
bool Map::addParameterBlock(
    std::shared_ptr<ceres_backend::ParameterBlock> parameter_block,
    int parameterization, const int /*group*/)
{

  DEBUG_CHECK(parameter_block != nullptr);
  VLOG(200) << "Adding parameter block with parameterization "
            << parameterization << " and id " << BackendId(parameter_block->id());

  // check Id availability
  if (parameterBlockExists(parameter_block->id()))
  {
    LOG(ERROR) << "Parameter block with id " << BackendId(parameter_block->id())
               << " exists already!";
    return false;
  }

  id_to_parameter_block_map_.insert(
      std::pair<uint64_t, std::shared_ptr<ceres_backend::ParameterBlock> >(
          parameter_block->id(), parameter_block));

  // also add to ceres problem
  switch (parameterization)
  {
    case Parameterization::Trivial:
    {
      problem_->AddParameterBlock(parameter_block->parameters(),
                                  parameter_block->dimension());
      break;
    }
    case Parameterization::HomogeneousPoint:
    {
      problem_->AddParameterBlock(parameter_block->parameters(),
                                  parameter_block->dimension(),
                                  &homogeneous_point_local_parameterization_);
      parameter_block->setLocalParameterizationPtr(
          &homogeneous_point_local_parameterization_);
      break;
    }
    case Parameterization::Pose6d:
    {
      problem_->AddParameterBlock(parameter_block->parameters(),
                                  parameter_block->dimension(),
                                  &pose_local_parameterization_);
      parameter_block->setLocalParameterizationPtr(&pose_local_parameterization_);
      break;
    }
    default:
    {
      LOG(ERROR) << "Unknown parameterization!";
      return false;
      break;  // just for consistency...
    }
  }

  /*const ceres_backend::LocalParamizationAdditionalInterfaces* ptr =
      dynamic_cast<const ceres_backend::LocalParamizationAdditionalInterfaces*>(
      parameter_block->localParameterizationPtr());
  if(ptr)
    std::cout<<"verify local size "<< parameter_block->localParameterizationPtr()->LocalSize() << " = "<<
            int(ptr->verify(parameter_block->parameters()))<<
            std::endl;*/

  return true;
}

// Remove a parameter block from the map.
bool Map::removeParameterBlock(uint64_t parameter_block_id)
{
  if (!parameterBlockExists(parameter_block_id))
  {
    return false;
  }
  VLOG(200) << "Removing paramter block with ID " << BackendId(parameter_block_id);

  // remove all connected residuals
  const ResidualBlockCollection res = residuals(parameter_block_id);
  for (size_t i = 0; i < res.size(); ++i)
  {
    removeResidualBlock(res[i].residual_block_id);  // remove in ceres and book-keeping
  }
  problem_->RemoveParameterBlock(
      parameterBlockPtr(parameter_block_id)->parameters());  // remove parameter block
  id_to_parameter_block_map_.erase(parameter_block_id);  // remove book-keeping
  return true;
}

// Remove a parameter block from the map.
bool Map::removeParameterBlock(
    std::shared_ptr<ceres_backend::ParameterBlock> parameter_block)
{
  return removeParameterBlock(parameter_block->id());
}

// Adds a residual block.
ceres::ResidualBlockId Map::addResidualBlock(
    std::shared_ptr< ceres::CostFunction> cost_function,
    ceres::LossFunction* loss_function,
    std::vector<std::shared_ptr<ceres_backend::ParameterBlock> >& parameter_block_ptrs)
{
  ceres::ResidualBlockId return_id;
  std::vector<double*> parameter_blocks;
  ParameterBlockCollection parameter_block_collection;
  for (size_t i = 0; i < parameter_block_ptrs.size(); ++i)
  {
    parameter_blocks.push_back(parameter_block_ptrs.at(i)->parameters());
    parameter_block_collection.push_back(
        ParameterBlockSpec(parameter_block_ptrs.at(i)->id(),
                           parameter_block_ptrs.at(i)));
  }

  // add in ceres
  return_id = problem_->AddResidualBlock(cost_function.get(), loss_function,
                                         parameter_blocks);

  if (FLAGS_v >=200)
  {
    std::stringstream s;
    s << "Adding residual block: "
      << kErrorToStr.at(std::dynamic_pointer_cast<ceres_backend::ErrorInterface>(cost_function)->typeInfo())
      << " with id " << return_id
      << " connected to the following parameter blocks:\n";
    for (auto block : parameter_block_ptrs)
    {
      s << BackendId(block->id()) << "\n";
    }
    VLOG(200) << s.str();
  }

  // add in book-keeping
  std::shared_ptr<ErrorInterface> error_interface_ptr =
      std::dynamic_pointer_cast<ErrorInterface>(cost_function);
  DEBUG_CHECK(error_interface_ptr!=0)
      << "Supplied a cost function without ceres_backend::ErrorInterface";
  residual_block_id_to_residual_block_spec_map_.insert(
      std::pair< ceres::ResidualBlockId, ResidualBlockSpec>(
          return_id,
          ResidualBlockSpec(return_id, loss_function, error_interface_ptr)));

  // update book-keeping
  bool insertion_success;
  std::tie(std::ignore, insertion_success) =
      residual_block_id_to_parameter_block_collection_map_.insert(
          std::make_pair(return_id, parameter_block_collection));
  if (insertion_success == false)
  {
    return ceres::ResidualBlockId(0);
  }

  // update ResidualBlock pointers on involved ParameterBlocks
  for (uint64_t parameter_id = 0;
      parameter_id < parameter_block_collection.size(); ++parameter_id)
  {
    id_to_residual_block_multimap_.insert(
        std::pair<uint64_t, ResidualBlockSpec>(
            parameter_block_collection[parameter_id].first,
            ResidualBlockSpec(return_id, loss_function, error_interface_ptr)));
  }

  return return_id;
}

// Add a residual block. See respective ceres docu. If more are needed, see other interface.
ceres::ResidualBlockId Map::addResidualBlock(
    std::shared_ptr< ceres::CostFunction> cost_function,
    ceres::LossFunction* loss_function,
    std::shared_ptr<ceres_backend::ParameterBlock> x0,
    std::shared_ptr<ceres_backend::ParameterBlock> x1,
    std::shared_ptr<ceres_backend::ParameterBlock> x2,
    std::shared_ptr<ceres_backend::ParameterBlock> x3,
    std::shared_ptr<ceres_backend::ParameterBlock> x4,
    std::shared_ptr<ceres_backend::ParameterBlock> x5,
    std::shared_ptr<ceres_backend::ParameterBlock> x6,
    std::shared_ptr<ceres_backend::ParameterBlock> x7,
    std::shared_ptr<ceres_backend::ParameterBlock> x8,
    std::shared_ptr<ceres_backend::ParameterBlock> x9)
{

  DEBUG_CHECK(cost_function != nullptr);
  std::vector<std::shared_ptr<ceres_backend::ParameterBlock> > parameter_block_ptrs;
  if (x0 != 0)
  {
    parameter_block_ptrs.push_back(x0);
  }
  if (x1 != 0)
  {
    parameter_block_ptrs.push_back(x1);
  }
  if (x2 != 0)
  {
    parameter_block_ptrs.push_back(x2);
  }
  if (x3 != 0)
  {
    parameter_block_ptrs.push_back(x3);
  }
  if (x4 != 0)
  {
    parameter_block_ptrs.push_back(x4);
  }
  if (x5 != 0)
  {
    parameter_block_ptrs.push_back(x5);
  }
  if (x6 != 0)
  {
    parameter_block_ptrs.push_back(x6);
  }
  if (x7 != 0)
  {
    parameter_block_ptrs.push_back(x7);
  }
  if (x8 != 0)
  {
    parameter_block_ptrs.push_back(x8);
  }
  if (x9 != 0)
  {
    parameter_block_ptrs.push_back(x9);
  }

  return Map::addResidualBlock(cost_function, loss_function, parameter_block_ptrs);

}

// Replace the parameters connected to a residual block ID.
void Map::resetResidualBlock(
    ceres::ResidualBlockId residual_block_id,
    std::vector<std::shared_ptr<ceres_backend::ParameterBlock> >& parameter_block_ptrs)
{
  // remember the residual block spec:
  ResidualBlockSpec spec =
      residual_block_id_to_residual_block_spec_map_[residual_block_id];
  // remove residual from old parameter set
  ResidualBlockIdToParameterBlockCollectionMap::iterator it =
      residual_block_id_to_parameter_block_collection_map_.find(residual_block_id);
  DEBUG_CHECK(it!=residual_block_id_to_parameter_block_collection_map_.end())
      << "residual block not in map.";
  for (ParameterBlockCollection::iterator parameter_it = it->second.begin();
      parameter_it != it->second.end(); ++parameter_it)
  {
    uint64_t parameter_id = parameter_it->second->id();
    std::pair<IdToResidualBlockMultimap::iterator,
        IdToResidualBlockMultimap::iterator> range = id_to_residual_block_multimap_
        .equal_range(parameter_id);
    DEBUG_CHECK(range.first!=id_to_residual_block_multimap_.end())
        << "book-keeping is broken";
    for (IdToResidualBlockMultimap::iterator it2 = range.first;
        it2 != range.second;)
    {
      if (residual_block_id == it2->second.residual_block_id)
      {
        it2 = id_to_residual_block_multimap_.erase(it2);  // remove book-keeping
      }
      else
      {
        it2++;
      }
    }
  }

  ParameterBlockCollection parameter_block_collection;
  for (size_t i = 0; i < parameter_block_ptrs.size(); ++i)
  {
    parameter_block_collection.push_back(
        ParameterBlockSpec(parameter_block_ptrs.at(i)->id(),
                           parameter_block_ptrs.at(i)));
  }

  // update book-keeping
  it->second = parameter_block_collection;

  // update ResidualBlock pointers on involved ParameterBlocks
  for (uint64_t parameter_id = 0;
      parameter_id < parameter_block_collection.size(); ++parameter_id)
  {
    id_to_residual_block_multimap_.insert(
        std::pair<uint64_t, ResidualBlockSpec>(
            parameter_block_collection[parameter_id].first, spec));
  }
}

// Remove a residual block.
bool Map::removeResidualBlock(ceres::ResidualBlockId residual_block_id)
{
  VLOG(200) << "Removing residual block with ID " << residual_block_id;
  problem_->RemoveResidualBlock(residual_block_id);  // remove in ceres

  ResidualBlockIdToParameterBlockCollectionMap::iterator it =
      residual_block_id_to_parameter_block_collection_map_.find(residual_block_id);
  if (it == residual_block_id_to_parameter_block_collection_map_.end())
  {
    return false;
  }

  for (ParameterBlockCollection::iterator parameter_it = it->second.begin();
      parameter_it != it->second.end(); ++parameter_it)
  {
    uint64_t parameter_id = parameter_it->second->id();
    std::pair<IdToResidualBlockMultimap::iterator,
        IdToResidualBlockMultimap::iterator> range = id_to_residual_block_multimap_
        .equal_range(parameter_id);
    DEBUG_CHECK(range.first!=id_to_residual_block_multimap_.end())
        << "book-keeping is broken";

    for (IdToResidualBlockMultimap::iterator it2 = range.first;
        it2 != range.second;)
    {
      if (residual_block_id == it2->second.residual_block_id)
      {
        it2 = id_to_residual_block_multimap_.erase(it2);  // remove book-keeping
      }
      else
      {
        it2++;
      }
    }
  }
  residual_block_id_to_parameter_block_collection_map_.erase(it);  // remove book-keeping
  residual_block_id_to_residual_block_spec_map_.erase(residual_block_id);  // remove book-keeping
  return true;
}

// Do not optimise a certain parameter block.
bool Map::setParameterBlockConstant(uint64_t parameter_block_id)
{
  if (!parameterBlockExists(parameter_block_id))
  {
    return false;
  }
  std::shared_ptr<ParameterBlock> parameter_block = id_to_parameter_block_map_.find(
      parameter_block_id)->second;
  parameter_block->setFixed(true);
  problem_->SetParameterBlockConstant(parameter_block->parameters());
  return true;
}

bool Map::isParameterBlockConstant(uint64_t parameter_block_id)
{
  if (!parameterBlockExists(parameter_block_id))
  {
    return false;
  }
  std::shared_ptr<ParameterBlock> parameter_block = id_to_parameter_block_map_.find(
      parameter_block_id)->second;
  CHECK_EQ(problem_->IsParameterBlockConstant(parameter_block->parameters()),
           parameter_block->fixed());
  return parameter_block->fixed();
}

// Optimise a certain parameter block (this is the default).
bool Map::setParameterBlockVariable(uint64_t parameter_block_id)
{
  if (!parameterBlockExists(parameter_block_id))
    return false;
  std::shared_ptr<ParameterBlock> parameter_block =
      id_to_parameter_block_map_.find(parameter_block_id)->second;
  parameter_block->setFixed(false);
  problem_->SetParameterBlockVariable(parameter_block->parameters());
  return true;
}

// Reset the (local) parameterisation of a parameter block.
bool Map::resetParameterization(uint64_t parameter_block_id,
                                int parameterization)
{
  if (!parameterBlockExists(parameter_block_id))
  {
    return false;
  }
  // the ceres documentation states that a parameterization may never be changed on.
  // therefore, we have to remove the parameter block in question and re-add it.
  ResidualBlockCollection res = residuals(parameter_block_id);
  std::shared_ptr<ParameterBlock> par_block_ptr =
      parameterBlockPtr(parameter_block_id);

  // get parameter block pointers
  std::vector<std::vector<std::shared_ptr<ceres_backend::ParameterBlock> > >
      parameter_block_ptrs(res.size());
  for (size_t r = 0; r < res.size(); ++r)
  {
    ParameterBlockCollection pspec = parameters(res[r].residual_block_id);
    for (size_t p = 0; p < pspec.size(); ++p)
    {
      parameter_block_ptrs[r].push_back(pspec[p].second);
    }
  }

  // remove
  // int group = options.linear_solver_ordering->GroupId(parBlockPtr->parameters());
  removeParameterBlock(parameter_block_id);
  // add with new parameterization
  addParameterBlock(par_block_ptr, parameterization/*,group*/);

  // re-assemble
  for (size_t r = 0; r < res.size(); ++r)
  {
    addResidualBlock(
        std::dynamic_pointer_cast< ceres::CostFunction>(
            res[r].error_interface_ptr),
        res[r].loss_function_ptr, parameter_block_ptrs[r]);
  }

  return true;
}

// Set the (local) parameterisation of a parameter block.
bool Map::setParameterization(
    uint64_t parameter_block_id,
    ceres::LocalParameterization* local_parameterization)
{
  if (!parameterBlockExists(parameter_block_id))
  {
    return false;
  }
  problem_->SetParameterization(
      id_to_parameter_block_map_.find(parameter_block_id)->second->parameters(),
      local_parameterization);
  id_to_parameter_block_map_.find(parameter_block_id)->second
      ->setLocalParameterizationPtr(local_parameterization);
  return true;
}

// getters
// Get a shared pointer to a parameter block.
std::shared_ptr<ceres_backend::ParameterBlock> Map::parameterBlockPtr(
    uint64_t parameter_block_id)
{
  // get a parameterBlock
  CHECK(parameterBlockExists(parameter_block_id))
      << "parameterBlock with id " << BackendId(parameter_block_id)
      << " does not exist";
  if (parameterBlockExists(parameter_block_id))
  {
    return id_to_parameter_block_map_.find(parameter_block_id)->second;
  }
  return std::shared_ptr<ceres_backend::ParameterBlock>();  // NULL
}

// Get a shared pointer to a parameter block.
std::shared_ptr<const ceres_backend::ParameterBlock> Map::parameterBlockPtr(
    uint64_t parameter_block_id) const
{
  // get a parameterBlock
  if (parameterBlockExists(parameter_block_id))
  {
    return id_to_parameter_block_map_.find(parameter_block_id)->second;
  }
  return std::shared_ptr<const ceres_backend::ParameterBlock>();  // NULL
}

// Get the residual blocks of a parameter block.
Map::ResidualBlockCollection Map::residuals(uint64_t parameter_block_id) const
{
  // get the residual blocks of a parameter block
  IdToResidualBlockMultimap::const_iterator it1 = id_to_residual_block_multimap_
      .find(parameter_block_id);
  if (it1 == id_to_residual_block_multimap_.end())
    return Map::ResidualBlockCollection();  // empty
  ResidualBlockCollection returnResiduals;
  std::pair<IdToResidualBlockMultimap::const_iterator,
      IdToResidualBlockMultimap::const_iterator> range =
      id_to_residual_block_multimap_.equal_range(parameter_block_id);
  for (IdToResidualBlockMultimap::const_iterator it = range.first;
      it != range.second; ++it)
  {
    returnResiduals.push_back(it->second);
  }
  return returnResiduals;
}

// Get a shared pointer to an error term.
std::shared_ptr<ceres_backend::ErrorInterface> Map::errorInterfacePtr(
    ceres::ResidualBlockId residual_block_id)
{  // get a vertex
  ResidualBlockIdToResidualBlockSpecMap::iterator it =
      residual_block_id_to_residual_block_spec_map_.find(residual_block_id);
  if (it == residual_block_id_to_residual_block_spec_map_.end())
  {
    return std::shared_ptr<ceres_backend::ErrorInterface>();  // NULL
  }
  return it->second.error_interface_ptr;
}

// Get a shared pointer to an error term.
std::shared_ptr<const ceres_backend::ErrorInterface> Map::errorInterfacePtr(
    ceres::ResidualBlockId residual_block_id) const
{  // get a vertex
  ResidualBlockIdToResidualBlockSpecMap::const_iterator it =
      residual_block_id_to_residual_block_spec_map_.find(residual_block_id);
  if (it == residual_block_id_to_residual_block_spec_map_.end())
  {
    return std::shared_ptr<ceres_backend::ErrorInterface>();  // NULL
  }
  return it->second.error_interface_ptr;
}

// Get the parameters of a residual block.
Map::ParameterBlockCollection Map::parameters(
    ceres::ResidualBlockId residual_block_id) const
{
  // get the parameter blocks connected
  ResidualBlockIdToParameterBlockCollectionMap::const_iterator it =
      residual_block_id_to_parameter_block_collection_map_.find(residual_block_id);
  if (it == residual_block_id_to_parameter_block_collection_map_.end())
  {
    ParameterBlockCollection empty;
    return empty;  // empty vector
  }
  return it->second;
}

}  //namespace svo
}  //namespace ceres_backend

