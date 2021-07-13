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
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file Map.hpp
 * @brief Header file for the Map class. This essentially encapsulates the ceres::Problem.
 * @author Stefan Leutenegger
 */

#pragma once

#include <memory>
#include <unordered_map>

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <ceres/ceres.h>
#pragma diagnostic pop

#include "svo/ceres_backend/error_interface.hpp"
#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/parameter_block.hpp"
#include "svo/ceres_backend/pose_local_parameterization.hpp"

namespace svo {
namespace ceres_backend {

/// @brief The Map class. This keeps track of how parameter blocks are connected
///        to residual blocks. In essence, it encapsulates the ceres::Problem.
///        This way, we can easily manipulate the optimisation problem.
///        You could argue why not use cere's internal mechanisms to do that.
///        We found that our implementation was faster...
class Map
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// @brief Constructor.
  Map();

  // definitions
  /// @brief Struct to store some infos about a residual.
  struct ResidualBlockSpec
  {
    ResidualBlockSpec()
        : residual_block_id(0),
          loss_function_ptr(0),
          error_interface_ptr(std::shared_ptr<ErrorInterface>())
    {}

    /// @brief Constructor
    /// @param[in] residual_block_id ID of residual block.
    /// @param[in] loss_function_ptr The m-estimator.
    /// @param[in] error_interface_ptr The pointer to the error interface of the
    ///            respective residual block.
    ResidualBlockSpec(ceres::ResidualBlockId residual_block_id,
                      ceres::LossFunction* loss_function_ptr,
                      std::shared_ptr<ErrorInterface> error_interface_ptr)
        : residual_block_id(residual_block_id),
          loss_function_ptr(loss_function_ptr),
          error_interface_ptr(error_interface_ptr)
    {}

    ceres::ResidualBlockId residual_block_id; ///< ID of residual block.
    ceres::LossFunction* loss_function_ptr; ///< The m-estimator.
    std::shared_ptr<ErrorInterface> error_interface_ptr;
    ///< The pointer to the error interface of the respective residual block.
  };
  typedef std::pair<uint64_t,
  std::shared_ptr<ceres_backend::ParameterBlock> > ParameterBlockSpec;

  typedef std::vector<ResidualBlockSpec> ResidualBlockCollection;
  typedef std::vector<ParameterBlockSpec> ParameterBlockCollection;

  /// @brief The Parameterisation enum
  enum Parameterization
  {
    HomogeneousPoint,     ///< Use ceres_backend::HomogeneousPointLocalParameterization.
    Pose6d,               ///< Use ceres_backend::PoseLocalParameterization.
    Trivial               ///< No local parameterisation.
  };

  /**
   * @brief Check whether a certain parameter block is part of the map.
   * @param parameter_block_id ID of parameter block to find.
   * @return True if parameter block is part of map.
   */
  bool parameterBlockExists(uint64_t parameter_block_id) const;

  /// @name Print info
  /// @{

  /// @brief Log information on a parameter block.
  void printParameterBlockInfo(uint64_t parameter_block_id) const;

  /// @brief Log information on a residual block.
  void printResidualBlockInfo(ceres::ResidualBlockId residual_block_id) const;

  /// @}

  // for quality assessment
  /**
   * @brief Obtain the Hessian block for a specific parameter block.
   * @param[in] parameter_block_id Parameter block ID of interest.
   * @param[out] H the output Hessian block.
   */
  void getLhs(uint64_t parameter_block_id, Eigen::MatrixXd& H);

  /// @name add/remove
  /// @{

  /**
   * @brief Add a parameter block to the map.
   * @param parameter_block    Parameter block to insert.
   * @param parameterization  ceres_backend::Parameterization to tell how to do the local
   *                          parameterisation.
   * @param group             Schur elimination group -- currently unused.
   * @return True if successful.
   */
  bool addParameterBlock(
      std::shared_ptr<ceres_backend::ParameterBlock> parameter_block,
      int parameterization = Parameterization::Trivial, const int group = -1);

  /**
   * @brief Remove a parameter block from the map.
   * @param parameter_block_id ID of block to remove.
   * @return True if successful.
   */
  bool removeParameterBlock(uint64_t parameter_block_id);

  /**
   * @brief Remove a parameter block from the map.
   * @param parameter_block Pointer to the block to remove.
   * @return True if successful.
   */
  bool removeParameterBlock(
      std::shared_ptr<ceres_backend::ParameterBlock> parameter_block);

  /**
   * @brief Adds a residual block.
   * @param[in] cost_function The error term to be used.
   * @param[in] loss_function Use an m-estimator? NULL, if not needed.
   * @param[in] parameter_block_ptrs A vector that contains all the parameter
   *            blocks the error term relates to.
   * @return
   */
  ceres::ResidualBlockId addResidualBlock(
      std::shared_ptr< ceres::CostFunction> cost_function,
      ceres::LossFunction* loss_function,
      std::vector<std::shared_ptr<ceres_backend::ParameterBlock> >& parameter_block_ptrs);

  /**
   * @brief Replace the parameters connected to a residual block ID.
   * @param[in] residual_block_id The ID of the residual block the parameter
   *            blocks of which are to be to be replaced.
   * @param[in] parameter_block_ptrs A vector containing the parameter blocks
   *            to be replaced.
   */
  void resetResidualBlock(
      ceres::ResidualBlockId residual_block_id,
      std::vector<std::shared_ptr<ceres_backend::ParameterBlock> >& parameter_block_ptrs);

  /**
   * @brief Add a residual block. See respective ceres docu. If more are needed,
   *        see other interface.
   * @param[in] cost_function The error term to be used.
   * @param[in] loss_function Use an m-estimator? NULL, if not needed.
   * @param[in] x0 The first parameter block.
   * @param[in] x1 The second parameter block (if existent).
   * @param[in] x2 The third parameter block (if existent).
   * @param[in] x3 The 4th parameter block (if existent).
   * @param[in] x4 The 5th parameter block (if existent).
   * @param[in] x5 The 6th parameter block (if existent).
   * @param[in] x6 The 7th parameter block (if existent).
   * @param[in] x7 The 8th parameter block (if existent).
   * @param[in] x8 The 9th parameter block (if existent).
   * @param[in] x9 The 10th parameter block (if existent).
   * @return The residual block ID, i.e. what cost_function points to.
   */
  ceres::ResidualBlockId addResidualBlock(
      std::shared_ptr< ceres::CostFunction> cost_function,
      ceres::LossFunction* loss_function,
      std::shared_ptr<ceres_backend::ParameterBlock> x0,
      std::shared_ptr<ceres_backend::ParameterBlock> x1 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x2 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x3 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x4 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x5 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x6 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x7 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x8 = std::shared_ptr<
          ceres_backend::ParameterBlock>(),
      std::shared_ptr<ceres_backend::ParameterBlock> x9 = std::shared_ptr<
          ceres_backend::ParameterBlock>());

  /**
   * @brief Remove a residual block.
   * @param[in] id The residual block ID of the residual block to be removed.
   * @return True on success.
   */
  bool removeResidualBlock(ceres::ResidualBlockId id);

  /// @}

  /// @name Set constant/variable/local parameterization
  /// @{

  /**
   * @brief Do not optimise a certain parameter block.
   * @param[in] parameter_block_id The parameter block ID of the parameter block
   *            to set fixed.
   * @return True on success.
   */
  bool setParameterBlockConstant(uint64_t parameter_block_id);
  bool isParameterBlockConstant(uint64_t parameter_block_id);

  /**
   * @brief Optimise a certain parameter block (this is the default).
   * @param[in] parameter_block_id The parameter block ID of the parameter block
   *            to set fixed.
   * @return True on success.
   */
  bool setParameterBlockVariable(uint64_t parameter_block_id);

  /**
   * @brief Do not optimise a certain parameter block.
   * @param[in] parameter_block Pointer to the parameter block that should be constant.
   * @return True on success.
   */
  bool setParameterBlockConstant(
      std::shared_ptr<ceres_backend::ParameterBlock> parameter_block)
  {
    return setParameterBlockConstant(parameter_block->id());
  }

  bool isParameterBlockConstant(
      std::shared_ptr<ceres_backend::ParameterBlock> parameter_block)
  {
    return isParameterBlockConstant(parameter_block->id());
  }


  /**
   * @brief Optimise a certain parameter block (this is the default).
   * @param[in] parameter_block Pointer to the parameter block that should be optimised.
   * @return True on success.
   */
  bool setParameterBlockVariable(
      std::shared_ptr<ceres_backend::ParameterBlock> parameter_block) {
    return setParameterBlockVariable(parameter_block->id());
  }

  /**
   * @brief Reset the (local) parameterisation of a parameter block.
   * @param[in] parameter_block_id The ID of the parameter block in question.
   * @param[in] parameterization ceres_backend::Parameterization to tell how to do the
   *            local parameterisation.
   * @return True on success.
   */
  bool resetParameterization(uint64_t parameter_block_id, int parameterization);

  /**
   * @brief Set the (local) parameterisation of a parameter block.
   * @param[in] parameter_block_id The ID of the parameter block in question.
   * @param[in] local_parameterization Give it an actual local parameterisation object.
   * @return True on success.
   */
  bool setParameterization(
      uint64_t parameter_block_id,
      ceres::LocalParameterization* local_parameterization);

  /**
   * @brief Set the (local) parameterisation of a parameter block.
   * @param[in] parameter_block The pointer to the parameter block in question.
   * @param[in] local_parameterization Give it an actual local parameterisation object.
   * @return True on success.
   */
  bool setParameterization(
      std::shared_ptr<ceres_backend::ParameterBlock> parameter_block,
      ceres::LocalParameterization* local_parameterization)
  {
    return setParameterization(parameter_block->id(), local_parameterization);
  }

  /// @}

  /// @name Getters
  /// @{

  /// @brief Get a shared pointer to a parameter block.
  std::shared_ptr<ceres_backend::ParameterBlock> parameterBlockPtr(
      uint64_t parameter_block_id);  // get a vertex

  /// @brief Get a shared pointer to a parameter block.
  std::shared_ptr<const ceres_backend::ParameterBlock> parameterBlockPtr(
      uint64_t parameter_block_id) const;  // get a vertex

  /// @brief Get a shared pointer to an error term.
  std::shared_ptr<ceres_backend::ErrorInterface> errorInterfacePtr(
      ceres::ResidualBlockId residual_block_id);  // get a vertex

  /// @brief Get a shared pointer to an error term.
  std::shared_ptr<const ceres_backend::ErrorInterface> errorInterfacePtr(
      ceres::ResidualBlockId residual_block_id) const;  // get a vertex

  /// @brief Get the residual blocks of a parameter block.
  /// @param[in] parameter_block_id The ID of the parameter block in question.
  /// @return Infos about all the residual blocks connected.
  ResidualBlockCollection residuals(uint64_t parameter_block_id) const;

  /// @brief Get the parameters of a residual block.
  /// @param[in] residual_block_id The ID of the residual block in question.
  /// @return Infos about all the parameter blocks connected.
  ParameterBlockCollection parameters(
      ceres::ResidualBlockId residual_block_id) const;  // get the parameter blocks connected

  /// @}

  // Jacobian checker
  /**
   * @brief Check a Jacobian with numeric differences.
   * @warning Checks the minimal version only.
   * @param[in] residual_block_id The ID of the residual block to be checked.
   * @param[in] relTol Relative numeric tolerance.
   * @return True if correct.
   */
  bool isMinimalJacobianCorrect(ceres::ResidualBlockId residual_block_id,
                                double relTol = 1e-6) const;

  // access to the map as such
  /// \brief The actual map from Id to parameter block pointer.
  typedef std::unordered_map<uint64_t,
      std::shared_ptr<ceres_backend::ParameterBlock> > IdToParameterBlockMap;

  /// \brief The actual map from Id to residual block specs.
  typedef std::unordered_map<ceres::ResidualBlockId, ResidualBlockSpec>
  ResidualBlockIdToResidualBlockSpecMap;

  /// @brief Get map connecting parameter block IDs to parameter blocks
  const IdToParameterBlockMap& idToParameterBlockMap() const
  {
    return id_to_parameter_block_map_;
  }
  /// @brief Get the actual map from Id to residual block specs.
  const ResidualBlockIdToResidualBlockSpecMap&
  residualBlockIdToResidualBlockSpecMap() const
  {
    return residual_block_id_to_residual_block_spec_map_;
  }

  // these are public for convenient manipulation
  /// \brief Ceres options
  ceres::Solver::Options options;

  /// \brief Ceres optimization summary
  ceres::Solver::Summary summary;

  /// @brief Solve the optimization problem.
  void solve()
  {
    Solve(options, problem_.get(), &summary);
  }

 protected:

  /// \brief count the inserted residual blocks.
  uint64_t residual_counter_;

  // member variables related to optimization
  /// \brief The ceres problem
  std::shared_ptr<ceres::Problem> problem_;

  // the actual maps
  /// \brief Go from Id to residual block pointer.
  typedef std::unordered_multimap<uint64_t,
  ResidualBlockSpec> IdToResidualBlockMultimap;

  /// \brief Go from residual block id to its parameter blocks.
  typedef std::unordered_map<ceres::ResidualBlockId,
      ParameterBlockCollection> ResidualBlockIdToParameterBlockCollectionMap;

  /// \brief The map connecting parameter block ID's and parameter blocks
  IdToParameterBlockMap id_to_parameter_block_map_;

  /// \brief Go from residual ID to specs.
  ResidualBlockIdToResidualBlockSpecMap residual_block_id_to_residual_block_spec_map_;

  /// \brief Go from Id to residual block pointer.
  IdToResidualBlockMultimap id_to_residual_block_multimap_;

  /// \brief Go from residual block id to its parameter blocks.
  ResidualBlockIdToParameterBlockCollectionMap
  residual_block_id_to_parameter_block_collection_map_;

  /// \brief Store parameterisation locally.
  ceres_backend::HomogeneousPointLocalParameterization
  homogeneous_point_local_parameterization_;

  /// \brief Store parameterisation locally.
  ceres_backend::PoseLocalParameterization pose_local_parameterization_;
};

}  //namespace svo
}  //namespace ceres_backend
