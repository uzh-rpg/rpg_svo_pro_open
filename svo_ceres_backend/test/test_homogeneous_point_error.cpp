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
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

#include <memory>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <svo/vio_common/test_utils.hpp>

#include "svo/ceres_backend/homogeneous_point_error.hpp"
#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/homogeneous_point_parameter_block.hpp"
#include "svo/ceres_backend/map.hpp"


TEST(okvisTestSuite, HomogeneousPointError)
{
  constexpr bool deterministic = true;
  constexpr size_t n_points = 100;
  constexpr double jacobian_rel_tol = 1e-6;

  // Build the problem.
  svo::ceres_backend::Map map;

  Eigen::Matrix<svo::FloatType, 4, Eigen::Dynamic, Eigen::ColMajor> points =
      svo::test_utils::randomMatrixNormalDistributed<4, n_points>(deterministic,
                                                                  0.0, 100.0);
  points.bottomLeftCorner<1, n_points>().setZero();
  for (size_t i = 0; i < n_points; ++i)
  {
    Eigen::Vector4d point = points.col(i);

    // create parameter block
    std::shared_ptr<svo::ceres_backend::HomogeneousPointParameterBlock>
        homogeneousPointParameterBlock(
          new svo::ceres_backend::HomogeneousPointParameterBlock(point, i));
    // add it as optimizable thing.
    map.addParameterBlock(homogeneousPointParameterBlock,
                          svo::ceres_backend::Map::HomogeneousPoint);
    map.setParameterBlockVariable(i);

    // invent a point error
    std::shared_ptr<svo::ceres_backend::HomogeneousPointError> homogeneousPointError(
        new svo::ceres_backend::HomogeneousPointError(
            homogeneousPointParameterBlock->estimate(), 0.1));

    // add it
    ceres::ResidualBlockId id = map.addResidualBlock(
        homogeneousPointError, nullptr, homogeneousPointParameterBlock);

    // disturb
    Eigen::Vector4d point_disturbed = point;
    point_disturbed.head<3>() += 0.2 * Eigen::Vector3d::Random();
    homogeneousPointParameterBlock->setEstimate(point_disturbed);

    // check Jacobian

    EXPECT_TRUE(map.isMinimalJacobianCorrect(id, jacobian_rel_tol))
        << "Jacobian verification on homogeneous point error failed.";
  }

  // Run the solver!
  map.options.minimizer_progress_to_stdout = false;
  std::cout << "run the solver... " << std::endl;
  map.solve();

  // check convergence. this must converge to zero, since it is not an overdetermined system.
  EXPECT_TRUE(map.summary.final_cost < 1.0e-10)
      << "No convergence. this must converge to zero, since it is not an overdetermined system.";
}

VIKIT_UNITTEST_ENTRYPOINT
