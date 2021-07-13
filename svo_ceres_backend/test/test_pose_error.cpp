// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <svo/vio_common/test_utils.hpp>
#include <svo/common/transformation.h>

#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/pose_error.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"

TEST(okvisTestSuite, PoseError)
{
  constexpr bool deterministic = true;
  constexpr size_t n_poses = 10;
  constexpr double jacobian_rel_tol = 1e-6;

  svo::ceres_backend::Map map;
  for (size_t i = 0; i < n_poses; ++i)
  {
    svo::Transformation T;
    T.setRandom(
          svo::test_utils::sampleUniformRealDistribution<double>(deterministic),
          svo::test_utils::sampleUniformRealDistribution<double>(deterministic,
                                                                 0.0, M_PI));
    // create parameter block
    std::shared_ptr<svo::ceres_backend::PoseParameterBlock> pose_parameter_block =
        std::make_shared<svo::ceres_backend::PoseParameterBlock>(T, i);
    // add it as optimizable thing.
    map.addParameterBlock(pose_parameter_block,
                          svo::ceres_backend::Map::Pose6d);
    map.setParameterBlockVariable(i);
    // invent a pose error
    Eigen::Matrix<svo::FloatType, 6, 1> dist =
        svo::test_utils::randomVectorNormalDistributed<6>(deterministic);
    svo::Transformation T_dist = svo::Transformation::exp(dist) * T;
    std::shared_ptr<svo::ceres_backend::PoseError> pose_error =
        std::make_shared<svo::ceres_backend::PoseError>(T_dist, 1.0, 1.0);
    // add it
    ceres::ResidualBlockId id = map.addResidualBlock(
          pose_error, nullptr, pose_parameter_block);
    // check Jacobian
    EXPECT_TRUE(map.isMinimalJacobianCorrect(id, jacobian_rel_tol))
        << "Jacobian verification on homogeneous point error failed.";
  }
  // Run the solver!
  map.options.minimizer_progress_to_stdout = false;
  map.solve();

  // check convergence. this must converge to zero, since it is not an overdetermined system.
  EXPECT_TRUE(map.summary.final_cost < 1.0e-10)
      << "No convergence. this must converge to zero, since it is not an overdetermined system.";
}

VIKIT_UNITTEST_ENTRYPOINT
