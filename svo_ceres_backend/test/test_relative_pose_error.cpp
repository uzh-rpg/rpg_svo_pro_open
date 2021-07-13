// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <svo/vio_common/test_utils.hpp>
#include <svo/common/transformation.h>

#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/relative_pose_error.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"

TEST(okvisTestSuite, RelativePoseError)
{
  constexpr bool deterministic = true;
  constexpr size_t n_poses = 10;
  constexpr double jacobian_rel_tol = 1e-6;

  svo::ceres_backend::Map map;
  size_t id = 0;
  for (size_t i = 0; i < n_poses; ++i)
  {
    svo::Transformation T1, T2;
    T1.setRandom(
          svo::test_utils::sampleUniformRealDistribution<double>(deterministic),
          svo::test_utils::sampleUniformRealDistribution<double>(
            deterministic, 0.0, M_PI));
    T2.setRandom(
          svo::test_utils::sampleUniformRealDistribution<double>(deterministic),
          svo::test_utils::sampleUniformRealDistribution<double>(
            deterministic, 0.0, M_PI));

    // create and add parameter blocks first constant, second variable
    std::shared_ptr<svo::ceres_backend::PoseParameterBlock> pose_parameter_block1 =
        std::make_shared<svo::ceres_backend::PoseParameterBlock>(T1, ++id);
    map.addParameterBlock(pose_parameter_block1,
                          svo::ceres_backend::Map::Pose6d);
    map.setParameterBlockConstant(id);

    std::shared_ptr<svo::ceres_backend::PoseParameterBlock> pose_parameter_block2 =
        std::make_shared<svo::ceres_backend::PoseParameterBlock>(T2, ++id);
    map.addParameterBlock(pose_parameter_block2,
                          svo::ceres_backend::Map::Pose6d);
    map.setParameterBlockVariable(id);

    // add a relative pose error
    std::shared_ptr<svo::ceres_backend::RelativePoseError> relative_pose_error =
        std::make_shared<svo::ceres_backend::RelativePoseError>(1.0, 1.0);
    // add it
    ceres::ResidualBlockId res_id = map.addResidualBlock(
          relative_pose_error, nullptr, pose_parameter_block1, pose_parameter_block2);
    // check Jacobian
    EXPECT_TRUE(map.isMinimalJacobianCorrect(res_id, jacobian_rel_tol))
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
