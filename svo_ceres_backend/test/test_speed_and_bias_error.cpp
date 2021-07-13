// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <svo/vio_common/test_utils.hpp>

#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/speed_and_bias_error.hpp"
#include "svo/ceres_backend/speed_and_bias_parameter_block.hpp"

TEST(okvisTestSuite, SpeedAndBiasError)
{
  constexpr bool deterministic = true;
  constexpr size_t n_terms = 10;
  constexpr double jacobian_rel_tol = 1e-6;

  svo::ceres_backend::Map map;
  for (size_t i = 0; i < n_terms; ++i)
  {
    svo::SpeedAndBias speed_and_bias =
        svo::test_utils::randomVectorUniformDistributed<9>(deterministic);

    // create parameter block
    std::shared_ptr<svo::ceres_backend::SpeedAndBiasParameterBlock>
        speedandbias_parameter_block =
        std::make_shared<svo::ceres_backend::SpeedAndBiasParameterBlock>(speed_and_bias, i);
    // add it as optimizable thing.
    map.addParameterBlock(speedandbias_parameter_block,
                          svo::ceres_backend::Map::Trivial);
    map.setParameterBlockVariable(i);
    // invent an error
    std::shared_ptr<svo::ceres_backend::SpeedAndBiasError> speedandbias_error =
        std::make_shared<svo::ceres_backend::SpeedAndBiasError>(
          speed_and_bias, 1.0, 1.0, 1.0);
    // add it
    ceres::ResidualBlockId id = map.addResidualBlock(
        speedandbias_error, nullptr, speedandbias_parameter_block);
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
