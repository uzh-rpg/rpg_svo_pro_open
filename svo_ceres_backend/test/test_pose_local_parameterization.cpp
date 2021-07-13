// Copyright (C) ETH Zurich, Wyss Zurich, Zurich Eye - All Rights Reserved
// Unauthorized copying of this file, via any medium is strictly prohibited
// Proprietary and confidential

#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <svo/vio_common/test_utils.hpp>
#include <svo/common/transformation.h>
#include <svo/common/types.h>

#include "svo/ceres_backend/pose_local_parameterization.hpp"

TEST(poseLocalParametrizationTests, PlusTest)
{
  Eigen::Matrix<double, 7,1> x =
      svo::test_utils::randomVectorNormalDistributed<7>(true, 0.0, 10.0);
  x.tail<4>().normalize();
  Eigen::Matrix<svo::FloatType, 6, 1> dx =
      svo::test_utils::randomVectorNormalDistributed<6>(true, 0.0, 0.1);
  Eigen::Matrix<double, 7,1> x_dist;
  svo::ceres_backend::PoseLocalParameterization::plus(x.data(), dx.data(), x_dist.data());
  svo::Transformation T(
        Eigen::Vector3d(x[0], x[1], x[2]),
      Eigen::Quaterniond(x[6], x[3], x[4], x[5]));
  svo::Transformation T_dist(
        Eigen::Vector3d(x_dist[0], x_dist[1], x_dist[2]),
      Eigen::Quaterniond(x_dist[6], x_dist[3], x_dist[4], x_dist[5]));
  svo::Quaternion q_dist_with_ze =
      svo::Quaternion::exp(dx.tail<3>()) * T.getRotation();
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(T_dist.getRotationMatrix(),
                                        q_dist_with_ze.getRotationMatrix()));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(T_dist.getPosition(),
                                        T.getPosition() + dx.head<3>()));
}

TEST(poseLocalParametrizationTests, MinusTest)
{
  //! @todo this test fails. check minus implementation...
  Eigen::Matrix<double, 7,1> x =
      svo::test_utils::randomVectorNormalDistributed<7>(true, 0.0, 10.0);
  x.tail<4>().normalize();
  Eigen::Matrix<svo::FloatType, 6, 1> dx =
      svo::test_utils::randomVectorNormalDistributed<6>(true, 0.0, 0.1);
  Eigen::Matrix<double, 7,1> x_dist;
  svo::ceres_backend::PoseLocalParameterization::plus(x.data(), dx.data(), x_dist.data());

  Eigen::Matrix<svo::FloatType, 6, 1> dx_from_minus;
  svo::ceres_backend::PoseLocalParameterization::minus(x.data(), x_dist.data(),
                                              dx_from_minus.data());
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(dx, dx_from_minus));
}

TEST(poseLocalParametrizationTests, PlusJacobianTest)
{
  Eigen::Matrix<double, 7,1> x =
      svo::test_utils::randomVectorNormalDistributed<7>(false, 0.0, 10.0);
  x.tail<4>().normalize();
  Eigen::Matrix<svo::FloatType, 6, 1> dx =
      Eigen::Matrix<svo::FloatType, 6, 1>::Zero();
  Eigen::Matrix<double, 7, 6, Eigen::RowMajor> jacobian_analytic;
  svo::ceres_backend::PoseLocalParameterization::plusJacobian(
        x.data(), jacobian_analytic.data());

  const double delta = 1e-3;
  Eigen::Matrix<double, 7,1> result_plus, result_minus;
  Eigen::Matrix<double, 7, 6> jacobian_numeric;
  for (size_t i = 0; i < 6; ++i)
  {
    dx[i] = delta;
    svo::ceres_backend::PoseLocalParameterization::plus(
          x.data(), dx.data(), result_plus.data());
    dx[i] = -delta;
    svo::ceres_backend::PoseLocalParameterization::plus(
          x.data(), dx.data(), result_minus.data());
    jacobian_numeric.col(i) = (result_plus - result_minus) * 1.0 / (2.0 * delta);
  }
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(jacobian_analytic,
                                jacobian_numeric, 1e-7));

}

TEST(poseLocalParametrizationTests, LiftJacobianTest)
{
  Eigen::Matrix<double, 7, 1> x =
      svo::test_utils::randomVectorNormalDistributed<7>(false, 0.0, 10.0);
  x.tail<4>().normalize();

  Eigen::Matrix<double, 7, 6, Eigen::RowMajor> plus_jacobian;
  Eigen::Matrix<double, 6, 7, Eigen::RowMajor> lift_jacobian;

  svo::ceres_backend::PoseLocalParameterization::plusJacobian(x.data(),
                                                     plus_jacobian.data());
  svo::ceres_backend::PoseLocalParameterization::liftJacobian(x.data(),
                                                     lift_jacobian.data());
  Eigen::Matrix<double, 6, 6> identity;
  identity.setIdentity();
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL_DOUBLE(lift_jacobian * plus_jacobian, identity));
}

VIKIT_UNITTEST_ENTRYPOINT
