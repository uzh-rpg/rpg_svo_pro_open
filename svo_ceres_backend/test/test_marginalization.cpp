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
 *  Created on: Sep 16, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

#include <memory>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <vikit/cameras.h>
#include <vikit/cameras/camera_factory.h>
#include <svo/common/camera.h>
#include <svo/vio_common/test_utils.hpp>

#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/marginalization_error.hpp"
#include "svo/ceres_backend/pose_error.hpp"
#include "svo/ceres_backend/pose_local_parameterization.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"
#include "svo/ceres_backend/reprojection_error.hpp"
#include "svo/ceres_backend/speed_and_bias_error.hpp"
#include "svo/ceres_backend/speed_and_bias_parameter_block.hpp"
#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/homogeneous_point_parameter_block.hpp"


//! @todo this test does not actually marginalize...
TEST(okvisTestSuite, Marginalization){
  constexpr size_t num_points = 100;
  constexpr bool deterministic = true;
  constexpr double keypoint_sigma = 0.0;

  // set up a random geometry
  std::cout << "set up a random geometry... " << std::flush;
  svo::Transformation T_WS0;  // world to sensor
  T_WS0.setRandom(10.0, M_PI);
  svo::Transformation T_S0S1;
  T_S0S1.setRandom(1.0, 0.01);
  svo::Transformation T_S1S2;
  T_S1S2.setRandom(1.0, 0.01);
  svo::Transformation T_WS1 = T_WS0 * T_S0S1;  // world to sensor
  svo::Transformation T_WS2 = T_WS1 * T_S1S2;  // world to sensor
  svo::Transformation T_disturb;
  T_disturb.setRandom(1, 0.01);
  svo::Transformation T_WS2_init = T_WS2 * T_disturb;  // world to sensor
  svo::Transformation T_SC;  // sensor to camera
  T_SC.setRandom(0.2, M_PI);
  std::shared_ptr<svo::ceres_backend::PoseParameterBlock> poseParameterBlock0_ptr(
      new svo::ceres_backend::PoseParameterBlock(T_WS0, 0));
  std::shared_ptr<svo::ceres_backend::PoseParameterBlock> poseParameterBlock1_ptr(
      new svo::ceres_backend::PoseParameterBlock(T_WS1, 1));
  std::shared_ptr<svo::ceres_backend::PoseParameterBlock> poseParameterBlock2_ptr(
      new svo::ceres_backend::PoseParameterBlock(T_WS2_init, 2));
  std::shared_ptr<svo::ceres_backend::PoseParameterBlock> extrinsicsParameterBlock_ptr(
      new svo::ceres_backend::PoseParameterBlock(T_SC, 3));

  // use the custom graph/map data structure now:
  svo::ceres_backend::Map map;
  map.addParameterBlock(poseParameterBlock0_ptr, svo::ceres_backend::Map::Pose6d);
  map.setParameterBlockConstant(poseParameterBlock0_ptr);
  map.addParameterBlock(poseParameterBlock1_ptr, svo::ceres_backend::Map::Pose6d);
  map.setParameterBlockConstant(poseParameterBlock1_ptr);
  map.addParameterBlock(poseParameterBlock2_ptr, svo::ceres_backend::Map::Pose6d);
  map.addParameterBlock(extrinsicsParameterBlock_ptr, svo::ceres_backend::Map::Pose6d);
  //map.setParameterBlockConstant(extrinsicsParameterBlock_ptr);
  std::cout << " [ OK ] " << std::endl;

  // set up a random camera geometry
  std::cout << "set up a random camera geometry... " << std::flush;
  Eigen::VectorXd pinhole_intrin(4);
  double f = 315.5;
  pinhole_intrin << f, f, 376.0, 240.0;
  svo::CameraPtr cameraGeometry =
      vk::cameras::factory::makePinholeCamera(pinhole_intrin, 752, 480);
  std::cout << " [ OK ] " << std::endl;

  // push the residual blocks to be removed in here:
  std::vector<ceres::ResidualBlockId> residualBlockIds;

  // add a prior to the extrinsics
  std::shared_ptr<ceres::CostFunction> extrinsics_prior_cost(
      new svo::ceres_backend::PoseError(T_SC, 1e-4, 1e-4));
  ceres::ResidualBlockId id = map.addResidualBlock(
      extrinsics_prior_cost, NULL, extrinsicsParameterBlock_ptr);
  residualBlockIds.push_back(id);

  // add a prior to the poses
  //std::shared_ptr< ceres::CostFunction> pose0_prior_cost(new svo::ceres_backend::PoseError(T_WS0, 1e-4, 1e-4));
  //id = map.addResidualBlock(pose0_prior_cost, NULL,poseParameterBlock0_ptr);
  //residualBlockIds.push_back(id);
  //std::shared_ptr< ceres::CostFunction> pose1_prior_cost(new svo::ceres_backend::PoseError(T_WS1, 1e-4, 1e-4));
  //id = map.addResidualBlock(pose1_prior_cost, NULL,poseParameterBlock1_ptr);
  //residualBlockIds.push_back(id);

  // get some random points and build error terms
  std::vector<uint64_t> marginalizationParametersBlocksIds;
  std::cout << "create N=" << num_points
            << " visible points and add respective reprojection error terms... "
            << std::flush;

  Eigen::Matrix<svo::FloatType, 4, Eigen::Dynamic, Eigen::ColMajor>
      points_homogeneous(4, num_points);
  points_homogeneous.row(3).setOnes();
  Eigen::Ref<svo::Positions> points =
      points_homogeneous.topLeftCorner<3, num_points>();
  svo::Keypoints keypoints(2, num_points);
  std::tie(keypoints, std::ignore, points) =
      svo::test_utils::generateRandomVisible3dPoints(
        *cameraGeometry, num_points, 10.0, 10.0);
  keypoints +=
      svo::test_utils::randomMatrixNormalDistributed<2, num_points>(
        deterministic, 0.0, keypoint_sigma);

  for (size_t i = 0; i < num_points; ++i) {
    // points in camera frames:
    Eigen::Vector4d pointC0 = points_homogeneous.col(i);
    Eigen::Vector4d pointC1 =
        (T_SC.inverse() * T_WS1.inverse() * T_WS0 * T_SC).transform4(pointC0);
    Eigen::Vector4d pointC2 =
        (T_SC.inverse() * T_WS2.inverse() * T_WS0 * T_SC).transform4(pointC0);

    std::shared_ptr<svo::ceres_backend::HomogeneousPointParameterBlock>
        homogeneousPointParameterBlock_ptr =
        std::make_shared<svo::ceres_backend::HomogeneousPointParameterBlock>(
            (T_WS0 * T_SC).transform4(pointC0), i + 4);
    map.addParameterBlock(homogeneousPointParameterBlock_ptr,
                          svo::ceres_backend::Map::HomogeneousPoint);

    // Set up cost function
    Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
    std::shared_ptr< ceres::CostFunction> cost_function0(
        new svo::ceres_backend::ReprojectionError(
            std::static_pointer_cast<const svo::Camera>(cameraGeometry),
            keypoints.col(i), information));

    ceres::ResidualBlockId id0 = map.addResidualBlock(
        cost_function0, NULL, poseParameterBlock0_ptr,
        homogeneousPointParameterBlock_ptr, extrinsicsParameterBlock_ptr);

    residualBlockIds.push_back(id0);

    // get a randomized projections
    Eigen::Vector2d kp1;
    bool success;
    //project homogeneous coordinates
    if (pointC1[3] < 0.0)
    {
      success = cameraGeometry->project3(-pointC1.head<3>(),&kp1).isKeypointVisible();
    }
    else
    {
      success = cameraGeometry->project3(pointC1.head<3>(),&kp1).isKeypointVisible();
    }
    if (success) {
      kp1 += svo::test_utils::randomVectorNormalDistributed<2>(
            deterministic, 0.0, keypoint_sigma);

      // Set up cost function
      Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
      std::shared_ptr< ceres::CostFunction> cost_function1(
          new svo::ceres_backend::ReprojectionError(
              std::static_pointer_cast<const svo::Camera>(cameraGeometry),
              kp1, information));

      ceres::ResidualBlockId id1 = map.addResidualBlock(
          cost_function1, NULL, poseParameterBlock1_ptr,
          homogeneousPointParameterBlock_ptr, extrinsicsParameterBlock_ptr);

      residualBlockIds.push_back(id1);
    }

    // get a randomized projections
    Eigen::Vector2d kp2;
    //project homogeneous coordinates
    if (pointC2[3] < 0.0)
    {
      success = cameraGeometry->project3(-pointC2.head<3>(),&kp2).isKeypointVisible();
    }
    else
    {
      success = cameraGeometry->project3(pointC2.head<3>(),&kp2).isKeypointVisible();
    }
    if (success) {
      kp2 += svo::test_utils::randomVectorNormalDistributed<2>(
            deterministic, 0.0, keypoint_sigma);

      // Set up cost function
      Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
      std::shared_ptr< ceres::CostFunction> cost_function2(
          new svo::ceres_backend::ReprojectionError(
              std::static_pointer_cast<const svo::Camera>(cameraGeometry),
              kp2, information));

      ceres::ResidualBlockId id2 = map.addResidualBlock(
          cost_function2, NULL, poseParameterBlock2_ptr,
          homogeneousPointParameterBlock_ptr, extrinsicsParameterBlock_ptr);

      residualBlockIds.push_back(id2);
    }

    marginalizationParametersBlocksIds.push_back(
        homogeneousPointParameterBlock_ptr->id());
  }
  std::cout << " [ OK ] " << std::endl;

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  map.options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold = google::WARNING;  // enable console warnings (Jacobian verification)
  map.solve();

  // print some infos about the optimization
  std::cout << map.summary.FullReport() << "\n";
  std::cout << "initial T_WS2 :\n"
            << T_WS2_init.getTransformationMatrix() << "\n"
            << "optimized T_WS2 :\n"
            << poseParameterBlock2_ptr->estimate().getTransformationMatrix()
            << "\n" << "correct T_WS:\n "
            << T_WS2.getTransformationMatrix() << "\n";

  // make sure it converged
  EXPECT_LT(
      2 * (T_WS2.getEigenQuaternion() * poseParameterBlock2_ptr->estimate()
           .getEigenQuaternion().inverse()).vec().norm(),
      1e-2)
      << "quaternions not close enough";
  EXPECT_LT(
      (T_WS2.getPosition() - poseParameterBlock2_ptr->estimate()
       .getPosition()).norm(), 1e-1)
      << "translation not close enough";
}

VIKIT_UNITTEST_ENTRYPOINT
