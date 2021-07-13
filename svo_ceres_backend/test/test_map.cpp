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
 *  Created on: Sep 3, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

#include <memory>
#include <sys/time.h>
#include <ceres/ceres.h>

#include <vikit/cameras.h>
#include <vikit/cameras/camera_factory.h>
#include <aslam/common/entrypoint.h>
#include <svo/common/camera.h>
#include <svo/vio_common/test_utils.hpp>

#include "svo/ceres_backend/map.hpp"
#include "svo/ceres_backend/reprojection_error.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"
#include "svo/ceres_backend/pose_local_parameterization.hpp"
#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/homogeneous_point_parameter_block.hpp"

TEST(okvisTestSuite, Map)
{
  constexpr size_t num_points = 1000;
  constexpr bool deterministic = true;
  constexpr double keypoint_sigma = 1.0;

  // set up a random geometry
  std::cout << "set up a random geometry... " << std::flush;
  svo::Transformation T_WS;  // world to sensor
  T_WS.setRandom(10.0, M_PI);
  svo::Transformation T_disturb;
  T_disturb.setRandom(1, 0.01);
  svo::Transformation T_WS_init = T_WS * T_disturb;  // world to sensor
  svo::Transformation T_SC;  // sensor to camera
  T_SC.setRandom(0.2, M_PI);
  std::shared_ptr<svo::ceres_backend::PoseParameterBlock> poseParameterBlock_ptr(
        new svo::ceres_backend::PoseParameterBlock(T_WS_init, 1));
  std::shared_ptr<svo::ceres_backend::PoseParameterBlock> extrinsicsParameterBlock_ptr(
        new svo::ceres_backend::PoseParameterBlock(T_SC, 2));

  // use the custom graph/map datastructure now:
  svo::ceres_backend::Map map;
  map.addParameterBlock(poseParameterBlock_ptr, svo::ceres_backend::Map::Pose6d);
  map.addParameterBlock(extrinsicsParameterBlock_ptr, svo::ceres_backend::Map::Pose6d);
  map.setParameterBlockConstant(extrinsicsParameterBlock_ptr);  // do not optimize this...
  std::cout << " [ OK ] " << std::endl;

  // set up a random camera geometry
  std::cout << "set up a random camera geometry... " << std::flush;
  Eigen::VectorXd pinhole_intrin(4);
  double f = 315.5;
  pinhole_intrin << f, f, 376.0, 240.0;
  svo::CameraPtr cameraGeometry =
      vk::cameras::factory::makePinholeCamera(pinhole_intrin, 752, 480);
  std::cout << " [ OK ] " << std::endl;

  // get some random points and build error terms
  std::cout << "create N=" << num_points
            << " visible points and add respective reprojection error terms... "
            << std::flush;

  Eigen::Matrix<svo::FloatType, 4, Eigen::Dynamic> points_homogeneous(4, num_points);
  points_homogeneous.row(3).setOnes();
  Eigen::Ref<svo::Positions> points =
      points_homogeneous.topLeftCorner<3, num_points>();
  svo::Keypoints keypoints(2, num_points);
  std::cout << "|got here|";

  std::tie(keypoints, std::ignore, points) =
      svo::test_utils::generateRandomVisible3dPoints(*cameraGeometry, num_points, 10.0, 10.0);

  keypoints += svo::test_utils::randomMatrixNormalDistributed<2, num_points>(
        deterministic, 0.0, keypoint_sigma);

  ceres::CauchyLoss loss(1);
  const Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  for (size_t i = 0; i < num_points; ++i)
  {
    std::shared_ptr<svo::ceres_backend::HomogeneousPointParameterBlock>
        homogeneousPointParameterBlock_ptr =
        std::make_shared<svo::ceres_backend::HomogeneousPointParameterBlock>(
          (T_WS * T_SC).transform4(static_cast<Eigen::Vector4d>(
                                     points_homogeneous.col(i))),
          i + 3);
    map.addParameterBlock(homogeneousPointParameterBlock_ptr,
                          svo::ceres_backend::Map::HomogeneousPoint);
    map.setParameterBlockConstant(homogeneousPointParameterBlock_ptr);  // no point optimization
    // Set up the only cost function (also known as residual).
    std::shared_ptr<ceres::CostFunction> cost_function(
          new svo::ceres_backend::ReprojectionError(
            cameraGeometry, keypoints.col(i), information));
    ceres::ResidualBlockId id = map.addResidualBlock(
          cost_function, &loss, poseParameterBlock_ptr,
          homogeneousPointParameterBlock_ptr, extrinsicsParameterBlock_ptr);
    EXPECT_TRUE(map.isMinimalJacobianCorrect(id)) << "wrong Jacobian";

    if (i % 10 == 0)
    {
      if (i % 20 == 0)
        map.removeParameterBlock(homogeneousPointParameterBlock_ptr);  // randomly delete some just for fun to test
      else
        map.removeResidualBlock(id);  // randomly delete some just for fun to test
    }
  }
  std::cout << " [ OK ] " << std::endl;

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  //map.options.check_gradients=true;
  //map.options.numeric_derivative_relative_step_size = 1e-6;
  //map.options.gradient_check_relative_precision=1e-2;
  map.options.minimizer_progress_to_stdout = true;
  map.options.max_num_iterations = 10;
  ::FLAGS_stderrthreshold = google::WARNING;  // enable console warnings (Jacobian verification)
  map.solve();

  // print some infos about the optimization
  //std::cout << map.summary.FullReport() << "\n";
  std::cout << "initial T_WS : " << T_WS_init.getTransformationMatrix() << "\n" << "optimized T_WS : "
            << poseParameterBlock_ptr->estimate().getTransformationMatrix() << "\n"
            << "correct T_WS : " << T_WS.getTransformationMatrix() << "\n";

  // make sure it converged
  EXPECT_LT(
        2*(T_WS.getEigenQuaternion() * poseParameterBlock_ptr->estimate().getEigenQuaternion().inverse()).vec().norm(),
        1e-2)  << "quaternions not close enough";
  EXPECT_LT(
        (T_WS.getPosition() - poseParameterBlock_ptr->estimate().getPosition()).norm(), 1e-1)
      << "translation not close enough";

  // also try out the resetting of parameterization:
  /*map.resetParameterization(poseParameterBlock_ptr->id(), svo::ceres_backend::Map::Pose2d);
  okvis::kinematics::Transformation T_start = poseParameterBlock_ptr->estimate();
  Eigen::Matrix<double, 6, 1> disturb_rp;
  disturb_rp.setZero();
  disturb_rp.segment<2>(3).setRandom();
  disturb_rp.segment<2>(3) *= 0.01;
  T_start.oplus(disturb_rp);
  poseParameterBlock_ptr->setEstimate(T_start);  // disturb again, so it has to optimize
  map.solve();*/
  //std::cout << map.summary.FullReport() << "\n";

}

VIKIT_UNITTEST_ENTRYPOINT
