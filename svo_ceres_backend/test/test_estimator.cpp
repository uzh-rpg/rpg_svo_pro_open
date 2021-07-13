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

#include <gtest/gtest.h>

#include <vikit/cameras.h>
#include <vikit/cameras/camera_factory.h>
#include <aslam/common/entrypoint.h>
#include <svo/common/camera.h>
#include <svo/common/conversions.h>
#include <svo/common/transformation.h>
#include <svo/common/frame.h>
#include <svo/common/imu_calibration.h>
#include <svo/vio_common/test_utils.hpp>
#include <opencv2/core/core.hpp>

#include "svo/ceres_backend/estimator.hpp"

TEST(okvisTestSuite, Estimator) {
  // ---------------------------------------------------------------------------
  // Parameters.
  constexpr double motion_duration = 10.0;  // 10 seconds motion
  constexpr double motion_speed_y = 1.0;
  constexpr double imu_rate = 100.0;  // 100 Hz
  constexpr double dt = 1.0 / imu_rate;  // time increments
  constexpr size_t imu_samples = motion_duration * imu_rate;
  constexpr bool deterministic = true;
  constexpr double keypoint_measurement_sigma = 0.75;
  constexpr size_t max_features = 300;

  const double t0 = 0;

  // set the imu parameters
  svo::ImuParameters imu_parameters;
  imu_parameters.a0.setZero();
  imu_parameters.g = 9.81;
  imu_parameters.a_max = 1000.0;
  imu_parameters.g_max = 1000.0;
  imu_parameters.rate = 1000;  // 1 kHz
  imu_parameters.sigma_g_c = 6.0e-4;
  imu_parameters.sigma_a_c = 2.0e-3;
  imu_parameters.sigma_gw_c = 3.0e-6;
  imu_parameters.sigma_aw_c = 2.0e-5;
  imu_parameters.sigma_ba = 0.1;
  imu_parameters.sigma_bg = 0.03;
  imu_parameters.delay_imu_cam = 0.0;

  // ---------------------------------------------------------------------------
  // Camera rig setup.
  vk::TransformationVector T_CB_i;
  T_CB_i.emplace_back();
  T_CB_i.emplace_back(Eigen::Vector3d(0.0, 0.1, 0.0), svo::Quaternion());
  std::vector<svo::CameraPtr> cameras(2);
  Eigen::VectorXd pinhole_intrin(4);
  double f = 329.11;
  pinhole_intrin << f, f, 320.0, 240.0;
  cameras[0] = vk::cameras::factory::makePinholeCamera(pinhole_intrin, 640, 480);
  cameras[1] = vk::cameras::factory::makePinholeCamera(pinhole_intrin, 640, 480);
  svo::CameraBundlePtr camera_rig = std::make_shared<svo::CameraBundle>(
        T_CB_i, cameras, "test rig");

  // ---------------------------------------------------------------------------
  // Generate IMU measurements
  // let's generate a really stupid motion: constant translation
  svo::SpeedAndBias speed_and_bias;
  speed_and_bias.setZero();
  speed_and_bias[1] = motion_speed_y;
  svo::ImuMeasurements imu_measurements(imu_samples+1);
  Eigen::Matrix<double, 6, 1> nominal_imu_sensor_readings;
  nominal_imu_sensor_readings
      << Eigen::Vector3d(0.0, 0.0, imu_parameters.g), Eigen::Vector3d::Zero();
  for (size_t i = imu_samples; i <=imu_samples; --i)
  {
    Eigen::Vector3d gyr = nominal_imu_sensor_readings.tail<3>()
        + svo::test_utils::randomVectorNormalDistributed<3>(
          deterministic, 0.0, imu_parameters.sigma_g_c * std::sqrt(dt));
    Eigen::Vector3d acc = nominal_imu_sensor_readings.head<3>()
        + svo::test_utils::randomVectorNormalDistributed<3>(
          deterministic, 0.0, imu_parameters.sigma_a_c * std::sqrt(dt));
    imu_measurements[i].timestamp_ = t0 + dt * (imu_samples-i);
    imu_measurements[i].linear_acceleration_ << acc;
    imu_measurements[i].angular_velocity_ << gyr;
  }


  // different cases of camera extrinsics;
  for (size_t extrinsics_case = 0; extrinsics_case < 4; ++extrinsics_case)
  {
    LOG(INFO) << "case " << extrinsics_case % 2 << ", " << extrinsics_case / 2;

    // -------------------------------------------------------------------------
    // Estimator setup.

    // some parameters on how to do the online estimation:
    svo::ExtrinsicsEstimationParameters extrinsics_estimation_parameters;
    extrinsics_estimation_parameters.sigma_absolute_translation = 1.0e-3
        * (extrinsics_case % 2);
    extrinsics_estimation_parameters.sigma_absolute_orientation = 1.0e-4
        * (extrinsics_case % 2);
    extrinsics_estimation_parameters.sigma_c_relative_translation = 1e-8
        * (extrinsics_case / 2);
    extrinsics_estimation_parameters.sigma_c_relative_orientation = 1e-7
        * (extrinsics_case / 2);
    svo::ExtrinsicsEstimationParametersVec
        extrinsics_estimation_parameters_vec(2, extrinsics_estimation_parameters);


    // create an Estimator
    svo::Estimator estimator;

    // add sensors
    estimator.addCameraBundle(extrinsics_estimation_parameters_vec, camera_rig);
    estimator.addImu(imu_parameters);

    //! @todo okvis has unique id's for landmarks or multiframes (states), i.e. all
    //! landmark handle have to be different from nframe handles. The id is used
    //! in a map relating these id's with ceres parameterblock ids. In our case
    //! we either need multiple maps/tables for landmarks and states or we
    //! have to somehow differentiate them (bitfield?)
    uint32_t id_counter = 0;

    // -------------------------------------------------------------------------
    // create landmark grid
    const svo::Transformation T_WS_0_okvis;
    std::vector<Eigen::Vector3d,
        Eigen::aligned_allocator<Eigen::Vector3d> > landmark_positions;
    std::vector<int> track_ids;
    const double y_end = motion_duration * motion_speed_y + 10.0;
    for (double y = -10.0; y <= y_end; y += 0.5)
    {
      for (double z = -10.0; z <= 10.0; z += 0.5)
      {
        landmark_positions.push_back(Eigen::Vector3d(3.0, y, z));
        track_ids.emplace_back(id_counter++);
        svo::PointPtr point =
            std::make_shared<svo::Point>(track_ids.back(),
                                         landmark_positions.back());
        bool success =
            estimator.addLandmark(point);
        CHECK(success) << "Could not add landmark.";
      }
    }

    // -------------------------------------------------------------------------
    // Adding frames and observations to estimator. Also optimizing at each timestep.
    const size_t num_frames = 6;
    svo::Transformation T_WS_est;
    svo::SpeedAndBias speed_and_bias_est;
    for (size_t k = 0; k < num_frames + 1; ++k)
    {
      // calculate the transformation
      const double duration =
          static_cast<double>(k) * motion_duration /
          static_cast<double>(num_frames);
      const double time = t0 + duration;
      const int64_t time_ns = static_cast<int64_t>(
            time / svo::common::conversions::kNanoSecondsToSeconds);
      const Eigen::Vector3d position = speed_and_bias.head<3>() * duration;
      svo::Transformation T_WB(position, svo::Quaternion());

      // assemble an n-frame
      const cv::Size image_size(640, 480);
      std::vector<svo::FramePtr> frames;
      frames.push_back(std::make_shared<svo::Frame>(
                         camera_rig->getCameraShared(0),
                         cv::Mat(image_size,CV_8UC1),time_ns,1));
      frames.push_back(std::make_shared<svo::Frame>(
                         camera_rig->getCameraShared(1),
                         cv::Mat(image_size,CV_8UC1),time_ns,1));
      svo::FrameBundlePtr nframe = std::make_shared<svo::FrameBundle>(frames);
      nframe->at(0)->setNFrameIndex(0u);
      nframe->at(1)->setNFrameIndex(1u);
      // add it in the window to create a new time instance
      bool success = estimator.addStates(
            nframe, imu_measurements,
            time);
      if(k % 3 == 0)
      {
        estimator.setKeyframe(svo::createNFrameId(nframe->getBundleId()),true);
      }
      CHECK(success) << "addStates() failed!";

      success = estimator.get_T_WS(nframe->getBundleId(), T_WS_est);
      CHECK(success) << "get_T_WS failed!";


      // now let's add also landmark observations
      for (size_t j = 0; j < landmark_positions.size(); ++j)
      {
        for (size_t i = 0; i < nframe->size(); ++i)
        {
          svo::FramePtr frame = nframe->at(i);
          frame->resizeFeatureStorage(max_features);
          const svo::Camera& cam = camera_rig->getCamera(i);
          const Eigen::Vector3d point_C =
              camera_rig->get_T_C_B(i) * T_WB.inverse() * landmark_positions[j];
          svo::Keypoint projection;
          //projectWithCheck(point_C);
          if(point_C[2]<0.0) continue;
          if (camera_rig->getCamera(i).project3(point_C,&projection)
              && frame->numFeatures() < max_features)
          {
            Eigen::Vector2d measurement =
                projection +
                svo::test_utils::randomVectorNormalDistributed<2>(
                  deterministic,
                  0.0,
                  keypoint_measurement_sigma);
            frame->px_vec_.col(frame->num_features_) = measurement;
            Eigen::Vector3d *bearing = new Eigen::Vector3d();
            success = cam.backProject3(Eigen::Ref<Eigen::Vector2d>(measurement)
                                       ,bearing);
            CHECK(success) << "backProject3 failed";
            frame->f_vec_.col(frame->num_features_) = *bearing;
            frame->level_vec_(frame->num_features_) = 1;
            frame->type_vec_[frame->num_features_] = svo::FeatureType::kCorner;
            frame->score_vec_(frame->num_features_) = 1.0;
            frame->track_id_vec_[frame->num_features_] = track_ids[j];

            //! @todo fill in keypoint angle and score.
            estimator.addObservation(frame, frame->numFeatures());
            ++frame->num_features_;
          }
        }
      }

      // run the optimization
      estimator.optimize(10, 4, false);
      LOG(INFO) << "Optimization done.";
    }

    LOG(INFO) << "== TRY MARGINALIZATION ==";
    // try out the marginalization strategy
    estimator.applyMarginalizationStrategy(2, 3);
    // run the optimization
    LOG(INFO) << "== LAST OPTIMIZATION ==";
    estimator.optimize(10, 4, false);

    // get the estimates
    estimator.get_T_WS(estimator.currentBundleId(), T_WS_est);
    estimator.getSpeedAndBias(estimator.currentFrameId(),
                              speed_and_bias_est);

    // inspect convergence:
    svo::Transformation T_WS(
          T_WS_0_okvis.getPosition() + speed_and_bias.head<3>() *motion_duration,
          T_WS_0_okvis.getEigenQuaternion());

    EXPECT_LT((speed_and_bias_est - speed_and_bias).norm(), 0.04)
        << "speed and biases not close enough";
    EXPECT_NEAR((T_WS.getEigenQuaternion() *
                 T_WS_est.getEigenQuaternion().inverse()).w(), 1.0, 1e-6)
        << "quaternions not close enough";
    EXPECT_LT((T_WS.getPosition() - T_WS_est.getPosition()).norm(), 1e-1)
        << "translation not close enough";
  }
}

VIKIT_UNITTEST_ENTRYPOINT
