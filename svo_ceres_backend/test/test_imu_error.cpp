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
 *  Created on: Sep 4, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

#include <ceres/ceres.h>

#include <aslam/common/entrypoint.h>
#include <svo/common/imu_calibration.h>

#include "svo/ceres_backend/imu_error.hpp"
#include "svo/ceres_backend/pose_error.hpp"
#include "svo/ceres_backend/speed_and_bias_error.hpp"
#include "svo/ceres_backend/pose_parameter_block.hpp"
#include "svo/ceres_backend/speed_and_bias_parameter_block.hpp"
#include "svo/ceres_backend/pose_local_parameterization.hpp"
#include "svo/ceres_backend/homogeneous_point_local_parameterization.hpp"
#include "svo/ceres_backend/homogeneous_point_parameter_block.hpp"

double sinc_test(double x){
  if(fabs(x)>1e-10) {
    return sin(x)/x;
  }
  else{
    static const double c_2=1.0/6.0;
    static const double c_4=1.0/120.0;
    static const double c_6=1.0/5040.0;
    const double x_2 = x*x;
    const double x_4 = x_2*x_2;
    const double x_6 = x_2*x_2*x_2;
    return 1.0 - c_2*x_2 + c_4*x_4 - c_6*x_6;
  }
}

const double jacobianTolerance = 1.0e-3;

TEST(okvisTestSuite, ImuError)
{
  using namespace svo;
  // initialize random number generator
  //srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

  // Build the problem.
  ceres::Problem problem;

  // set the imu parameters
  ImuParameters imuParameters;
  imuParameters.a0.setZero();
  imuParameters.g = 9.81;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.rate = 1000; // 1 kHz
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
  imuParameters.sigma_aw_c = 2.0e-5;
  imuParameters.delay_imu_cam = 0.0;

  // generate random motion
  const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
  const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
  const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
  const double w_a_W_x = Eigen::internal::random(0.1,10.0);
  const double w_a_W_y = Eigen::internal::random(0.1,10.0);
  const double w_a_W_z = Eigen::internal::random(0.1,10.0);
  const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
  const double m_a_W_x = Eigen::internal::random(0.1,10.0);
  const double m_a_W_y = Eigen::internal::random(0.1,10.0);
  const double m_a_W_z = Eigen::internal::random(0.1,10.0);

  // generate randomized measurements - duration 10 seconds
  const double duration = 1.0;
  const size_t num_measurements = static_cast<size_t>(duration * imuParameters.rate);
  ImuMeasurements imu_measurements(num_measurements);
  Transformation T_WS;
  //T_WS.setRandom();

  // time increment
  const double dt=1.0/double(imuParameters.rate); // time discretization

  // states
  Eigen::Quaterniond q = T_WS.getEigenQuaternion();
  Eigen::Vector3d r = T_WS.getPosition();
  SpeedAndBias speedAndBias;
  speedAndBias.setZero();
  Eigen::Vector3d v=speedAndBias.head<3>();

  // start
  Transformation T_WS_0;
  SpeedAndBias speedAndBias_0;
  double t_0;

  // end
  Transformation T_WS_1;
  SpeedAndBias speedAndBias_1;
  double t_1;

  for(size_t i = num_measurements-1; i<num_measurements; --i)
  {
    double time_s = double(num_measurements-i)/imuParameters.rate;
    if (i==num_measurements-11){ // set this as starting pose
      T_WS_0 = T_WS;
      speedAndBias_0=speedAndBias;
      t_0=time_s;
    }
    if (i==9){ // set this as ending pose
      T_WS_1 = T_WS;
      speedAndBias_1=speedAndBias;
      t_1=time_s;
    }

    Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time_s+p_omega_S_x),
        m_omega_S_y*sin(w_omega_S_y*time_s+p_omega_S_y),
        m_omega_S_z*sin(w_omega_S_z*time_s+p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time_s+p_a_W_x),
          m_a_W_y*sin(w_a_W_y*time_s+p_a_W_y),
          m_a_W_z*sin(w_a_W_z*time_s+p_a_W_z));

    //omega_S.setZero();
    //a_W.setZero();

    Eigen::Quaterniond dq;

    // propagate orientation
    const double theta_half = omega_S.norm()*dt*0.5;
    const double sinc_theta_half = ceres_backend::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec()=sinc_theta_half*0.5*dt*omega_S;
    dq.w()=cos_theta_half;
    q = q * dq;

    // propagate speed
    v+=dt*a_W;

    // propagate position
    r+=dt*v;

    // T_WS
    T_WS = Transformation(r,q);

    // speedAndBias - v only, obviously, since this is the Ground Truth
    speedAndBias.head<3>()=v;

    // generate measurements
    Eigen::Vector3d gyr = omega_S + imuParameters.sigma_g_c/sqrt(dt)
        *Eigen::Vector3d::Random();
    Eigen::Vector3d acc = T_WS.inverse().getRotationMatrix()
        *(a_W+Eigen::Vector3d(0,0,imuParameters.g))
        + imuParameters.sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random();
    imu_measurements[i].timestamp_ = time_s;
    imu_measurements[i].linear_acceleration_ << acc;
    imu_measurements[i].angular_velocity_ << gyr;
  }

  // create the pose parameter blocks
  Transformation T_disturb;
  T_disturb.setRandom(1.0, 0.02);
  Transformation T_WS_1_disturbed = T_WS_1 * T_disturb;
  ceres_backend::PoseParameterBlock poseParameterBlock_0(T_WS_0, 0); // ground truth
  ceres_backend::PoseParameterBlock poseParameterBlock_1(T_WS_1_disturbed, 2); // disturbed...
  problem.AddParameterBlock(poseParameterBlock_0.parameters(),
                            ceres_backend::PoseParameterBlock::c_dimension);
  problem.AddParameterBlock(poseParameterBlock_1.parameters(),
                            ceres_backend::PoseParameterBlock::c_dimension);
  //problem.SetParameterBlockConstant(poseParameterBlock_0.parameters());

  // create the speed and bias
  ceres_backend::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(speedAndBias_0,1);
  ceres_backend::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(speedAndBias_1,3);
  problem.AddParameterBlock(speedAndBiasParameterBlock_0.parameters(),
                            ceres_backend::SpeedAndBiasParameterBlock::c_dimension);
  problem.AddParameterBlock(speedAndBiasParameterBlock_1.parameters(),
                            ceres_backend::SpeedAndBiasParameterBlock::c_dimension);

  // let's use our own local quaternion perturbation
  std::cout<<"setting local parameterization for pose... "<<std::flush;
  ceres::LocalParameterization* poseLocalParameterization =
      new ceres_backend::PoseLocalParameterization;
  problem.SetParameterization(poseParameterBlock_0.parameters(),
                              poseLocalParameterization);
  problem.SetParameterization(poseParameterBlock_1.parameters(),
                              poseLocalParameterization);
  std::cout<<" [ OK ] "<<std::endl;

  // create the Imu error term
  ceres_backend::ImuError* cost_function_imu =
      new ceres_backend::ImuError(imu_measurements, imuParameters,t_0, t_1);
  problem.AddResidualBlock(cost_function_imu, NULL,
                           poseParameterBlock_0.parameters(),
                           speedAndBiasParameterBlock_0.parameters(),
                           poseParameterBlock_1.parameters(),
                           speedAndBiasParameterBlock_1.parameters());

  // let's also add some priors to check this alongside
  ceres::CostFunction* cost_function_pose =
      new ceres_backend::PoseError(T_WS_0, 1e-12, 1e-4); // pose prior...
  problem.AddResidualBlock(cost_function_pose, NULL,
                           poseParameterBlock_0.parameters());
  ceres::CostFunction* cost_function_speedAndBias =
      new ceres_backend::SpeedAndBiasError(speedAndBias_0, 1e-12, 1e-12, 1e-12); // speed and biases prior...
  problem.AddResidualBlock(cost_function_speedAndBias, NULL,
                           speedAndBiasParameterBlock_0.parameters());

  // check Jacobians: only by manual inspection...
  // they verify pretty badly due to the fact that the information matrix
  // is also a function of the states
  double* parameters[4];
  parameters[0]=poseParameterBlock_0.parameters();
  parameters[1]=speedAndBiasParameterBlock_0.parameters();
  parameters[2]=poseParameterBlock_1.parameters();
  parameters[3]=speedAndBiasParameterBlock_1.parameters();
  double* jacobians[4];
  Eigen::Matrix<double,15,7,Eigen::RowMajor> J0;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J1;
  Eigen::Matrix<double,15,7,Eigen::RowMajor> J2;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J3;
  jacobians[0]=J0.data();
  jacobians[1]=J1.data();
  jacobians[2]=J2.data();
  jacobians[3]=J3.data();
  double* jacobians_minimal[4];
  Eigen::Matrix<double,15,6,Eigen::RowMajor> J0min;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J1min;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> J2min;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J3min;
  jacobians_minimal[0]=J0min.data();
  jacobians_minimal[1]=J1min.data();
  jacobians_minimal[2]=J2min.data();
  jacobians_minimal[3]=J3min.data();
  Eigen::Matrix<double,15,1> residuals;
  // evaluate twice to be sure that we will be using the linearisation of
  // the biases (i.e. no preintegrals redone)
  static_cast<ceres_backend::ImuError*>(cost_function_imu)->EvaluateWithMinimalJacobians(
        parameters,residuals.data(),jacobians,jacobians_minimal);
  static_cast<ceres_backend::ImuError*>(cost_function_imu)->EvaluateWithMinimalJacobians(
        parameters,residuals.data(),jacobians,jacobians_minimal);

  // and now num-diff:
  double dx=1e-6;

  Eigen::Matrix<double,15,6> J0_numDiff;
  for(size_t i=0; i<6; ++i){
    Eigen::Matrix<double,6,1> dp_0;
    Eigen::Matrix<double,15,1> residuals_p;
    Eigen::Matrix<double,15,1> residuals_m;
    dp_0.setZero();
    dp_0[i]=dx;
    poseLocalParameterization->Plus(parameters[0],dp_0.data(),parameters[0]);
    //std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_p.data(),NULL);
    //std::cout<<residuals_p.transpose()<<std::endl;
    poseParameterBlock_0.setEstimate(T_WS_0); // reset
    dp_0[i]=-dx;
    //std::cout<<residuals.transpose()<<std::endl;
    poseLocalParameterization->Plus(parameters[0],dp_0.data(),parameters[0]);
    //std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_m.data(),NULL);
    //std::cout<<residuals_m.transpose()<<std::endl;
    poseParameterBlock_0.setEstimate(T_WS_0); // reset
    J0_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));
  }
  EXPECT_TRUE((J0min-J0_numDiff).norm()<jacobianTolerance)
      << "minimal Jacobian 0 = \n" << J0min
      << "\nnumDiff minimal Jacobian 0 = \n" << J0_numDiff;
  //std::cout << "minimal Jacobian 0 = \n"<<J0min<<std::endl;
  //std::cout << "numDiff minimal Jacobian 0 = \n"<<J0_numDiff<<std::endl;
  Eigen::Matrix<double,7,6,Eigen::RowMajor> Jplus;
  poseLocalParameterization->ComputeJacobian(parameters[0],Jplus.data());
  //std::cout << "Jacobian 0 times Plus Jacobian = \n"<<J0*Jplus<<std::endl;

  Eigen::Matrix<double,15,6> J2_numDiff;
  for(size_t i=0; i<6; ++i){
    Eigen::Matrix<double,6,1> dp_1;
    Eigen::Matrix<double,15,1> residuals_p;
    Eigen::Matrix<double,15,1> residuals_m;
    dp_1.setZero();
    dp_1[i]=dx;
    poseLocalParameterization->Plus(parameters[2],dp_1.data(),parameters[2]);
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_p.data(),NULL);
    poseParameterBlock_1.setEstimate(T_WS_1_disturbed); // reset
    dp_1[i]=-dx;
    poseLocalParameterization->Plus(parameters[2],dp_1.data(),parameters[2]);
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_m.data(),NULL);
    poseParameterBlock_1.setEstimate(T_WS_1_disturbed); // reset
    J2_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));
  }
  EXPECT_TRUE((J2min-J2_numDiff).norm()<jacobianTolerance)
      << "minimal Jacobian 2 = \n" << J2min << std::endl
      << "numDiff minimal Jacobian 2 = \n" << J2_numDiff;
  poseLocalParameterization->ComputeJacobian(parameters[2],Jplus.data());
  //std::cout << "Jacobian 2 times Plus Jacobian = \n"<<J2*Jplus<<std::endl;

  Eigen::Matrix<double,15,9> J1_numDiff;
  for(size_t i=0; i<9; ++i){
    Eigen::Matrix<double,9,1> ds_0;
    Eigen::Matrix<double,15,1> residuals_p;
    Eigen::Matrix<double,15,1> residuals_m;
    ds_0.setZero();
    ds_0[i]=dx;
    Eigen::Matrix<double,9,1> plussed=speedAndBias_0+ds_0;
    speedAndBiasParameterBlock_0.setEstimate(plussed);
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_p.data(),NULL);
    ds_0[i]=-dx;
    plussed=speedAndBias_0+ds_0;
    speedAndBiasParameterBlock_0.setEstimate(plussed);
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_m.data(),NULL);
    speedAndBiasParameterBlock_0.setEstimate(speedAndBias_0); // reset
    J1_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));
  }
  EXPECT_TRUE((J1min-J1_numDiff).norm()<jacobianTolerance)
      << "minimal Jacobian 1 = \n" << J1min << std::endl <<
         "numDiff minimal Jacobian 1 = \n" << J1_numDiff;
  //std::cout << "minimal Jacobian 1 = \n"<<J1min<<std::endl;
  //std::cout << "numDiff minimal Jacobian 1 = \n"<<J1_numDiff<<std::endl;

  Eigen::Matrix<double,15,9> J3_numDiff;
  for(size_t i=0; i<9; ++i){
    Eigen::Matrix<double,9,1> ds_1;
    Eigen::Matrix<double,15,1> residuals_p;
    Eigen::Matrix<double,15,1> residuals_m;
    ds_1.setZero();
    ds_1[i]=dx;
    Eigen::Matrix<double,9,1> plussed=speedAndBias_1+ds_1;
    speedAndBiasParameterBlock_1.setEstimate(plussed);
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_p.data(),NULL);
    ds_1[i]=-dx;
    plussed=speedAndBias_1+ds_1;
    speedAndBiasParameterBlock_1.setEstimate(plussed);
    static_cast<ceres_backend::ImuError*>(cost_function_imu)->Evaluate(
          parameters,residuals_m.data(),NULL);
    speedAndBiasParameterBlock_1.setEstimate(speedAndBias_0); // reset
    J3_numDiff.col(i)=(residuals_p-residuals_m)*(1.0/(2*dx));
  }
  EXPECT_TRUE((J3min-J3_numDiff).norm()<jacobianTolerance)
      << "minimal Jacobian 1 = \n" << J3min << std::endl
      << "numDiff minimal Jacobian 1 = \n" << J3_numDiff;
  //std::cout << "minimal Jacobian 3 = \n"<<J3min<<std::endl;
  //std::cout << "numDiff minimal Jacobian 3 = \n"<<J3_numDiff<<std::endl;

  // Run the solver!
  std::cout<<"run the solver... "<<std::endl;
  ceres::Solver::Options options;
  //options.check_gradients=true;
  //options.numeric_derivative_relative_step_size = 1e-6;
  //options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold=google::WARNING; // enable console warnings (Jacobian verification)
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // print some infos about the optimization
  //std::cout << summary.FullReport() << "\n";
  std::cout << "initial T_WS_1 : "
            << T_WS_1_disturbed.getTransformationMatrix() << "\n"
            << "optimized T_WS_1 : "
            << poseParameterBlock_1.estimate().getTransformationMatrix() << "\n"
            << "correct T_WS_1 : " << T_WS_1.getTransformationMatrix() << "\n";

  // make sure it converged
  EXPECT_TRUE(summary.final_cost<1e-2) << "cost not reducible";
  EXPECT_TRUE(
        2*(T_WS_1.getEigenQuaternion()
           *poseParameterBlock_1.estimate().getEigenQuaternion()
           .inverse()).vec().norm() < 1e-2)
      << "quaternions not close enough";
  EXPECT_TRUE((T_WS_1.getPosition()-poseParameterBlock_1.estimate()
               .getPosition()).norm()<0.04)
      << "translation not close enough";
}

VIKIT_UNITTEST_ENTRYPOINT
