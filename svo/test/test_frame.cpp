#include <cmath>

#include <aslam/common/entrypoint.h>
#include <aslam/common/numdiff-jacobian-tester.h>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <svo/common/camera_fwd.h>
#include <svo/common/frame.h>
#include <svo/test_utils/test_utils.h>
#include <vikit/cameras.h>
#include <vikit/cameras/camera_factory.h>
#include <vikit/cameras/camera_geometry_base.h>
#include <vikit/cameras/omni_geometry.h>
#include <vikit/timer.h>

namespace svo {

struct BearingVectorFunctor: public aslam::common::NumDiffFunctor<3, 6>
{
  BearingVectorFunctor(
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu)
    : T_cam_imu_(T_cam_imu)
    , p_in_imu_(p_in_imu)
  {}
  virtual ~BearingVectorFunctor() {}

  virtual bool functional(const typename NumDiffFunctor::InputType& x,
                          typename NumDiffFunctor::ValueType& fvec,
                          typename NumDiffFunctor::JacobianType* Jout) const
  {
    Transformation T_update = Transformation::exp(x);

    const Eigen::Vector3d p_in_imu = T_update * p_in_imu_;
    const Eigen::Vector3d p_in_cam = T_cam_imu_ * p_in_imu;
    if(Jout)
      Frame::jacobian_xyz2f_imu(T_cam_imu_, p_in_imu, *Jout);
    fvec = p_in_cam.normalized();
    return true;
  }

  Transformation T_cam_imu_;
  Eigen::Vector3d p_in_imu_;
};


struct UnitPlaneFunctor: public aslam::common::NumDiffFunctor<2, 6>
{
  UnitPlaneFunctor(
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu,
      const Eigen::Vector3d& f_measured)
    : T_cam_imu_(T_cam_imu)
    , p_in_imu_(p_in_imu)
    , f_measured_(f_measured)
  {}
  virtual ~UnitPlaneFunctor() {}

  virtual bool functional(const typename NumDiffFunctor::InputType& x,
                          typename NumDiffFunctor::ValueType& fvec,
                          typename NumDiffFunctor::JacobianType* Jout) const
  {
    Transformation T_update = Transformation::exp(x);

    const Eigen::Vector3d p_in_imu = T_update * p_in_imu_;
    const Eigen::Vector3d p_in_cam = T_cam_imu_ * p_in_imu;
    if(Jout)
      Frame::jacobian_xyz2uv_imu(T_cam_imu_, p_in_imu, *Jout);
    fvec = vk::project2(f_measured_) - vk::project2(p_in_cam);
    return true;
  }

  Transformation T_cam_imu_;
  Eigen::Vector3d p_in_imu_;
  Eigen::Vector3d f_measured_;
};

struct ImagePlaneFunctor: public aslam::common::NumDiffFunctor<2, 6>
{
  ImagePlaneFunctor(
      std::shared_ptr<const svo::Camera> cam,
      const Transformation& T_cam_imu,
      const Eigen::Vector3d& p_in_imu)
    : cam_(cam)
    , T_cam_imu_(T_cam_imu)
    , p_in_imu_(p_in_imu)
  {}
  virtual ~ImagePlaneFunctor() {}
  virtual bool functional(const typename NumDiffFunctor::InputType& x,
                          typename NumDiffFunctor::ValueType& fvec,
                          typename NumDiffFunctor::JacobianType* Jout) const
  {
    Transformation T_update = Transformation::exp(x);
    const Eigen::Vector3d p_in_imu = T_update * p_in_imu_;
    if(Jout)
      Frame::jacobian_xyz2image_imu(*cam_, T_cam_imu_, p_in_imu, *Jout);
    const Eigen::Vector3d p_in_cam = T_cam_imu_ * p_in_imu;
    cam_->project3(p_in_cam, &fvec);
    return true;
  }

  std::shared_ptr<const svo::Camera> cam_;
  Transformation T_cam_imu_;
  Eigen::Vector3d p_in_imu_;
};

class SVOFrameTest: public ::testing::Test
{
protected:
  SVOFrameTest()
  {  std::srand(std::time(NULL)); }
};

TEST_F(SVOFrameTest, testBearingVectorJacobian)
{
  Eigen::Matrix<double, 6, 1> input;
  input.block<3, 1>(0, 0) = Eigen::Vector3d(0, 0, 0);
  input.block<3, 1>(3, 0) = Eigen::Vector3d(0, 0, 0);
  for(int i=0; i<10; i++)
  {
    Transformation T_cam_imu;
    T_cam_imu.setRandom(1, 0.5);
    Eigen::Vector3d p_in_imu = test_utils::generateRandomPoint(1, 10);
    // make sure z_cam is not zero
    TEST_JACOBIAN_FINITE_DIFFERENCE(
          BearingVectorFunctor, input, 1e-6, 1e-5,
          T_cam_imu, p_in_imu);
  }
}

TEST_F(SVOFrameTest, testUnitPlaneJacobian)
{
  Eigen::Matrix<double, 6, 1> input;
  input.block<3, 1>(0, 0) = Eigen::Vector3d(0, 0, 0);
  input.block<3, 1>(3, 0) = Eigen::Vector3d(0, 0, 0);
  for(int i=0; i<10; i++)
  {
    Transformation T_cam_imu;
    T_cam_imu.setRandom(1, 0.5);
    Eigen::Vector3d p_in_imu, f_in_imu;
    // project2d requires nonzero z coordinate
    while(true)
    {
      p_in_imu = test_utils::generateRandomPoint(1, 10);
      f_in_imu = test_utils::generateRandomPoint(1, 10);
      const Eigen::Vector3d p_in_cam = T_cam_imu * p_in_imu;
      const Eigen::Vector3d f_in_cam = T_cam_imu * f_in_imu;
      if(p_in_cam(2) > 1e-3 && f_in_cam(2) > 1e-3)
        break;
    }
    f_in_imu.normalize();
    TEST_JACOBIAN_FINITE_DIFFERENCE(
          UnitPlaneFunctor, input, 1e-7, 1e-3,
          T_cam_imu, p_in_imu, f_in_imu);
  }
}

// TODO(zzc): make typed test
TEST_F(SVOFrameTest, testOmniImageJacobian)
{
  Eigen::Matrix<double, 6, 1> input;
  input.block<3, 1>(0, 0) = Eigen::Vector3d(0, 0, 0);
  input.block<3, 1>(3, 0) = Eigen::Vector3d(0, 0, 0);
  // omni camera
  Eigen::Matrix<double, 5, 1> polynomial;
  polynomial << -69.6915, 0.0, 5.4772e-4, 2.1371e-5, -8.7523e-9;
  const Eigen::Vector2d principal_point(320.0, 240.0);
  const Eigen::Vector3d distortion(1.0,  0.0, 0.0);
  Eigen::Matrix<double, 12, 1> inverse_polynomial;
  inverse_polynomial << 142.7468, 104.8486, 7.3973, 17.4581, 12.6308, -4.3751,
      6.9093,  10.9703, -0.6053, -3.9119, -1.0675, 0.0;
  const Eigen::Vector2d mask_relative_radii(0.0, 1.0);

  svo::CameraPtr omni_cam = svo::CameraPtr(
      new vk::cameras::OmniGeometry(
          640, 480, polynomial, principal_point, distortion, inverse_polynomial,
          mask_relative_radii));

  for(int i=0; i<10; i++)
  {
    Transformation T_cam_imu;
    T_cam_imu.setRandom(1, 0.5);
    Eigen::Vector3d p_in_imu;
    while(true)
    {
      p_in_imu = test_utils::generateRandomPoint(1, 10);
      Eigen::Vector2d kp;
      if(omni_cam->project3(T_cam_imu*p_in_imu, &kp).isKeypointVisible())
        break;
    }
    // on image plane, should be 2-order higher than unit plane error
    TEST_JACOBIAN_FINITE_DIFFERENCE(
          ImagePlaneFunctor, input, 1e-4, 5*1e-2,
          omni_cam, T_cam_imu, p_in_imu);
  }
}

TEST_F(SVOFrameTest, testPinholeImageJacobian)
{
  Eigen::Matrix<double, 6, 1> input;
  input.block<3, 1>(0, 0) = Eigen::Vector3d(0, 0, 0);
  input.block<3, 1>(3, 0) = Eigen::Vector3d(0, 0, 0);
  // pinhole camera
  double f = 315.5;
  Eigen::VectorXd pinhole_intrin(4);
  pinhole_intrin << f, f, 376.0, 240.0;
  svo::CameraPtr pinhole_cam =
      vk::cameras::factory::makePinholeCamera(pinhole_intrin, 752, 480);

  for(int i=0; i<10; i++)
  {
    Transformation T_cam_imu;
    T_cam_imu.setRandom(1, 0.5);
    Eigen::Vector3d p_in_imu;
    while(true)
    {
      p_in_imu = test_utils::generateRandomPoint(1, 10);
      Eigen::Vector2d kp;
      if(pinhole_cam->project3(T_cam_imu*p_in_imu, &kp).isKeypointVisible())
        break;
    }
    // on image plane, should be 2-order higher than unit plane error
    TEST_JACOBIAN_FINITE_DIFFERENCE(
          ImagePlaneFunctor, input, 1e-4, 5*1e-2,
          pinhole_cam, T_cam_imu, p_in_imu);
  }

  // compare time
  vk::Timer timer;
  Transformation T_cam_imu;
  T_cam_imu.setRandom(1, 0.5);
  Eigen::Vector3d p_in_imu;
  while(true)
  {
    p_in_imu = test_utils::generateRandomPoint(1, 10);
    Eigen::Vector2d kp;
    if(pinhole_cam->project3(T_cam_imu*p_in_imu, &kp).isKeypointVisible())
      break;
  }
  Eigen::Matrix<double, 2, 6> frame_jac;
  timer.start();
  for(int i=0; i<100; i++)
  {
    Frame::jacobian_xyz2uv_imu(T_cam_imu, p_in_imu, frame_jac);
    frame_jac *= f;
  }
  timer.stop();
  std::cout << "Analytical inline function(ms):" << timer.getMilliseconds() << std::endl;
  timer.start();
  for(int i=0; i<100; i++)
    Frame::jacobian_xyz2image_imu(*pinhole_cam, T_cam_imu, p_in_imu, frame_jac);
  timer.stop();
  std::cout << "Using vikit library(ms):" << timer.getMilliseconds() << std::endl;

}

}  // namespace svo

VIKIT_UNITTEST_ENTRYPOINT
