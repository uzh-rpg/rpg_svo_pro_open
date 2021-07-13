#include "vikit/cameras.h"
#include "vikit/cameras/atan_distortion.h"
#include "vikit/cameras/camera_factory.h"
#include "vikit/cameras/equidistant_distortion.h"
#include "vikit/cameras/ncamera.h"
#include "vikit/cameras/radial_tangential_distortion.h"
#include <vikit/cameras/equidistant_fisheye_geometry.h>
#include "vikit/cameras/omni_geometry.h"

#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/numdiff-jacobian-tester.h>
#include <vikit/path_utils.h>

class CamerasTest: public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    data_dir_ = vk::path_utils::getBaseName(__FILE__) + "/data";
    omni_cam_ = vk::cameras::CameraGeometryBase::loadFromYaml(data_dir_+"/calib_omni.yaml");
    CHECK(!data_dir_.empty()) << "Fail to extract data directory.";
    CHECK(omni_cam_) << "Fail to load omni cameras";
  }

  // virtual void TearDown();

  std::string data_dir_;
  vk::cameras::CameraGeometryBase::Ptr omni_cam_;

};

TEST_F(CamerasTest, AtanDistortion)
{
  double x = 0.5, y = 0.8;
  vk::cameras::AtanDistortion distortion_model(0.934479);
  distortion_model.distort(x, y);
  distortion_model.undistort(x, y);
  EXPECT_NEAR(x, 0.5, 1e-10);
  EXPECT_NEAR(y, 0.8, 1e-10);
}

TEST_F(CamerasTest, EquidistantDistortion)
{
  double x = 0.5, y = 0.8;
  vk::cameras::EquidistantDistortion distortion_model(
        -0.0027, 0.0241, -0.0430, 0.0311);
  distortion_model.distort(x, y);
  std::cout << "x = " << x << ", y = " << y << std::endl;
  distortion_model.undistort(x, y);
  EXPECT_NEAR(x, 0.5, 1e-10);
  EXPECT_NEAR(y, 0.8, 1e-10);

  struct EqJacobianFunctor: public aslam::common::NumDiffFunctor<2, 2>
  {
    EqJacobianFunctor(const vk::cameras::EquidistantDistortion& dist)
      : dist_(dist)
    {};
    virtual ~EqJacobianFunctor() {};
    bool functional(
          const typename NumDiffFunctor::InputType& x,
          typename NumDiffFunctor::ValueType& fvec,
          typename NumDiffFunctor::JacobianType* Jout) const
    {
        fvec = dist_.distort(x);
        if (Jout)
        {
          (*Jout) = dist_.jacobian(x);
        }
        return true;
    };

    const vk::cameras::EquidistantDistortion& dist_;
  };
  constexpr double kStepSize = 1e-4;
  constexpr double kEpsJacobian = 1e-4;
  TEST_JACOBIAN_FINITE_DIFFERENCE(
        EqJacobianFunctor, Eigen::Vector2d(x, y),
        kStepSize, kEpsJacobian, distortion_model);
}

TEST_F(CamerasTest, RadialTangentialDistortion)
{
  double x = 0.5, y = 0.8;
  vk::cameras::RadialTangentialDistortion distortion_model(
        -0.3, 0.1, 9.52e-05, -0.00057);
  distortion_model.distort(x, y);
  distortion_model.undistort(x, y);
  // NOTE: with radtan we don't achieve same precision with only five iterations
  // at this pose (0.5, 0.8) far from image center
  EXPECT_NEAR(x, 0.5, 1e-2);
  EXPECT_NEAR(y, 0.8, 1e-2);

  struct RadialTangentialJacobianFunctor: public aslam::common::NumDiffFunctor<2, 2>
  {
    RadialTangentialJacobianFunctor(const vk::cameras::RadialTangentialDistortion& dist)
      : dist_(dist)
    {};
    virtual ~RadialTangentialJacobianFunctor() {};
    bool functional(
          const typename NumDiffFunctor::InputType& x,
          typename NumDiffFunctor::ValueType& fvec,
          typename NumDiffFunctor::JacobianType* Jout) const
    {
        fvec = dist_.distort(x);
        if (Jout)
        {
          (*Jout) = dist_.jacobian(x);
        }
        return true;
    };

    const vk::cameras::RadialTangentialDistortion& dist_;
  };
  constexpr double kStepSize = 1e-4;
  constexpr double kEpsJacobian = 1e-4;
  TEST_JACOBIAN_FINITE_DIFFERENCE(
        RadialTangentialJacobianFunctor, Eigen::Vector2d(x, y),
        kStepSize, kEpsJacobian, distortion_model);
}

TEST_F(CamerasTest, CameraFactoryNoDistortion)
{
  vk::cameras::CameraGeometryBase::Ptr cam1 =
      vk::cameras::factory::loadFromYAML(
        data_dir_+"/calib_pinhole_nodistortion.yaml", "cam0");
  cam1->printParameters(std::cout, "Test Load Camera No Distortion:");
}

TEST_F(CamerasTest, CameraFactoryAtan)
{
  vk::cameras::CameraGeometryBase::Ptr cam1 =
      vk::cameras::factory::loadFromYAML(
        data_dir_+"/calib_pinhole_atan.yaml", "cam0");
  cam1->printParameters(std::cout, "Test Load Camera Atan:");
}

TEST_F(CamerasTest, CameraFactoryEquidistant)
{
  vk::cameras::CameraGeometryBase::Ptr cam1 =
      vk::cameras::factory::loadFromYAML(
        data_dir_+"/calib_pinhole_equidistant.yaml", "cam0");
  cam1->printParameters(std::cout, "Test Load Camera Equidistant:");
}

TEST_F(CamerasTest, CameraFactoryRadTan)
{
  vk::cameras::CameraGeometryBase::Ptr cam1 =
      vk::cameras::factory::loadFromYAML(
        data_dir_+"/calib_pinhole_radtan.yaml", "cam0");
  cam1->printParameters(std::cout, "Test Load Camera Radial Tangential:");
}

TEST_F(CamerasTest, CameraYaml)
{
  vk::cameras::CameraGeometryBase::Ptr cam =
      vk::cameras::CameraGeometryBase::loadFromYaml(data_dir_ + "/calib_cam.yaml");
  CHECK_NOTNULL(cam.get());
}

TEST_F(CamerasTest, CameraProjection)
{
  vk::cameras::CameraGeometryBase::Ptr cam =
      vk::cameras::CameraGeometryBase::loadFromYaml(data_dir_ + "/calib_cam.yaml");

  Eigen::Vector3d xyz(0.1, 0.2, 2.0), xyz_res;
  Eigen::Vector2d px;
  cam->project3(xyz, &px);
  cam->backProject3(px, &xyz_res);
  EXPECT_NEAR((xyz/xyz[2]-xyz_res).norm(), 0.0, 0.00000001);
  CHECK_NOTNULL(cam.get());
}

TEST_F(CamerasTest, NCameraYamlStereo)
{
  vk::cameras::NCamera::Ptr ncam =
      vk::cameras::NCamera::loadFromYaml(data_dir_ + "/calib_kitti.yaml");
  CHECK_NOTNULL(ncam.get());
}

TEST(CameraTest, EquidistantProjection)
{
  const double f = 350;
  vk::cameras::Camera::Ptr cam =
      std::make_shared<vk::cameras::EquidistantFisheyeGeometry>(640, 480, f,
                                                                Eigen::Vector2d(320, 240),
                                                                320);
  // Forward
  Eigen::Vector2d projected;
  cam->project3(Eigen::Vector3d(10.0, 20.0, 15.0), &projected);
  EXPECT_NEAR(projected[0],473.38230111, 1e-7);
  EXPECT_NEAR(projected[1], 546.76460222, 1e-7);

  cam->project3(Eigen::Vector3d(0.0, 0.0, 15.0), &projected);
  EXPECT_NEAR(projected[0], 320.0, 1e-7);
  EXPECT_NEAR(projected[1], 240.0, 1e-7);

  EXPECT_DEATH(cam->project3(Eigen::Vector3d(0.0, 0.0, 0.0), &projected), "");

  // Backward
  Eigen::Vector3d ray;
  cam->backProject3(Eigen::Vector2d(20, 190), &ray);
  EXPECT_NEAR(ray[0], -0.7532713858288931, 1e-7);
  EXPECT_NEAR(ray[1], -0.12554523097148218, 1e-7);
  EXPECT_NEAR(ray[2], 0.6456164606573597, 1e-7);

  cam->backProject3(Eigen::Vector2d(320, 240), &ray);
  EXPECT_NEAR(ray[0], 0, 1e-7);
  EXPECT_NEAR(ray[1], 0, 1e-7);
  EXPECT_NEAR(ray[2], 1, 1e-7);
}

TEST_F(CamerasTest, OmniCameraInitialization)
{
  vk::cameras::CameraGeometryBase::Ptr omni_cam =
      vk::cameras::CameraGeometryBase::loadFromYaml(data_dir_ + "/calib_omni.yaml");
  CHECK_NOTNULL(omni_cam.get());
  omni_cam->printParameters(std::cout, "Test omni cameras:");
}

TEST_F(CamerasTest, OmniCameraBackprojection)
{
  constexpr double kEpsBackprojection = 1e-5;
  const Eigen::Vector2d keypoint(400.0, 300.0);
  const Eigen::Vector3d expected_bearing(
        0.733011271294813, 0.549758453471110, 0.400574735838165);

  Eigen::Vector3d bearing;
  omni_cam_->backProject3(keypoint, &bearing);
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(expected_bearing, bearing, kEpsBackprojection));
}

TEST_F(CamerasTest, OmniCameraProjection)
{
  constexpr double kEpsProjection = 1e-4;
  const Eigen::Vector3d landmark(1.0, 1.0, -1.0);
  const Eigen::Vector2d expected_keypoint(
        4.729118411664447e+02, 3.929118411664447e+02);

  Eigen::Vector2d keypoint;
  omni_cam_->project3(landmark, &keypoint);
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(expected_keypoint, keypoint, kEpsProjection));
}

TEST_F(CamerasTest, OmniCameraJacobian)
{
  struct OmniJacobianFunctor: public aslam::common::NumDiffFunctor<2, 3>
  {
    OmniJacobianFunctor(vk::cameras::OmniGeometry::Ptr cam)
      : cam_(cam)
    {};
    virtual ~OmniJacobianFunctor() {};
    bool functional(
          const typename NumDiffFunctor::InputType& x,
          typename NumDiffFunctor::ValueType& fvec,
          typename NumDiffFunctor::JacobianType* Jout) const
    {
        cam_->project3(x, &fvec, Jout);
        return true;
    };

    vk::cameras::OmniGeometry::Ptr cam_;
  };

  constexpr double kStepSize = 1e-4;
  constexpr double kEpsJacobian = 1e-4;
  const Eigen::Vector3d landmark(0.5, 0.2, -1.0);
  TEST_JACOBIAN_FINITE_DIFFERENCE(
        OmniJacobianFunctor, landmark, kStepSize, kEpsJacobian, omni_cam_);
}

VIKIT_UNITTEST_ENTRYPOINT
