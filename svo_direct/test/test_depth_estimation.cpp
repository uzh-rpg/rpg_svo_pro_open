#include <aslam/common/entrypoint.h>
#include <aslam/common/memory.h>
#include <aslam/common/numdiff-jacobian-tester.h>
#include <eigen-checks/gtest.h>
#include <ros/package.h>
#include <svo/common/camera.h>
#include <svo/common/frame.h>
#include <svo/common/seed.h>
#include <svo/common/transformation.h>
#include <svo/common/types.h>
#include <svo/test_utils/synthetic_dataset.h>
#include <vikit/cameras.h>
#include <vikit/cameras/camera_geometry.h>
#include <vikit/cameras/no_distortion.h>
#include <vikit/cameras/pinhole_projection.h>

#include "svo/direct/depth_estimation.h"

/// Wrapper that brings the distortion function to the form needed by the differentiator.
struct DepthJacobianFunctor : public aslam::common::NumDiffFunctor<2, 1>
{
  DepthJacobianFunctor(
      const svo::Camera::ConstPtr& camera,
      const Eigen::Vector3d& f_ref,
      const svo::Transformation& T_cur_ref)
    : camera_(camera)
    , f_ref_(f_ref)
    , T_cur_ref_(T_cur_ref)
  { ; }

  virtual ~DepthJacobianFunctor() {}

  virtual bool functional(
      const typename aslam::common::NumDiffFunctor<1, 1>::InputType& inv_depth,
      typename aslam::common::NumDiffFunctor<2, 1>::ValueType& px_cur,
      typename aslam::common::NumDiffFunctor<2, 1>::JacobianType* Jout) const
  {
    CHECK(camera_);
    svo::BearingVector f_cur = T_cur_ref_ * (f_ref_ * (1.0 / inv_depth(0.0)));

    Eigen::Matrix<double, 2, 3> projection_jacobian;
    auto res = camera_->project3(f_cur, &px_cur, &projection_jacobian);
    if(Jout)
    {
      std::cout << "inv_depth = " << inv_depth(0,0) << std::endl;
      std::cout << "f_cur = " << f_cur.transpose() << std::endl;
      std::cout << "px = " << px_cur.transpose() << std::endl;
      std::cout << "Jacobian = " << std::endl;
      std::cout << projection_jacobian << std::endl;
      *Jout = projection_jacobian * T_cur_ref_.getRotation().rotate( f_ref_ * (-1.0 / std::pow(inv_depth(0,0), 2)) );
    }
    return static_cast<bool>(res);
  }

  svo::Camera::ConstPtr camera_;
  Eigen::Vector3d f_ref_;
  svo::Transformation T_cur_ref_;
};

TEST(TestDepthEstimation, JacobianWrtInverseDepth)
{
  svo::CameraPtr camera = vk::cameras::PinholeGeometry::createTestCamera();
  svo::BearingVector f_ref(0, 0, 1);
  svo::Transformation T_cur_ref;
  T_cur_ref.setRandom(0.5, 0.1);
  Eigen::Matrix<double, 1, 1> inv_depth;
  inv_depth.setConstant(1.0/2.3);
  TEST_JACOBIAN_FINITE_DIFFERENCE(
        DepthJacobianFunctor, inv_depth, 1e-8, 1e-6, camera, f_ref, T_cur_ref);
}

TEST(TestDepthEstimation, testBearing)
{
  svo::Transformation T_cur_ref;
  T_cur_ref.setRandom(0.5, 0.1);
  svo::BearingVector f_ref(0, 0, 1);
  double inv_depth = 1.0/2.7;
  svo::BearingVector f_cur_1 =
      T_cur_ref.getRotation().rotate(f_ref) + T_cur_ref.getPosition()*inv_depth;
  svo::BearingVector f_cur_2 = T_cur_ref * (f_ref * (1.0 / inv_depth));
  std::cout << f_cur_1.transpose() << std::endl;
  std::cout << f_cur_2.transpose() << std::endl;
  EIGEN_MATRIX_EQUAL(f_cur_1, f_cur_2); // TODO(cfo) Check doesn't work!??
}

TEST(TestDepthEstimation, DISABLED_testDataset)
{
  std::string dataset_dir = ros::package::getPath("rpg_datasets")+"/rpg_vfr_pinhole";
  CHECK(!dataset_dir.empty());
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  // Load images.
  svo::FramePtr ref_frame, cur_frame;
  cv::Mat ref_depthmap;
  CHECK(dataset.getNextFrame(5u, ref_frame, &ref_depthmap));
  dataset.skipNImages(4);
  CHECK(dataset.getNextFrame(5u, cur_frame, nullptr));

  // Set feature.
  ref_frame->resizeFeatureStorage(1);
  svo::FeatureWrapper ref_ftr = ref_frame->getEmptyFeatureWrapper();
  ref_ftr.level = 0u;
  ref_ftr.px = svo::Keypoint(219, 251);
  Eigen::Vector3d f;
  ref_frame->cam()->backProject3(ref_ftr.px, &f);
  ref_ftr.f = f.normalized();

  // Depth estimation.
  float ground_truth_depth =
      ref_depthmap.at<float>(ref_ftr.px(1), ref_ftr.px(0));
  float depth_init = ground_truth_depth*0.7;
  ref_frame->invmu_sigma2_a_b_vec_(0,0) = svo::seed::getMeanFromDepth(depth_init);
  VLOG(200) << "ground-truth depth = " << 1.0/ground_truth_depth;
  svo::DepthEstimator depth_estimator(svo::DepthEstimator::getDefaultSolverOptions());
  depth_estimator.run(cur_frame, ref_frame, 0);
  VLOG(200) << "estimated depth = " << ref_frame->invmu_sigma2_a_b_vec_(0,0);

  CHECK_NEAR(svo::seed::getMeanFromDepth(ground_truth_depth),
             ref_frame->invmu_sigma2_a_b_vec_(0,0), 0.001);
}

VIKIT_UNITTEST_ENTRYPOINT
