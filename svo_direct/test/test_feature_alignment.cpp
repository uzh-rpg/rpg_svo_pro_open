#include <gflags/gflags.h>
#include <ros/package.h>

// svo
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/direct/feature_alignment.h>
#include <svo/direct/patch_warp.h>
#include <svo/direct/patch_utils.h>

// others
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/imgproc/imgproc.hpp>

namespace {

using namespace svo;

void testAlignment1D_sameImg()
{
  return; // skip

  constexpr int kHalfPatchSize = 4;
  constexpr int kPatchSize = 2 * kHalfPatchSize;

  // Load dataset.
  std::string dataset_dir = ros::package::getPath("rpg_datasets")+"/rpg_urban_pinhole";
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  // Load detector.
  svo::DetectorOptions options;
  options.detector_type = svo::DetectorType::kGridGrad;
  options.cell_size = 40;
  svo::AbstractDetectorPtr detector =
      svo::feature_detection_utils::makeDetector(options, dataset.cam());

  // Extract features on first dataset
  svo::FramePtr ref_frame, cur_frame;
  cv::Mat ref_depthmap;
  dataset.getNextFrame(5u, ref_frame, &ref_depthmap);
  detector->detect(ref_frame);
  VLOG(1) << "Detected " << ref_frame->numFeatures() << " features.";

  // For each feature, test 1D alignment.
  cur_frame = ref_frame;
  uint8_t ref_patch_with_border[(kPatchSize+2)*(kPatchSize+2)] __attribute__ ((aligned (16)));
  uint8_t ref_patch[kPatchSize*kPatchSize] __attribute__ ((aligned (16)));
  for(int i = 0; i < ref_frame->px_vec_.cols(); ++i)
  {
    Eigen::Ref<Keypoint> kp_ref = ref_frame->px_vec_.col(i);
    warp::createPatchNoWarpInterpolated(
          ref_frame->img_pyr_[0], kp_ref, kHalfPatchSize+1, ref_patch_with_border);
    patch_utils::createPatchFromPatchWithBorder(ref_patch_with_border, kPatchSize, ref_patch);
    Keypoint kp_cur = kp_ref + 0.5 * ref_frame->grad_vec_.col(i);
    double init_error = (kp_cur - kp_ref).norm();
    bool res = feature_alignment::align1D(
        cur_frame->img_pyr_[0],
        ref_frame->grad_vec_.col(i),
        ref_patch_with_border, ref_patch,
        30, true, false,
        &kp_cur);

    Keypoint kp_error = kp_cur - kp_ref;
    VLOG(1) << "init_error = " << init_error
            << ", final error = " << kp_error.norm()
            << ((res) ? " SUCCESS" : " FAILURE");
  }
}

void testAlignment1D_differentImg()
{
  constexpr int kHalfPatchSize = 4;
  constexpr int kPatchSize = 2 * kHalfPatchSize;

  // Load dataset.
  std::string dataset_dir = ros::package::getPath("rpg_datasets")+"/rpg_urban_pinhole";
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  // Load detector.
  svo::DetectorOptions options;
  options.detector_type = svo::DetectorType::kGridGrad;
  options.cell_size = 40;
  svo::AbstractDetectorPtr detector =
      svo::feature_detection_utils::makeDetector(options, dataset.cam());

  // Extract features on first dataset
  svo::FramePtr ref_frame, cur_frame;
  cv::Mat ref_depthmap;
  dataset.getNextFrame(5u, ref_frame, &ref_depthmap);
  detector->detect(ref_frame);
  VLOG(1) << "Detected " << ref_frame->numFeatures() << " features.";

  // Get next frame.
  for(size_t i = 0; i < 10; ++i)
    dataset.getNextFrame(5u, cur_frame, nullptr);
  const Transformation T_cur_ref = cur_frame->T_cam_world() * ref_frame->T_world_cam();

  // For each feature, test 1D alignment.
  Matcher matcher;
  CHECK_EQ(static_cast<size_t>(ref_frame->px_vec_.cols()), ref_frame->numFeatures());
  for(int i = 0; i < ref_frame->px_vec_.cols(); ++i)
  {
    double ref_depth = ref_depthmap.at<float>(ref_frame->px_vec_(1,i),
                                              ref_frame->px_vec_(0,i));
    const Eigen::Vector3d xyz_cur = T_cur_ref * (ref_frame->f_vec_.col(i) * ref_depth);
    Keypoint px_cur;
    cur_frame->cam()->project3(xyz_cur, &px_cur);
    if(!cur_frame->cam()->isKeypointVisibleWithMargin(px_cur, 10))
      continue;

    Keypoint px_cur_distorted = px_cur + Keypoint(0.5, 0.5);
    FeatureWrapper ref_ftr = ref_frame->getFeatureWrapper(i);
    matcher.findMatchDirect(*ref_frame, *cur_frame, ref_ftr, ref_depth, px_cur_distorted);

    // Compute gradient at true location.
    double angle;
    feature_detection_utils::getCornerAngle(cur_frame->img_pyr_, px_cur, 0u, &angle);
    GradientVector cur_grad(std::cos(angle), std::sin(angle));

    double alignment_error = cur_grad.dot(px_cur_distorted - px_cur);
    VLOG(1) << "final error = " << alignment_error;
  }
}

} // namespace

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  VLOG(1) << "TEST 1 ---------------------------------------------------------";
  testAlignment1D_differentImg();
  VLOG(1) << "TEST 2 ---------------------------------------------------------";
  testAlignment1D_sameImg();
  VLOG(1) << "FINISHED -------------------------------------------------------";

  return 0;
}
