#include <gflags/gflags.h>
#include <ros/package.h>

// svo
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>

// others
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/imgproc/imgproc.hpp>

namespace {

using namespace svo;

void test()
{
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
  std::cout << "Detected " << ref_frame->numFeatures() << " features." << std::endl;

  // Run Dataset.
  Matcher matcher;
  size_t i = 0;
  while(dataset.getNextFrame(5u, cur_frame, nullptr))
  {
    cur_frame->resizeFeatureStorage(ref_frame->numFeatures());
    Transformation T_cur_ref = cur_frame->T_cam_world() * ref_frame->T_world_cam();
    size_t num_matches = 0;
    for(size_t i = 0; i < ref_frame->numFeatures(); ++i)
    {
      double ref_depth = ref_depthmap.at<float>(ref_frame->px_vec_(1,i),
                                                ref_frame->px_vec_(0,i));
      const Eigen::Vector3d xyz_cur = T_cur_ref * (ref_frame->f_vec_.col(i) * ref_depth);
      Keypoint px;
      if(!cur_frame->cam()->project3(xyz_cur, &px))
        continue;

      // Apply matching
      FeatureWrapper ref_ftr = ref_frame->getFeatureWrapper(i);
      if(matcher.findMatchDirect(*ref_frame, *cur_frame, ref_ftr, ref_depth, px)
         == Matcher::MatchResult::kSuccess)
      {
        cur_frame->px_vec_.col(num_matches) = px;
        cur_frame->f_vec_.col(num_matches) = matcher.f_cur_;
        cur_frame->type_vec_[num_matches] = ref_ftr.type;
        cur_frame->grad_vec_.col(num_matches) = matcher.A_cur_ref_ * ref_ftr.grad;
        ++num_matches;
      }
    }
    cur_frame->num_features_ = num_matches;
    std::cout << "Successfully projected " << num_matches << " features." << std::endl;

    cv::Mat img_rgb;
    feature_detection_utils::drawFeatures(*cur_frame, 0u, false, &img_rgb);
    cv::imshow("img_rgb", img_rgb);
    cv::waitKey(0);
    ++i;
    if(i > 50)
      break;
  }
}

} // namespace svo

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  test();

  return 0;
}
