#include <gflags/gflags.h>
#include <ros/package.h>

// svo
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/direct/patch_utils.h>
#include <svo/direct/patch_warp.h>
#include <svo/direct/elder_zucker.h>

// others
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/imgproc/imgproc.hpp>

namespace {

using namespace svo;

void detectNonlinearDiffusionEdges(const ImgPyr& img_pyr)
{
  int pyr_level = 0;
  float dt = 0.2;
  float T  = 10.0;
  int num_iter = T / dt;
  int canny_low_threshold = 20;
  int canny_ratio = 3;


  cv::Mat src_gray = img_pyr[0];
  cv::Mat dest1, dest2, dest3;
  int ratio = 3;


  // ---------------------------------------------------------------------------
  // Gaussian Blurring (Linear Diffusion)

  // Reduce noise with a kernel 3x3
  cv::GaussianBlur(src_gray, dest1, cv::Size(3, 3), 0);
  cv::GaussianBlur(src_gray, dest2, cv::Size(9, 9), 0);
  cv::GaussianBlur(src_gray, dest3, cv::Size(17, 17), 0);

  // Canny detector
  cv::Canny(dest1, dest1, canny_low_threshold, canny_low_threshold*ratio);
  cv::Canny(dest2, dest2, canny_low_threshold, canny_low_threshold*ratio);
  cv::Canny(dest3, dest3, canny_low_threshold, canny_low_threshold*ratio);

  cv::imshow("guillermo edges1", dest1);
  cv::imshow("guillermo edges2", dest2);
  cv::imshow("guillermo edges3", dest3);

  // ---------------------------------------------------------------------------
  // Nonlinear Diffusion

  cv::Mat img_32f;
  img_pyr.at(pyr_level).convertTo(img_32f, CV_32FC1, 1.0f/255.0);
  cv::Mat kernel_deriv_x  = (cv::Mat_<float>(1,3) << -0.5, 0.0, 0.5);
  cv::Mat kernel_deriv_xx = (cv::Mat_<float>(1,3) << 1.0, -2.0, 1.0);
  cv::Mat kernel_deriv_xy = (cv::Mat_<float>(3,3) << 0.25, 0.00,-0.25,
                                                     0.00, 0.00, 0.00,
                                                    -0.25, 0.00, 0.25);
  // Apply filter
  cv::Mat Lt = img_32f;
  cv::Point anchor = cv::Point(-1,-1);
  cv::Mat Lx, Ly;
  cv::Mat Lxx, Lxy, Lyy;
  cv::Mat edges;
  for(int iter = 0; iter < num_iter; ++iter)
  {
    // Compute image derivatives
    cv::filter2D(Lt, Lx, CV_32FC1, kernel_deriv_x, anchor, 0, cv::BORDER_REPLICATE);
    cv::filter2D(Lt, Ly, CV_32FC1, kernel_deriv_x.t(), anchor, 0, cv::BORDER_REPLICATE);
    cv::filter2D(Lt, Lxx, CV_32FC1, kernel_deriv_xx, anchor, 0, cv::BORDER_REPLICATE);
    cv::filter2D(Lt, Lyy, CV_32FC1, kernel_deriv_xx.t(), anchor, 0, cv::BORDER_REPLICATE);
    cv::filter2D(Lt, Lxy, CV_32FC1, kernel_deriv_xy, anchor, 0, cv::BORDER_REPLICATE);

    // Update equation
    // Explicit scheme for Geometric Heat Equation (GHE)
    cv::Mat Lx_Squared = Lx.mul(Lx);
    cv::Mat Ly_Squared = Ly.mul(Ly);

    cv::Mat numerator = Ly_Squared.mul(Lxx) - 2 * Lx.mul(Ly.mul(Lxy)) + Lx_Squared.mul(Lyy);
    cv::Mat denom = 1e-6 + Lx_Squared + Ly_Squared;

    Lt = Lt + dt * numerator / denom;

    // Edge detection.
    cv::Mat Lt_u8;
    Lt.convertTo(Lt_u8, CV_8UC1, 255);
    cv::Canny(Lt_u8 , edges, canny_low_threshold, canny_low_threshold*canny_ratio);

    std::cout << "iter = " << iter << std::endl;
    cv::imshow("Lt", Lt);
    cv::imshow("Edges", edges);
    cv::waitKey(0);
  }



}

/*
void detectGuillermoEdges(const ImgPyr& img_pyr)
{
  cv::Mat src_gray = img_pyr[0];
  cv::Mat dest1, dest2, dest3;
  int low_threshold = 20;
  int smoothing_kernel_size = 21;
  int ratio = 3;

  // Reduce noise with a kernel 3x3
  cv::GaussianBlur(src_gray, dest1, cv::Size(3, 3), 0);
  cv::GaussianBlur(src_gray, dest2, cv::Size(9, 9), 0);
  cv::GaussianBlur(src_gray, dest3, cv::Size(17, 17), 0);

  // Canny detector
  cv::Canny(dest1, dest1, low_threshold, low_threshold*ratio);
  cv::Canny(dest2, dest2, low_threshold, low_threshold*ratio);
  cv::Canny(dest3, dest3, low_threshold, low_threshold*ratio);

  cv::imshow("guillermo edges1", dest1);
  cv::imshow("guillermo edges2", dest2);
  cv::imshow("guillermo edges3", dest3);
  cv::waitKey(0);

}
*/

void edgeletDetection(const ImgPyr& img_pyr)
{
  const int border = 10;
  const int level = 1;
  const float thresh = 200.0f;
  cv::Mat score(img_pyr[level].size(), CV_32FC1, cv::Scalar(0.0f));
  cv::Mat angle(img_pyr[level].size(), CV_8UC1, cv::Scalar(255));

  // compute image first derivative
  cv::Mat img, dx, dy;
  cv::GaussianBlur(img_pyr[level], img, cv::Size(3,3), 0);
  cv::Scharr(img, dx, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT);
  cv::Scharr(img, dy, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT);

  // compute angle and magnitude in angle direction
  const int max_row = dx.rows-border;
  const int max_col = dx.cols-border;
  for(int y = border; y < max_row; ++y)
  {
    int16_t* p_dx = dx.ptr<int16_t>(y);
    int16_t* p_dy = dy.ptr<int16_t>(y);
    for(int x = border; x < max_col; ++x)
    {
      angle.at<uint8_t>(y,x) = ((std::atan2(p_dy[x], p_dx[x]) + M_PI) / (2.0f * M_PI)) * 10;
      const float mag = std::sqrt(p_dx[x]*p_dx[x]+p_dy[x]*p_dy[x]);
      score.at<float>(y,x) = (mag > thresh) ? mag : 0.0f ;
    }
  }

    const int stride=score.step;

  /*
  // 8-neighbor nonmax suppression
  cv::Mat ang_best(score.size(), CV_8UC1, cv::Scalar(0));
  for(int y=2; y<score.rows-2; ++y)
  {
    const float* s = &score.at<float>(y,2);
    const uint8_t* a = &angle.at<uint8_t>(y,2);
    for(int x=2; x<score.cols-2; ++x, ++a, ++s)
    {
      if(*s == 0) continue;

      const uint8_t angle = *a;
      if( *(s+1) > 0 && *(a+1) != angle) continue;
      if( *(s-1) > 0 && *(a-1) != angle) continue;

      const float* const s1 = (s + stride);
      const float* const s2 = (s - stride);
      const uint8_t* const a1 = (a + stride);
      const uint8_t* const a2 = (a - stride);

      if( *(s1) > 0 && *(a1) != angle) continue;
      if( *(s2) > 0 && *(a2) != angle) continue;

      if( *(s1+1) > 0 && *(a1+1) != angle) continue;
      if( *(s1-1) > 0 && *(a1-1) != angle) continue;
      if( *(s2+1) > 0 && *(a2+1) != angle) continue;
      if( *(s2-1) > 0 && *(a2-1) != angle) continue;
      ang_best.at<uint8_t>(y,x) = 255;
    }
  }
  */

  // 8-neighbor nonmax suppression
  cv::Mat score_nonmax(score.size(), CV_8UC1, cv::Scalar(0));
  for(int y=2; y<score.rows-2; ++y)
  {
    const float* p = &score.at<float>(y,2);
    //const uint8_t* a = &ang_best.at<uint8_t>(y,2);
    for(int x=2; x<score.cols-2; ++x, ++p)//, ++a)
    {
      //if(*a == 0) continue;
      const float* const center=p;
      if(*center<thresh) continue;
      if(*(center+1)>=*center) continue;
      if(*(center-1)>*center) continue;
      const float* const p1=(center+stride);
      const float* const p2=(center-stride);
      if(*p1>=*center) continue;
      if(*p2>*center) continue;
      if(*(p1+1)>=*center) continue;
      if(*(p1-1)>*center) continue;
      if(*(p2+1)>=*center) continue;
      if(*(p2-1)>*center) continue;
      score_nonmax.at<uint8_t>(y,x) = 255;
    }
  }



  double minval, maxval;
  cv::minMaxLoc(score, &minval, &maxval);
  cv::Mat score_normalized = (score - minval) / (maxval - minval);
  cv::imshow("score", score_normalized);
  cv::imshow("score_nonmax", score_nonmax);
  //        cv::imshow("ang_best", ang_best);
  cv::waitKey(0);
}

void test()
{
  // Load dataset.
  std::string dataset_dir = ros::package::getPath("rpg_datasets")+"/rpg_urban_pinhole";
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  // Load detector.
  svo::DetectorOptions options;
  options.detector_type = svo::DetectorType::kGridGrad;
  options.cell_size = 40;
  options.threshold_secondary = 400;
  svo::AbstractDetectorPtr detector =
      svo::feature_detection_utils::makeDetector(options, dataset.cam());

  // Extract features on first dataset
  svo::FramePtr frame;
  dataset.getNextFrame(5u, frame, nullptr);

  /*
  detector->detect(frame);
  cv::Mat img_rgb_1;
  feature_detection_utils::drawFeatures(*frame, 0u, false, &img_rgb_1);
  cv::imshow("img_rgb_1", img_rgb_1);
  */

  detectNonlinearDiffusionEdges(frame->img_pyr_);

  // Elder Zucker
  if(false)
  {
    edgeletDetection(frame->img_pyr_);

    double sigma = 0.2*255;
    cv::Mat edge_map, level_map;
    elder_zucker::detectEdges(frame->img_pyr_, sigma, edge_map, level_map);

    double minval, maxval;
    cv::minMaxLoc(edge_map, &minval, &maxval);
    cv::Mat edge_map_normalized = (edge_map - minval) / (maxval - minval);
    cv::Mat level_map_normalized = level_map*40;

    cv::imshow("edge_map", edge_map_normalized);
    cv::imshow("level_map", level_map_normalized);

    constexpr int kHalfPatchSize = 4;
    constexpr int kPatchSize = 2 * kHalfPatchSize;
    uint8_t patch[kPatchSize*kPatchSize] __attribute__ ((aligned (16)));
    for(size_t i = 0; i < frame->numFeatures(); ++i)
    {
      size_t level = frame->level_vec_[i];

      // create patch
      warp::createPatchNoWarp(
            frame->img_pyr_[level], (frame->px_vec_.col(i)/ (1 << level)).cast<int>(),
            kHalfPatchSize, patch);
      cv::Mat patch_upsampled;
      cv::Mat patch_img(kPatchSize, kPatchSize, CV_8UC1, patch);
      patch_utils::normalizeAndUpsamplePatch(patch_img, kPatchSize, &patch_upsampled);

      // compute angle using histogram (GREEN
      double angle_hist = feature_detection_utils::getAngleAtPixelUsingHistogram(
            frame->img_pyr_[level], (frame->px_vec_.col(i)/(1<<level)).cast<int>(), kHalfPatchSize);
      cv::line(patch_upsampled,
               cv::Point2f(patch_upsampled.rows/2, patch_upsampled.rows/2),
               cv::Point2f(patch_upsampled.rows/2+15*std::cos(angle_hist), patch_upsampled.rows/2 +15*std::sin(angle_hist)),
               cv::Scalar(0,255,0), 3);

      // compute angle using gradient (RED)
      double angle_grad;
      feature_detection_utils::getCornerAngle
          (frame->img_pyr_, frame->px_vec_.col(i), level, &angle_grad);
      cv::line(patch_upsampled,
               cv::Point2f(patch_upsampled.rows/2, patch_upsampled.rows/2),
               cv::Point2f(patch_upsampled.rows/2+15*std::cos(angle_grad), patch_upsampled.rows/2 +15*std::sin(angle_grad)),
               cv::Scalar(0,0,255), 3);

      cv::imshow("patch", patch_upsampled);
      cv::waitKey(0);
      frame->grad_vec_.col(i) = GradientVector(std::cos(angle_hist), std::sin(angle_hist));
    }
    cv::Mat img_rgb_2;
    feature_detection_utils::drawFeatures(*frame, 0u, false, &img_rgb_2);
    cv::imshow("img_rgb_2", img_rgb_2);
    cv::waitKey(0);
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
