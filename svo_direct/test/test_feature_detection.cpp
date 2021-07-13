// svo
#include <svo/test_utils/synthetic_dataset.h>
#include <svo/common/frame.h>
#include <svo/direct/matcher.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>

// others
#include <opencv2/highgui/highgui.hpp> // imread
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat CannyThreshold(cv::Mat& src_gray,int lowThreshold = 20, int ratio = 3, int kernel_size = 3)
{
  cv::Mat detected_edges;
  /// Reduce noise with a kernel 3x3
  cv::blur( src_gray, detected_edges, cv::Size(3,3) );

  /// Canny detector
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  return detected_edges;
}

int main(int argc, char **argv)
{
  // Load dataset.
  std::string dataset_dir = ros::package::getPath("rpg_datasets")+"/rpg_vfr_pinhole";
  svo::test_utils::SyntheticDataset dataset(dataset_dir, 0, 0);

  svo::DetectorOptions options;
  options.detector_type = svo::DetectorType::kGradHuangMumford;

  if(argc >= 4)
    options.threshold_primary = std::atof(argv[3]);

  svo::AbstractDetectorPtr detector =
      svo::feature_detection_utils::makeDetector(options, dataset.cam());
  svo::FramePtr frame;
  while(dataset.getNextFrame(5u, frame, nullptr))
  {
    cv::Mat img_src = image_loader.loadNext();
    Eigen::VectorXd intrinsics(4,1);
    intrinsics << 315.5, 315.5, 376.0, 240.0;
    svo::CameraPtr cam = aslam::createCamera<aslam::PinholeCamera>(intrinsics, img_src.cols, img_src.rows);
    svo::DetectorOptions detectorOptions;
    svo::GradientHuangMumfordDetector detector(detectorOptions, cam);

    svo::ImgPyr mag_pyr_32f(frame->img_pyr_.size());
    cv::Mat canny_img;

    cv::Mat blurred_input;

    for(size_t i=0; i<frame->img_pyr_.size(); ++i)
    {
      cv::blur( frame->img_pyr_[i], blurred_input, cv::Size(5,5) );
      svo::feature_detection_utils::computeDerivHuangMumford(
            blurred_input, mag_pyr_32f[i], atof(std::argv[1]), std::atof(std::argv[2]));
    }

    cv::Mat mag_level_32f = mag_pyr_32f[detectorOptions.sampling_level];
    cv::Mat mag_level_thresholded;
    cv::threshold(mag_level_32f, mag_level_thresholded, detectorOptions.threshold_primary,
                  1.0, CV_THRESH_BINARY_INV);

    cv::imshow("mag_level_32f", mag_level_32f);
    cv::imshow("mag_level_thresholded", mag_level_thresholded);

    canny_img = CannyThreshold(frame->img_pyr_[0]);
    cv::namedWindow("Canny");
    cv::imshow("Canny",canny_img);

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    cv::Mat grad_x, grad_y, grad;
    cv::Mat abs_grad_x, abs_grad_y;

    cv::blur( frame->img_pyr_[0], blurred_input, cv::Size(3,3) );
    cv::Sobel( blurred_input, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //cv::Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    cv::Sobel( blurred_input, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    cv::threshold(grad, grad, atof(std::argv[4]),
        255, CV_THRESH_BINARY);

    cv::imshow( "window_name", grad );
    cv::waitKey(0);
  }

  return 0;
}
