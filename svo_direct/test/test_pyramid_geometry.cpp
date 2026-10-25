#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <svo/common/camera.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/common/pyramid.h>
#include <svo/direct/feature_alignment.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/direct/matcher.h>
#include <svo/direct/patch_score.h>
#include <svo/direct/patch_warp.h>
#include <vikit/cameras.h>

namespace svo {
namespace {

CameraPtr makeCamera(const int size)
{
  return std::make_shared<vk::cameras::PinholeGeometry>(
      size, size, vk::cameras::PinholeProjection<vk::cameras::NoDistortion>(
          size/2, size/2, size/2, size/2, vk::cameras::NoDistortion()));
}

cv::Mat makeRamp(const bool cropped)
{
  cv::Mat storage(82, 82, CV_8UC1);
  cv::Mat image = cropped ? storage(cv::Rect(1, 1, 80, 80)) : cv::Mat(80, 80, CV_8UC1);
  for(int y = 0; y < image.rows; ++y)
    for(int x = 0; x < image.cols; ++x)
      image.at<uint8_t>(y, x) = std::min(255, 2*x + 2*y);
  return image;
}

TEST(PyramidGeometry, PixelCoordinatesMatchTheAveragedSampleCentres)
{
  ImgPyr pyramid_images;
  frame_utils::createImgPyramid(makeRamp(false), 4, pyramid_images);
  const double offsets[] = {0.0, 0.5, 1.5, 3.5};
  for(int level = 0; level < 4; ++level)
  {
    SCOPED_TRACE(level);
    const Keypoint pixel = pyramid::pixelToLevelZero(Keypoint(4, 5), level);
    EXPECT_DOUBLE_EQ(pixel.x(), 4*(1 << level) + offsets[level]);
    EXPECT_DOUBLE_EQ(pixel.y(), 5*(1 << level) + offsets[level]);
    // The ramp gives an independent measurement of the pyramid's sample centre.
    EXPECT_DOUBLE_EQ(pyramid_images[level].at<uint8_t>(5, 4), 2*pixel.x() + 2*pixel.y());
    const Eigen::Vector2f lower = pyramid::pixelToLevel(pixel.cast<float>(), level);
    EXPECT_FLOAT_EQ(lower.x(), 4.0f);
    EXPECT_FLOAT_EQ(lower.y(), 5.0f);
  }
}

TEST(PyramidGeometry, AffinePatchesSampleTheSameFullResolutionPosition)
{
  for(const bool cropped : {false, true})
  {
    ImgPyr images;
    frame_utils::createImgPyramid(makeRamp(cropped), 4, images);
    for(int ref_level = 0; ref_level < 4; ++ref_level)
    {
      for(int search_level = 0; search_level < 3; ++search_level)
      {
        SCOPED_TRACE(::testing::Message() << cropped << ", " << ref_level << ", " << search_level);
        uint8_t patch[16] = {};
        Keypoint pixel(40, 40);
        const warp::AffineTransformation2 affine =
            (1 << ref_level)*warp::AffineTransformation2::Identity();
        ASSERT_TRUE(warp::warpAffine(affine, images[ref_level], pixel,
                                    ref_level, search_level, 2, patch));
        EXPECT_EQ(patch[10], 160);
        EXPECT_EQ(patch[11], 160 + 2*(1 << search_level));
        EXPECT_EQ(patch[14], 160 + 2*(1 << search_level));
      }
    }
  }
}

TEST(PyramidGeometry, PyramidalAlignmentPreservesFractionalReferencePixels)
{
  cv::Mat image(256, 256, CV_8UC1);
  cv::RNG rng(1984);
  rng.fill(image, cv::RNG::UNIFORM, 0, 256);
  ImgPyr images;
  frame_utils::createImgPyramid(image, 4, images);
  const Keypoint reference(129.5, 125.5);
  for(int min_level = 0; min_level < 4; ++min_level)
  {
    SCOPED_TRACE(min_level);
    std::vector<cv::Point2f> ref(1, cv::Point2f(reference.x(), reference.y()));
    std::vector<cv::Point2f> cur = ref;
    std::vector<uint8_t> status(1, 0);
    feature_alignment::alignPyr2DVec(images, images, 3, min_level,
                                   std::vector<int>(4, 8), 10, .001f, ref, cur, status);
    ASSERT_EQ(status[0], 1);
    EXPECT_FLOAT_EQ(cur[0].x, reference.x());
    EXPECT_FLOAT_EQ(cur[0].y, reference.y());
  }
}

TEST(PyramidGeometry, PixelwisePatchesPreserveTheCameraRayAcrossLevels)
{
  Frame frame(makeCamera(80), makeRamp(false), 0, 4);
  frame.resizeFeatureStorage(1);
  FeatureWrapper feature = frame.getFeatureWrapper(0);
  feature.px = Keypoint(40, 40);
  feature.landmark = std::make_shared<Point>(Position(0, 0, 2));
  for(int ref_level = 0; ref_level < 4; ++ref_level)
  {
    for(int search_level = 0; search_level < 3; ++search_level)
    {
      SCOPED_TRACE(::testing::Message() << ref_level << ", " << search_level);
      uint8_t patch[16] = {};
      ASSERT_TRUE(warp::warpPixelwise(frame, frame, feature, ref_level, search_level, 2, patch));
      // Interpolation truncates to uint8_t after floating-point reprojection.
      EXPECT_NEAR(patch[10], 160, 1);
      EXPECT_NEAR(patch[11], 160 + 2*(1 << search_level), 1);
      EXPECT_NEAR(patch[14], 160 + 2*(1 << search_level), 1);
    }
  }
}

TEST(PyramidGeometry, DirectMatchingReturnsLevelZeroPixelsAfterChangingLevel)
{
  const CameraPtr camera = makeCamera(256);
  cv::Mat reference(256, 256, CV_8UC1), current(256, 256, CV_8UC1);
  const auto intensity = [](double x, double y) {
    return cv::saturate_cast<uint8_t>(128 + 40*std::sin(x*.12) + 40*std::cos(y*.15)
                                    + 25*std::sin((x+y)*.08));
  };
  for(int y = 0; y < 256; ++y)
    for(int x = 0; x < 256; ++x)
    {
      reference.at<uint8_t>(y, x) = intensity(x, y);
      current.at<uint8_t>(y, x) = intensity(128 + (x-128)*.5, 128 + (y-128)*.5);
    }
  Frame ref_frame(camera, reference, 0, 4), cur_frame(camera, current, 1, 4);
  cur_frame.T_f_w_.getPosition() = Position(0, 0, -1);
  ref_frame.resizeFeatureStorage(1);
  FeatureWrapper feature = ref_frame.getFeatureWrapper(0);
  feature.type = FeatureType::kCorner;
  feature.px = Keypoint(128, 128);
  feature.f = BearingVector(0, 0, 1);
  for(int level = 0; level <= 1; ++level)
  {
    SCOPED_TRACE(level);
    feature.level = level;
    Matcher matcher;
    matcher.options_.align_max_iter = 30;
    Keypoint pixel(128.2, 127.8);
    ASSERT_EQ(matcher.findMatchDirect(ref_frame, cur_frame, feature, 2, pixel), Matcher::MatchResult::kSuccess);
    EXPECT_EQ(matcher.search_level_, level + 1);
    EXPECT_NEAR(pixel.x(), 128, .2);
    EXPECT_NEAR(pixel.y(), 128, .2);
  }
}

TEST(PyramidGeometry, EpipolarScansScoreTheCorrectPyramidPixel)
{
  cv::Mat image(256, 256, CV_8UC1);
  cv::RNG rng(1984);
  rng.fill(image, cv::RNG::UNIFORM, 0, 256);
  const Frame frame(makeCamera(256), image, 0, 4);
  for(int level = 1; level < 4; ++level)
  {
    uint8_t patch[Matcher::kPatchSize*Matcher::kPatchSize] = {};
    warp::createPatchNoWarp(frame.img_pyr_[level], Eigen::Vector2i(10, 10), Matcher::kHalfPatchSize, patch);
    const Matcher::PatchScore score(patch);
    const double scale = 1 << level;
    const Keypoint pixel = Keypoint::Constant(10.4*scale + (scale-1)/2);
    BearingVector centre;
    ASSERT_TRUE(frame.cam()->backProject3(pixel, &centre));
    for(const bool sphere : {false, true})
    {
      SCOPED_TRACE(::testing::Message() << level << ", " << sphere);
      Matcher matcher;
      matcher.options_.scan_on_unit_sphere = sphere;
      matcher.options_.max_epi_search_steps = 1;
      matcher.epi_length_pyramid_ = .7;
      Keypoint best(-1, -1);
      int best_score = Matcher::PatchScore::threshold();
      matcher.scanEpipolarLine(frame, centre + BearingVector(.01, 0, 0), centre - BearingVector(.01, 0, 0),
                              centre, score, level, &best, &best_score);
      EXPECT_EQ(best_score, 0);
      EXPECT_NEAR((best-pixel).norm(), 0, 1e-6);
    }
  }
}

TEST(PyramidGeometry, FastDetectionRetainsFractionalLevelZeroCoordinates)
{
  cv::Mat image(256, 256, CV_8UC1);
  cv::RNG rng(1984);
  rng.fill(image, cv::RNG::UNIFORM, 0, 256);
  ImgPyr images;
  frame_utils::createImgPyramid(image, 4, images);
  for(int level = 1; level < 4; ++level)
  {
    SCOPED_TRACE(level);
    OccupandyGrid2D grid(30, 9, 9);
    Corners corners(grid.size(), Corner(0, 0, 0, 0, 0));
    feature_detection_utils::fastDetector(images, 2, 8, level, level, corners, grid);
    size_t count = 0;
    const double scale = 1 << level;
    for(size_t i = 0; i < corners.size(); ++i)
    {
      const Corner& corner = corners[i];
      if(corner.score <= 0)
        continue;
      ++count;
      EXPECT_EQ(grid.getCellIndex(Keypoint(corner.x, corner.y)), i);
      EXPECT_DOUBLE_EQ(corner.x - std::floor(corner.x/scale)*scale, (scale-1)/2);
      EXPECT_DOUBLE_EQ(corner.y - std::floor(corner.y/scale)*scale, (scale-1)/2);
    }
    EXPECT_GT(count, 0u);
  }
}

} // namespace
} // namespace svo

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
