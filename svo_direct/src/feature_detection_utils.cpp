// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/direct/feature_detection_utils.h>

#include <algorithm>
#include <numeric>

#include <fast/fast.h>
#include <vikit/vision.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <svo/common/frame.h>
#include <svo/common/camera.h>
#include <svo/common/logging.h>
#include <svo/direct/feature_detection_types.h>
#include <svo/direct/feature_detection.h>

namespace svo {
namespace feature_detection_utils {

//------------------------------------------------------------------------------
AbstractDetector::Ptr makeDetector(
    const DetectorOptions& options,
    const CameraPtr& cam)
{
  AbstractDetector::Ptr detector;
  switch(options.detector_type)
  {
  case DetectorType::kFast:
    detector.reset(new FastDetector(options, cam));
    break;
  case DetectorType::kFastGrad:
    detector.reset(new FastGradDetector(options, cam));
    break;
  case DetectorType::kShiTomasi:
    detector.reset(new ShiTomasiDetector(options, cam));
    break;
  case DetectorType::kShiTomasiGrad:
    detector.reset(new ShiTomasiGradDetector(options, cam));
    break;
  case DetectorType::kGrad:
    detector.reset(new GradientDetector(options, cam));
    break;
  case DetectorType::kGridGrad:
    detector.reset(new GradientDetectorGrid(options, cam));
    break;
  case DetectorType::kAll:
    detector.reset(new AllPixelsDetector(options, cam));
    break;
  case DetectorType::kGradHuangMumford:
    detector.reset(new GradientHuangMumfordDetector(options, cam));
    break;
  case DetectorType::kCanny:
    detector.reset(new CannyDetector(options, cam));
    break;
  case DetectorType::kSobel:
    detector.reset(new SobelDetector(options, cam));
    break;
  default:
    SVO_ERROR_STREAM("Detector unknown!");
  }
  return detector;
}

//------------------------------------------------------------------------------
void fillFeatures(const Corners& corners,
    const FeatureType& type,
    const cv::Mat& mask,
    const double& threshold,
    const size_t max_n_features,
    Keypoints& keypoints,
    Scores& scores,
    Levels& levels,
    Gradients& gradients,
    FeatureTypes& types,
    OccupandyGrid2D& grid)
{
  CHECK_EQ(keypoints.cols(), levels.size());
  CHECK_EQ(keypoints.cols(), gradients.cols());
  CHECK_EQ(static_cast<size_t>(keypoints.cols()), types.size());

  // copy new features in temporary vectors
  aslam::Aligned<std::vector, Keypoint>::type keypoint_vec;
  aslam::Aligned<std::vector, GradientVector>::type gradient_vec;
  std::vector<Score> score_vec;
  std::vector<Level> level_vec;
  keypoint_vec.reserve(corners.size());
  gradient_vec.reserve(corners.size());
  level_vec.reserve(corners.size());
  score_vec.reserve(corners.size());
  for(const Corner& c : corners)
  {
    if(c.score > threshold)
    {
      // we don't need to check for grid occupancy since we already do that during
      // feature extraction.
      if(!mask.empty() &&
         mask.at<uint8_t>(static_cast<int>(c.y), static_cast<int>(c.x)) == 0)
      {
        continue;
      }
      keypoint_vec.emplace_back(Keypoint(c.x, c.y));
      level_vec.emplace_back(c.level);
      score_vec.emplace_back(c.score);
      gradient_vec.emplace_back(GradientVector(std::cos(c.angle), std::sin(c.angle)));
      grid.occupancy_[grid.getCellIndex(c.x, c.y)] = true;
      grid.feature_occupancy_[grid.getCellIndex(c.x, c.y)] = Keypoint(c.x, c.y);
    }
  }

  // Sort according to scores (bigger score should be better)
  std::vector<size_t> idx(score_vec.size());
  std::iota(idx.begin(), idx.end(), 0u);
  std::sort(idx.begin(), idx.end(), [&score_vec](size_t i1, size_t i2)
            { return score_vec[i1] > score_vec[i2]; });

  // copy temporary in eigen block
  const size_t n_old = keypoints.cols();
  const size_t n_new = std::min(max_n_features, keypoint_vec.size());
  const size_t n_tot = n_old + n_new;

  keypoints.conservativeResize(Eigen::NoChange, n_tot);
  gradients.conservativeResize(Eigen::NoChange, n_tot);
  scores.conservativeResize(n_tot);
  levels.conservativeResize(n_tot);
  types.resize(n_tot, type);

  for(size_t i = 0; i < n_new; ++i)
  {
    const size_t j = idx[i];
    keypoints.col(n_old+i) = keypoint_vec[j];
    gradients.col(n_old+i) = gradient_vec[j];
    scores(n_old+i) = score_vec[j];
    levels(n_old+i) = level_vec[j];   
  }
}

//------------------------------------------------------------------------------
void fastDetector(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const size_t min_level,
    const size_t max_level,
    Corners& corners,
    OccupandyGrid2D& grid)
{
  CHECK_EQ(corners.size(), grid.occupancy_.size());
  CHECK_LE(max_level, img_pyr.size()-1);

  for(size_t level=min_level; level<=max_level; ++level)
  {
    const int scale = (1<<level);
    std::vector<fast::fast_xy> fast_corners;
#if __SSE2__
    fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[level].data, img_pyr[level].cols,
          img_pyr[level].rows, img_pyr[level].step, threshold, fast_corners);
#elif HAVE_FAST_NEON
    fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[level].data, img_pyr[level].cols,
          img_pyr[level].rows, img_pyr[level].step, threshold, fast_corners);
#else
    fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].step, threshold, fast_corners);
#endif
    std::vector<int> scores, nm_corners;
    fast::fast_corner_score_10((fast::fast_byte*) img_pyr[level].data, img_pyr[level].step,
                               fast_corners, threshold, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    const int maxw = img_pyr[level].cols-border;
    const int maxh = img_pyr[level].rows-border;
    for(const int& i : nm_corners)
    {
      fast::fast_xy& xy = fast_corners.at(i);
      if(xy.x < border || xy.y < border || xy.x >= maxw || xy.y >= maxh)
        continue;
      const size_t k = grid.getCellIndex(xy.x, xy.y, scale);
      if(grid.occupancy_.at(k))
        continue;
      const float score = scores.at(i); //vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(k).score)
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, level, 0.0f);
    }
  }
}

//------------------------------------------------------------------------------
void shiTomasiDetector(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const size_t min_level,
    const size_t max_level,
    Corners& corners,
    OccupandyGrid2D& grid,
    OccupandyGrid2D& closeness_check_grid)
{
  CHECK_EQ(corners.size(), grid.occupancy_.size());
  CHECK_LE(max_level, img_pyr.size()-1);

  for(size_t level=min_level; level<=max_level; ++level)
  {
    size_t nrows = img_pyr[level].rows;
    size_t ncols = img_pyr[level].cols;
    const int scale = (1<<level);
    std::vector<Eigen::Vector2i> shitomasi_corners;
    std::vector<int> scores, nm_corners;
    for(size_t i=0; i<nrows; ++i)
    {
      for(size_t j=0; j<ncols; ++j)
      {
        double score;
        if(getShiTomasiScore(img_pyr[level], Eigen::Vector2i(i,j), &score))
        {
          if(score > threshold)
          {
            shitomasi_corners.push_back(Eigen::Vector2i(i,j));
            scores.push_back(score);
          }
        }
      }
    }
    nonmax_3x3(shitomasi_corners, scores, nm_corners);
    const int maxw = ncols-border;
    const int maxh = nrows-border;
    for(const int& i : nm_corners)
    {
      Eigen::Vector2i xy = shitomasi_corners.at(i);
      if(xy.x() < border || xy.y() < border || xy.x() >= maxw || xy.y() >= maxh)
        continue;
      const size_t occ_k = grid.getCellIndex(xy.x(), xy.y(), scale);
      if (grid.occupancy_.at(occ_k))
        continue;
      const size_t close_k =
          closeness_check_grid.getCellIndex(xy.x(), xy.y(), scale);
      if(closeness_check_grid.occupancy_.at(close_k))
      {
        Keypoint kp = closeness_check_grid.feature_occupancy_.at(close_k);
        Eigen::Vector2i kp_eig = Eigen::Vector2i(kp(0,0), kp(1,0));
        Eigen::Vector2i diff = kp_eig - xy;
        if(diff(0,0)<5 || diff(1,0)<5)
        {
          continue;
        }
      }
      const float score = scores.at(i); //vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      if(score > corners.at(occ_k).score)
      {
        corners.at(occ_k) = Corner(xy.x()*scale, xy.y()*scale, score, level, 0.0f);
      }
    }
  }
}

//------------------------------------------------------------------------------
void edgeletDetector_V1(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const int min_level,
    const int max_level,
    Corners& corners,
    OccupandyGrid2D& grid)
{
  CHECK_EQ(corners.size(), grid.occupancy_.size());

  for(int level=min_level+1; level<=max_level; ++level) // note, we start one level higher, so we already have some smoothing
  {
    const int scale = (1<<level);

    // compute image first derivative
    cv::Mat dx, dy;
    cv::Scharr(img_pyr[level], dx, CV_16S, 1, 0, 1, 0, cv::BORDER_DEFAULT);
    cv::Scharr(img_pyr[level], dy, CV_16S, 0, 1, 1, 0, cv::BORDER_DEFAULT);

    // compute angle and magnitude in angle direction
    const int max_row = dx.rows-border;
    const int max_col = dx.cols-border;
    for(int y=border; y<max_row; ++y)
    {
      int16_t* p_dx = dx.ptr<int16_t>(y);
      int16_t* p_dy = dy.ptr<int16_t>(y);
      for(int x=border; x<max_col; ++x)
      {
        const int k = grid.getCellIndex(x, y, scale);
        if(grid.occupancy_.at(k))
          continue;
        Corner& c = corners.at(k);
        const double score = std::sqrt(p_dx[x]*p_dx[x]+p_dy[x]*p_dy[x]);
        if(score > c.score)
        {
          c.x=x*scale;
          c.y=y*scale;
          c.level=level-1;
          c.score=score;
          c.angle=std::atan2(p_dy[x],p_dx[x]);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
void edgeletDetector_V2(
    const ImgPyr& img_pyr,
    const int threshold,
    const int border,
    const int min_level,
    const int max_level,
    Corners& corners,
    OccupandyGrid2D& grid)
{
  CHECK_EQ(corners.size(), grid.occupancy_.size());

  constexpr int level = 1;
  constexpr int scale = (1<<level);
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
      score.at<float>(y,x) = (mag > threshold) ? mag : 0.0f ;
    }
  }

  // 8-neighbor nonmax suppression
  const int stride=score.step;
  for(int y=border; y<score.rows-border; ++y)
  {
    const float* p = &score.at<float>(y,border);
    for(int x=border; x<score.cols-border; ++x, ++p)
    {
      const int k = grid.getCellIndex(x, y, scale);
      if(grid.occupancy_.at(k))
        continue;

      const float* const center=p;
      if(*center<threshold) continue;
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

      Corner& c = corners.at(k);
      if(*p > c.score)
      {
        c.x=x*scale;
        c.y=y*scale;
        c.level=level-1;
        c.score=*p;
        c.angle = getAngleAtPixelUsingHistogram(img_pyr[level], Eigen::Vector2i(x,y), 4);
      }
    }
  }
}


//------------------------------------------------------------------------------
bool getCornerAngle(
    const ImgPyr& img_pyr,
    const Eigen::Ref<const Keypoint>& px_level_0,
    const size_t level,
    double* angle)
{
  CHECK_NOTNULL(angle);
  constexpr int kHalfPatchSize = 2;
  constexpr int kPatchSize = 2*kHalfPatchSize;
  constexpr int kHalfPatchSizePlusBorder = kHalfPatchSize + 1;
  const cv::Mat& img = img_pyr.at(level);
  const Eigen::Vector2i px = (px_level_0 / (1 << level)).cast<int>();

  if(px(0) < kHalfPatchSizePlusBorder
     || px(1) < kHalfPatchSizePlusBorder
     || px(0) >= img.cols - kHalfPatchSizePlusBorder
     || px(1) >= img.rows - kHalfPatchSizePlusBorder)
  {
    VLOG(100) << "Can't determine corner angle. Patch too close to border. "
              << "px = " << px.transpose() << ", level = " << level;
    return false;
  }

  const int stride = img.step;
  float dx = 0.0f, dy = 0.0f;
  for(int y = 0; y < kPatchSize; ++y)
  {
    uint8_t* it = img.data + (px(1) + y - kHalfPatchSize) * stride + (px(0) - kHalfPatchSize);
    for(int x = 0; x < kPatchSize; ++x, ++it)
    {
      dx += it[1] - it[-1];
      dy += it[stride] - it[-stride];
    }
  }
  *angle = std::atan2(dy, dx);
  return true;
}

//------------------------------------------------------------------------------
bool getShiTomasiScore(
    const cv::Mat& img,
    const Eigen::Vector2i& px,
    double* score)
{
  CHECK_NOTNULL(score);
  CHECK(img.type() == CV_8UC1);
  constexpr int kHalfPatchSize = 4;
  constexpr int kPatchSize = 2 * kHalfPatchSize;
  constexpr int kPatchArea = kPatchSize * kPatchSize;

  const int x_min = px(0) - kHalfPatchSize;
  const int x_max = px(0) + kHalfPatchSize;
  const int y_min = px(1) - kHalfPatchSize;
  const int y_max = px(1) + kHalfPatchSize;

  if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
    return false;

  float dXX = 0.0;
  float dYY = 0.0;
  float dXY = 0.0;
  const int stride = img.step.p[0];
  for( int y=y_min; y<y_max; ++y )
  {
    const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
    const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
    const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
    const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
    for(int x = 0; x < kPatchSize; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom)
    {
      float dx = *ptr_right - *ptr_left;
      float dy = *ptr_bottom - *ptr_top;
      dXX += dx*dx;
      dYY += dy*dy;
      dXY += dx*dy;
    }
  }

  // Find and return smaller eigenvalue:
  dXX = dXX / (2.0f * kPatchArea);
  dYY = dYY / (2.0f * kPatchArea);
  dXY = dXY / (2.0f * kPatchArea);
  *score = 0.5f * (dXX + dYY - std::sqrt( (dXX - dYY) * (dXX - dYY) + 4 * dXY * dXY  ));
  return true;
}

//------------------------------------------------------------------------------
void setCornerAngles(const ImgPyr& img_pyr, Corners* corners)
{
  // compute dominant gradient direction at corner
  const int stride = img_pyr[0].step;
  for(Corners::iterator it=corners->begin(); it!=corners->end(); ++it)
  {
    uint8_t* p = (uint8_t*) img_pyr[0].data + it->y*stride + it->x;
    float dx = 0.5f*(p[1]-p[-1]);
    float dy = 0.5f*(p[stride]-p[-stride]);
    it->angle = std::atan2(dy, dx);
  }
}

//------------------------------------------------------------------------------
void setCornerLevel(const ImgPyr& mag_pyr, Corners* corners)
{
  const size_t max_level = mag_pyr.size();
  for(Corners::iterator it=corners->begin(); it!=corners->end(); ++it)
  {
    float max_mag = mag_pyr[0].at<float>(it->y, it->x);
    float max_lev = 0;
    for(size_t level=1; level<max_level; ++level)
    {
      const int scale = 1.0f / (1<<level);
      const int u = static_cast<int>(scale*it->x+0.5f);
      const int v = static_cast<int>(scale*it->y+0.5f);
      float mag = mag_pyr[level].at<float>(v,u);
      if(mag > max_mag)
      {
        max_mag = mag;
        max_lev = level;
      }
    }
    it->level = max_lev;
  }
}

//------------------------------------------------------------------------------
void computeDerivMaxMagnitude(
    const cv::Mat& img_8u,
    cv::Mat& mag_8u)
{
  mag_8u = cv::Mat::zeros(img_8u.size(), CV_8UC1);
  const int stride=img_8u.step;
  for(int y=1; y<img_8u.rows-1; ++y)
  {
    uint8_t* mag_ptr = (uint8_t*) mag_8u.data + y*stride + 1;
    uint8_t* img_ptr = (uint8_t*) img_8u.data + y*stride + 1;
    for(int x=1; x<img_8u.cols-1; ++x, ++mag_ptr, ++img_ptr)
      *mag_ptr = std::max(abs(img_ptr[1]-img_ptr[-1]), abs(img_ptr[stride]-img_ptr[-stride]));
    //*mag_ptr = abs(img_ptr[1]-img_ptr[-1]) + abs(img_ptr[stride]-img_ptr[-stride]);
  }
}

//------------------------------------------------------------------------------
void computeDerivHuangMumford(
    const cv::Mat& img_8u,
    cv::Mat& mag_32f,
    float alpha,
    float q)
{
  cv::Size sz = img_8u.size();
  const int img_stride=img_8u.step;
  mag_32f = cv::Mat::zeros(sz, CV_32FC1);
  //  cv::Mat dx(sz, CV_32FC1);
  //  cv::Mat dy(sz, CV_32FC1);

  for (int y=0; y<sz.height; ++y)
  {
    // get rows
    const uint8_t* rimg = img_8u.ptr<const uint8_t>(y);
    //    float* rgx = dx.ptr<float>(y);
    //    float* rgy = dy.ptr<float>(y);
    float* rmag = mag_32f.ptr<float>(y);

    for (int x=0; x<sz.width; ++x)
    {
      float img_c = (float)rimg[x];
      float img_xp = (x==sz.width-1) ? (float)rimg[x] : (float)rimg[x+1];
      float img_yp = (y==sz.height-1) ? (float)rimg[x] : (float)rimg[x+img_stride];
      // uchar -> float
      img_c /= 255.0f;
      img_xp /= 255.0f;
      img_yp /= 255.0f;
      float dx = std::max(1e-3f,
                          std::exp(-alpha * std::pow(std::abs(img_xp - img_c), q)));
      float dy = std::max(1e-3f,
                          std::exp(-alpha * std::pow(std::abs(img_yp - img_c), q)));
      //      rgx[x] = dx;
      //      rgy[x] = dy;
      rmag[x] = std::sqrt(dx*dx + dy*dy);
    }
  }
}

//------------------------------------------------------------------------------
void nonmax(
    const cv::Mat& img_32f,
    const float thresh,
    Corners* corners)
{
  // 8-neighbor nonmax suppression
  const int stride=img_32f.step;
  for(int y=2; y<img_32f.rows-2; ++y)
  {
    const float* p = &img_32f.at<float>(y,2);
    for(int x=2; x<img_32f.cols-2; ++x, ++p)
    {
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
      corners->push_back(Corner(x, y, *center, 0, 0.0f));
    }
  }
}

//------------------------------------------------------------------------------
void displayGrid(
    const OccupandyGrid2D& old_grid,
    const Keypoints& keypoints,
    const int img_width,
    const int img_height)
{
  // copy detected features in feature_occupancy grid
  OccupandyGrid2D new_grid(old_grid.cell_size, old_grid.n_cols, old_grid.n_rows);

  for(int i=0; i<keypoints.cols(); ++i)
  {
    new_grid.occupancy_.at(
          new_grid.getCellIndex(keypoints(0,i), keypoints(1,i))) = true;
  }

  // create image of grid with previously occupied cells in gray, new detected
  // features in white and cells without features in black.
  cv::Mat img(img_height, img_width, CV_8UC1);
  for(int y=0; y<img_height; ++y)
  {
    uchar* img_ptr = img.ptr<uchar>(y);
    for(int x=0; x<img_width; ++x)
    {
      if(old_grid.occupancy_.at(old_grid.getCellIndex(x,y)))
        img_ptr[x] = 125;
      else if(new_grid.occupancy_.at(new_grid.getCellIndex(x,y)))
        img_ptr[x] = 255;
      else
        img_ptr[x] = 0;
    }
  }

  cv::imshow("detector", img);
  cv::waitKey(1);
}


void nonlinearDiffusion(const cv::Mat& img_8u,
                        cv::Mat& img_8u_diffused,
                        const double timestep,
                        const double final_time)
{
  const int num_iter = final_time / timestep;

  cv::Mat Lt;
  img_8u.convertTo(Lt, CV_32FC1, 1.0f/255.0);
  cv::Mat kernel_deriv_x  = (cv::Mat_<float>(1,3) << -0.5, 0.0, 0.5);
  cv::Mat kernel_deriv_xx = (cv::Mat_<float>(1,3) << 1.0, -2.0, 1.0);
  cv::Mat kernel_deriv_xy = (cv::Mat_<float>(3,3) << 0.25, 0.00,-0.25,
                                                     0.00, 0.00, 0.00,
                                                    -0.25, 0.00, 0.25);
  // Apply filter
  cv::Point anchor = cv::Point(-1,-1);
  cv::Mat Lx, Ly;
  cv::Mat Lxx, Lxy, Lyy;
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

    Lt = Lt + timestep * numerator / denom;
  }

  img_8u_diffused = Lt;
  Lt.convertTo(img_8u_diffused, CV_8UC1, 255.0);
}

//------------------------------------------------------------------------------
void detectCannyEdges(
    const cv::Mat& src_gray,
    cv::Mat& dest,
    int low_threshold,
    int ratio,
    int smoothing_kernel_size)
{
  smoothing_kernel_size = 21;

  // Reduce noise
  //cv::blur(src_gray, dest, cv::Size(smoothing_kernel_size, smoothing_kernel_size));
  //cv::GaussianBlur(src_gray, dest, cv::Size(smoothing_kernel_size, smoothing_kernel_size), 0);
  nonlinearDiffusion(src_gray, dest, 0.25, 5);

  cv::imshow("diffused image", dest);
  cv::waitKey(10);

  // Canny detector
  cv::Canny(dest, dest, low_threshold, low_threshold*ratio);
}

//------------------------------------------------------------------------------
void detectSobelEdges(
    const cv::Mat& src_gray,
    cv::Mat& dest,
    int low_threshold,
    int kernel_size)
{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  cv::Mat blurred_input, grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  cv::blur( src_gray, blurred_input, cv::Size(kernel_size,kernel_size) );
  cv::Sobel( blurred_input, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );
  cv::Sobel( blurred_input, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_y, abs_grad_y );

  // Total Gradient (approximate)
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dest );

  cv::threshold(dest, dest, low_threshold,
      255, cv::THRESH_BINARY);
}

//------------------------------------------------------------------------------
void drawFeatures(
    const Frame& frame,
    const size_t level,
    const bool only_matched_features,
    cv::Mat* img_rgb)
{
  CHECK_NOTNULL(img_rgb);
  CHECK_GT(frame.img_pyr_.size(), level);

  const int scale = (1<<level);
  const int size = (level == 1) ? 2 : 0;
  *img_rgb = cv::Mat(frame.img_pyr_[level].size(), CV_8UC3);
  cv::cvtColor(frame.img_pyr_[level], *img_rgb, cv::COLOR_GRAY2RGB);
  if(level == 0)
  {
    for(size_t i = 0; i < frame.num_features_; ++i)
    {
      const auto& px = frame.px_vec_.col(i);
      if(frame.landmark_vec_[i] == nullptr
         && frame.seed_ref_vec_[i].keyframe == nullptr
         && only_matched_features)
        continue;

      const auto& g = frame.grad_vec_.col(i);
      switch (frame.type_vec_[i])
      {
        case FeatureType::kEdgelet:
          cv::line(*img_rgb, cv::Point2f(px(0) + 3 * g(1), px(1) - 3 * g(0)),
                   cv::Point2f(px(0) - 3 * g(1), px(1) + 3 * g(0)),
                   cv::Scalar(255, 0, 255), 2);
          break;
        case FeatureType::kCorner:
          cv::rectangle(*img_rgb, cv::Point2f(px(0) - 2, px(1) - 2),
                        cv::Point2f(px(0) + 2, px(1) + 2),
                        cv::Scalar(0, 255, 0), -1);
          break;
        case FeatureType::kMapPoint:
          cv::rectangle(*img_rgb, cv::Point2f(px(0) - 2, px(1) - 2),
                        cv::Point2f(px(0) + 2, px(1) + 2),
                        cv::Scalar(255, 0, 0), -1);
          break;
        case FeatureType::kFixedLandmark:
          cv::rectangle(*img_rgb, cv::Point2f(px(0) - 3, px(1) - 3),
                        cv::Point2f(px(0) + 3, px(1) + 3),
                        cv::Scalar(101, 236, 255), -1);
          break;
        case FeatureType::kEdgeletSeed:
        case FeatureType::kEdgeletSeedConverged:
          cv::line(*img_rgb, cv::Point2f(px(0) + 3 * g(1), px(1) - 3 * g(0)),
                   cv::Point2f(px(0) - 3 * g(1), px(1) + 3 * g(0)),
                   cv::Scalar(0, 0, 255), 2);
          break;
        case FeatureType::kCornerSeed:
        case FeatureType::kCornerSeedConverged:
          cv::circle(*img_rgb, cv::Point2f(px(0), px(1)),
                     5, cv::Scalar(0, 255, 0), 1);
          break;
        case FeatureType::kMapPointSeed:
        case FeatureType::kMapPointSeedConverged:
          cv::circle(*img_rgb, cv::Point2f(px(0), px(1)),
                     5, cv::Scalar(255, 0, 0), 1);
          break;
        default:
          cv::circle(*img_rgb, cv::Point2f(px(0), px(1)),
                     5, cv::Scalar(0, 0, 255), -1);
          break;
      }
    }
  }
  else
  {
    for(size_t i = 0; i < frame.num_features_; ++i)
    {
      if(frame.isValidLandmark(i))
      {
        const auto& px = frame.px_vec_.col(i);
        cv::rectangle(*img_rgb,
                      cv::Point2f(px(0)/scale-size, px(1)/scale-size),
                      cv::Point2f(px(0)/scale+size, px(1)/scale+size),
                      cv::Scalar(0,255,0), -1);
      }
      else if(isCornerEdgeletSeed(frame.type_vec_[i]))
      {
        const auto& px = frame.px_vec_.col(i);
        cv::rectangle(*img_rgb,
                      cv::Point2f(px(0)/scale-size, px(1)/scale-size),
                      cv::Point2f(px(0)/scale+size, px(1)/scale+size),
                      cv::Scalar(255,0,255), -1);
      }
//      else if(isMapPoint(frame.type_vec_[i]))
//      {
//        const auto& px = frame.px_vec_.col(i);
//        cv::rectangle(*img_rgb,
//                      cv::Point2f(px(0)/scale-size, px(1)/scale-size),
//                      cv::Point2f(px(0)/scale+size, px(1)/scale+size),
//                      cv::Scalar(255,0,0), -1);
//      }
    }
  }
}

double getAngleAtPixelUsingHistogram(
    const cv::Mat& img, const Eigen::Vector2i& px, const size_t halfpatch_size)
{
  angle_hist::AngleHistogram hist;
  hist.fill(0.0);
  angle_hist::angleHistogram(img, px(0), px(1), halfpatch_size, hist);
  angle_hist::smoothOrientationHistogram(hist);
  return angle_hist::getDominantAngle(hist);
}

void nonmax_3x3(const std::vector<Eigen::Vector2i>& corners, const std::vector<int>& scores,
                std::vector<int>& nonmax_corners)
{
  nonmax_corners.clear();
  nonmax_corners.reserve(corners.size());

  if (corners.size() < 1)
    return;

  // Find where each row begins
  // (the corners are output in raster scan order). A beginning of -1 signifies
  // that there are no corners on that row.
  int last_row = corners.back().x();
  std::vector<int> row_start(last_row + 1, -1);

  int prev_row = -1;
  for (unsigned int i = 0; i < corners.size(); i++)
    if (corners[i].x() != prev_row)
    {
      row_start[corners[i].x()] = i;
      prev_row = corners[i].x();
    }

  // Point above points (roughly) to the pixel above the one of interest, if there
  // is a feature there.
  int point_above = 0;
  int point_below = 0;

  const int sz = (int)corners.size();

  for (int i = 0; i < sz; i++)
  {
    int score = scores[i];
    Eigen::Vector2i pos = corners[i];

    // Check left
    if (i > 0)
      // if(corners[i-1] == pos-ImageRef(1,0) && (scores[i-1] >= score))
      if (corners[i - 1].y() == pos.y() - 1 && corners[i - 1].x() == pos.x() && scores[i - 1] >= score)
        continue;

    // Check right
    if (i < (sz - 1))
      // if(corners[i+1] == pos+ImageRef(1,0) &&  (scores[i+1] >= score))
      if (corners[i + 1].y() == pos.y() + 1 && corners[i + 1].x() == pos.x() && scores[i + 1] >= score)
        continue;

    // Check above (if there is a valid row above)
    if (pos.x() != 0 && row_start[pos.x() - 1] != -1)
    {
      // Make sure that current point_above is one
      // row above.
      if (corners[point_above].x() < pos.x() - 1)
        point_above = row_start[pos.x() - 1];

      // Make point_above point to the first of the pixels above the current point,
      // if it exists.
      for (; corners[point_above].x() < pos.x() && corners[point_above].y() < pos.y() - 1; point_above++)
      {
      }

      for (int i = point_above; corners[i].x() < pos.x() && corners[i].y() <= pos.y() + 1; i++)
      {
        int y = corners[i].y();
        if ((y == pos.y() - 1 || y == pos.y() || y == pos.y() + 1) && (scores[i] >= score))
          goto cont;
      }
    }
    // Check below (if there is anything below)
    if (pos.x() != last_row && row_start[pos.x() + 1] != -1 && point_below < sz)  // Nothing below
    {
      if (corners[point_below].x() < pos.x() + 1)
        point_below = row_start[pos.x() + 1];

      // Make point below point to one of the pixels belowthe current point, if it
      // exists.
      for (; point_below < sz && corners[point_below].x() == pos.x() + 1 && corners[point_below].y() < pos.y() - 1;
           point_below++)
      {
      }

      for (int i = point_below; i < sz && corners[i].x() == pos.x() + 1 && corners[i].y() <= pos.y() + 1; i++)
      {
        int y = corners[i].y();
        if ((y == pos.y() - 1 || y == pos.y() || y == pos.y() + 1) && (scores[i] >= score))
          goto cont;
      }
    }
    nonmax_corners.push_back(i);

  cont:;
  }
}

void mergeGrids(const OccupandyGrid2D& grid1, OccupandyGrid2D* grid2)
{
  CHECK(grid1.occupancy_.size() == grid2->occupancy_.size());
  for(size_t i=0; i<grid1.occupancy_.size(); i++)
  {
    if(grid2->occupancy_.at(i))
      continue;
    grid2->occupancy_.at(i) = grid1.occupancy_.at(i);
    grid2->feature_occupancy_.at(i) = grid1.feature_occupancy_.at(i);
  }
}

namespace angle_hist {

void angleHistogram(
    const cv::Mat& img, int x, int y, int halfpatch_size, AngleHistogram& hist)
{
  constexpr double pi2 = 2.0 * M_PI;
  for(int dy = -halfpatch_size; dy <= halfpatch_size; ++dy)
  {
    for(int dx = -halfpatch_size; dx <= halfpatch_size; ++dx)
    {
      double mag, angle;
      if(gradientAndMagnitudeAtPixel(img, x + dx, y + dy, &mag, &angle))
      {
        size_t bin = std::round( n_bins * ( angle + M_PI ) / pi2 );
        bin = (bin < n_bins) ? bin : 0u;
        hist[bin] += mag;
      }
    }
  }
}

bool gradientAndMagnitudeAtPixel(
    const cv::Mat& img, int x, int y, double* mag, double* angle)
{
  if(y > 0 && y < img.rows-1 && x > 0  &&  x < img.cols-1 )
  {
    double dx = img.at<uint8_t>(y, x+1) - img.at<uint8_t>(y, x-1);
    double dy = img.at<uint8_t>(y+1, x) - img.at<uint8_t>(y-1, x);
    *mag = std::sqrt( dx*dx + dy*dy );
    *angle = std::atan2( dy, dx );
    return true;
  }
  return false;
}

void smoothOrientationHistogram(AngleHistogram& hist)
{
  double prev = hist[n_bins-1], h0 = hist[0];
  for(size_t i = 0; i < n_bins; ++i)
  {
    double tmp = hist[i];
    hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * ( (i+1 == n_bins) ? h0 : hist[i+1]);
    prev = tmp;
  }
}

double getDominantAngle(const AngleHistogram& hist)
{
  double max_angle;
  int max_bin;

  max_angle = hist[0];
  max_bin = 0;
  for(size_t i = 1; i < n_bins; i++ )
  {
    if(hist[i] > max_angle)
    {
      max_angle = hist[i];
      max_bin = i;
    }
  }
  return max_bin * 2.0 * M_PI / n_bins;
}

} // angle_hist
} // namespace feature_detection_utils
} // namespace svo

