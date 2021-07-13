/*
 * homography.cpp
 *
 *  Created on: Sep 2, 2012
 *      by: cforster
 */

#include <glog/logging.h>
#include <vikit/homography.h>
#include <vikit/math_utils.h>
#include <vikit/homography_decomp.h> // copy of homography decomposition in opencv3
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace vk {

using namespace Eigen;

Homography estimateHomography(
    const Bearings& f_cur,
    const Bearings& f_ref,
    const double focal_length,
    const double reproj_error_thresh,
    const size_t min_num_inliers)
{
  // TODO: too long. split up and write tests

  CHECK_EQ(f_cur.cols(), f_ref.cols());
  const size_t N = f_cur.cols();
  const double thresh = reproj_error_thresh/focal_length;

  // compute homography using RANSAC
  std::vector<cv::Point2f> ref_pts(N), cur_pts(N);
  for(size_t i=0; i<N; ++i)
  {
    const Vector2d& uv_ref(vk::project2(f_ref.col(i)));
    const Vector2d& uv_cur(vk::project2(f_cur.col(i)));
    ref_pts[i] = cv::Point2f(uv_ref[0], uv_ref[1]);
    cur_pts[i] = cv::Point2f(uv_cur[0], uv_cur[1]);
  }
  cv::Mat H = cv::findHomography(ref_pts, cur_pts, cv::RANSAC, thresh);
  Matrix3d H_cur_ref;
  H_cur_ref << H.at<double>(0,0), H.at<double>(0,1), H.at<double>(0,2),
               H.at<double>(1,0), H.at<double>(1,1), H.at<double>(1,2),
               H.at<double>(2,0), H.at<double>(2,1), H.at<double>(2,2);

  // compute number of inliers
  std::vector<bool> inliers(N);
  size_t n_inliers = 0;
  for(size_t i=0; i<N; i++)
  {
    const Vector2d uv_cur = vk::project2(H_cur_ref * f_ref.col(i));
    const Vector2d e = vk::project2(f_cur.col(i)) - uv_cur;
    inliers[i] = (e.norm() < thresh);
    n_inliers += inliers[i];
  }

  VLOG(100) << "Homography has " << n_inliers << " inliers";
  if(n_inliers < min_num_inliers)
  {
    return Homography(); // return homography with score zero.
  }

  // compute decomposition
  cv::Matx33d K(1, 0, 0, 0, 1, 0, 0, 0, 1);
  std::vector<cv::Mat> rotations;
  std::vector<cv::Mat> translations;
  std::vector<cv::Mat> normals;
  cv::decomposeHomographyMat(H, K, rotations, translations, normals);
  CHECK_EQ(rotations.size(), 4u);

  // copy in decompositions struct
  std::vector<Homography> decomp;
  for(size_t i=0; i<4; ++i)
  {
    Homography d;
    const cv::Mat& t = translations[i];
    d.t_cur_ref = Vector3d(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    const cv::Mat& n = normals[i];
    d.n_cur = Vector3d(n.at<double>(0), n.at<double>(1), n.at<double>(2));
    const cv::Mat& R = rotations[i];
    d.R_cur_ref << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
                   R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
                   R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    decomp.push_back(d);
  }

  // check that plane is in front of camera
  for(Homography& D : decomp)
  {
    D.score = 0;
    for(size_t i=0; i<N; i++)
    {
      if(!inliers[i])
        continue;
      const double test = f_cur.col(i).dot(D.n_cur);
      if(test > 0.0)
        D.score += 1.0;
    }
  }
  std::sort(decomp.begin(), decomp.end(),
            [&](const Homography& lhs, const Homography& rhs)
            { return lhs.score > rhs.score; });
  decomp.resize(2);

  // According to Faugeras and Lustman, ambiguity exists if the two scores are equal
  // but in practive, better to look at the ratio!
  if(decomp[1].score/decomp[0].score < 0.9)
  {
    decomp.erase(decomp.begin() + 1); // no ambiguity
  }
  else
  {
    // two-way ambiguity: resolve by sampsonus score of all points.
    // Sampson error can be roughly thought as the squared distance between a
    // point x to the corresponding epipolar line x'F
    const double thresh_squared  = thresh * thresh * 4.0;
    for(Homography& D : decomp)
    {
      D.score = 0.0; // sum of sampsonus score
      const Matrix3d E_cur_ref = D.R_cur_ref * vk::skew(D.t_cur_ref); // Essential Matrix
      for(size_t i=0; i<N; ++i)
      {
        const double d = vk::sampsonDistance(f_cur.col(i), E_cur_ref, f_ref.col(i));
        D.score += std::min(d, thresh_squared);
      }
    }

    if(decomp[0].score < decomp[1].score)
      decomp.erase(decomp.begin() + 1);
    else
      decomp.erase(decomp.begin());
  }
  decomp[0].score = n_inliers;
  return decomp[0];
}

} // namespace vk
