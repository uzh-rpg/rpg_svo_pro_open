#include "epipolar_error.hpp"
#include "reprojection_error_simple.hpp"

#include <svo/common/types.h>
#include <svo/common/transformation.h>

#include <opengv/point_cloud/methods.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>

namespace svo
{
struct MapAlignmentOptions
{
  int ransac3d_min_pts;
  double ransac3d_inlier_percent;
};

class MapAlignmentSE3
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MapAlignmentSE3(const MapAlignmentOptions& map_alignment_options);
  ~MapAlignmentSE3(){}

  /**
   * @brief Reset map alignment class
   */
  void reset();

  /**
   * @brief Get optimal transformation from old 3d position to the new 3d
   * positions of corresponding points
   * @param T_old_new Transformation such that T_old_new*p_old = p_new
   */
  bool getTransformation(Transformation &T_old_new);

  /**
   * @brief Add corresponding pair of old and new position of same point
   * @param old_pos position of the point before
   * @param new_pos position after the transformation
   * @return true if successful
   */
  void addCorrespondencePair(const Position &old_pos, const Position &new_pos);

  /**
   * @brief Get transformation from old 3d position to new using ransac
   * @param T_old_new Transformation such that T_old_new*p_old = p_new
   */
  bool getTransformRansac(Transformation &T_old_new, const int& min_num_3d, int& n_ransac_inliers,  bool& relax_thresh);

  /**
   * @brief Get transformation by jointly minimizing epipolar and
   * reprojection error from 2d and 3d matches respectively
   * @param T_old_new Transformation such that T_old_new*p_old = p_new
   */
  Transformation getTransformationCombined(Transformation& T_w_lc, Transformation& T_w_cf);

  /**
   * @brief Jointly optimize the epipolar and reprojection error to get relative pose
   */
  bool solveJointOptimisation(const std::vector<cv::Point3f>& landmarks_lc, const std::vector<cv::Point3f>& landmarks_cf,
                              const std::vector<cv::Point2f>& keypoints_lc, const std::vector<cv::Point2f>& keypoints_cf,
                              const std::vector<std::pair<size_t, size_t> > &keypoint_correspondences,
                              const std::vector<std::pair<size_t, size_t> > &point_correspondences,
                              const Eigen::MatrixXd& rel_pose, const Transformation& T_w_lc, const size_t& num_bow_features_lc,
                              const size_t& num_bow_features_cf, const int& min_num_3d);

  typedef std::shared_ptr<MapAlignmentSE3> Ptr;

  inline int getMinRansacPts()
  {
    return ransac3d_min_pts_;
  }

private:
  Positions points_new_;
  Positions points_old_;

  opengv::points_t points_vec_new_;
  opengv::points_t points_vec_old_;

  int num_points_ = 0;
  int num_points_ransac_ = 0;

  //params
  const int max_num_points_;
  int ransac3d_min_pts_;
  double ransac3d_inlier_percent_;

  MapAlignmentOptions options_;

  Transformation t_rel_combined_;

  Eigen::VectorXd ransac_inliers_;
};
} //namespace svo
