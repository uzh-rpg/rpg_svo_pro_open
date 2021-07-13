#include "svo/online_loopclosing/map_alignment.h"

#include <iomanip>
#include <iostream>
#include <svo/common/frame.h>

namespace svo
{
MapAlignmentSE3::MapAlignmentSE3(
    const MapAlignmentOptions& map_alignment_options)
  : num_points_(0), max_num_points_(100), options_(map_alignment_options)
{
  points_old_.conservativeResize(Eigen::NoChange, max_num_points_);
  points_new_.conservativeResize(Eigen::NoChange, max_num_points_);
  ransac3d_min_pts_ = options_.ransac3d_min_pts;
  ransac3d_inlier_percent_ = options_.ransac3d_inlier_percent;
}

void MapAlignmentSE3::addCorrespondencePair(const Position& old_pos,
                                            const Position& new_pos)
{
  if (num_points_ >= points_old_.cols())
  {
    points_old_.conservativeResize(Eigen::NoChange, 2 * num_points_);
    points_new_.conservativeResize(Eigen::NoChange, 2 * num_points_);
  }
  points_old_.col(num_points_) = old_pos;
  points_new_.col(num_points_) = new_pos;
  points_vec_old_.push_back(old_pos);
  points_vec_new_.push_back(new_pos);
  ++num_points_;
  ++num_points_ransac_;
}

bool MapAlignmentSE3::getTransformation(Transformation& T_old_new)
{
  Transformation T_old_new_pre;
  Positions points_new_no_outliers;
  Positions points_old_no_outliers;
  points_old_no_outliers.resize(Eigen::NoChange, num_points_);
  points_new_no_outliers.resize(Eigen::NoChange, num_points_);

  // compute means to zero center data
  Position mean_old(Position::Zero());
  Position mean_new(Position::Zero());
  for (int i = 0; i < num_points_; ++i)
  {
    mean_old += points_old_.col(i);
    mean_new += points_new_.col(i);
  }
  mean_old /= num_points_;
  mean_new /= num_points_;
  // Reject outliers based on comparison with mean
  Position mean_old_no_outliers(Position::Zero());
  Position mean_new_no_outliers(Position::Zero());
  int num_points_new = 0;
  for (int i = 0; i < num_points_; ++i)
  {
    /* Don't close loop if associated points are too far apart. This can happen
     * if SVO didn't assign good points in the beginning for example.
     * ToDo: Find a better way for outlier rejection
     */
    if ((points_old_.col(i) - points_new_.col(i)).norm() < 1.73)
    {
      mean_old_no_outliers += points_old_.col(i);
      mean_new_no_outliers += points_new_.col(i);
      points_new_no_outliers.col(num_points_new) = points_new_.col(i);
      points_old_no_outliers.col(num_points_new) = points_old_.col(i);
      num_points_new++;
    }
  }
  mean_old_no_outliers /= num_points_new;
  mean_new_no_outliers /= num_points_new;
  mean_old = mean_old_no_outliers;
  mean_new = mean_new_no_outliers;
  int num_points_old = num_points_;
  num_points_ = num_points_new;

  //! @todo it seems to work fine with two points in general!
  if (num_points_ / (double)num_points_old < 0.5 && num_points_ < 8)
  {
    VLOG(40) << "Not Enough 3D inliers. Not Closing the loop.";
    return false;
  }

  // compute the transformation using the method of Horn (closed-form)
  Eigen::Matrix<FloatType, 3, 3> W(Eigen::Matrix<FloatType, 3, 3>::Zero());

  for (int i = 0; i < num_points_; ++i)
  {
    W += (points_new_no_outliers.col(i) - mean_new) *
         (points_old_no_outliers.col(i) - mean_old).transpose();
  }
  W /= num_points_;

  //  Eigen::JacobiSVD<Eigen::Matrix<FloatType,3,3> > svd(
  //          W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::BDCSVD<Eigen::Matrix<FloatType, 3, 3> > svd(
      W, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix<FloatType, 3, 3> S = Eigen::Matrix<FloatType, 3, 3>::Identity();
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
  {
    S(2, 2) = -1;
  }
  Eigen::Matrix<FloatType, 3, 3> R =
      svd.matrixU() * S * svd.matrixV().transpose();

  T_old_new_pre.getRotation() = Transformation::Rotation(R);
  T_old_new_pre.getPosition() = mean_new - R * mean_old;

  /* --------------------------------------------------------------------------------------------------------
   * This section will be removed after ransac is implemented */
  /* Calculate the residual and check if its small enough
   * Additionally remove outliers and do another iteration */
  Positions points_new_no_outliers_newit;
  Positions points_old_no_outliers_newit;
  points_old_no_outliers_newit.resize(Eigen::NoChange, num_points_);
  points_new_no_outliers_newit.resize(Eigen::NoChange, num_points_);
  Position mean_old_no_outliers_newit(Position::Zero());
  Position mean_new_no_outliers_newit(Position::Zero());
  num_points_new = 0;
  double residual = 0;
  for (int i = 0; i < num_points_; ++i)
  {
    double curr_res =
        (points_new_no_outliers.col(i) - R * points_old_no_outliers.col(i) -
         T_old_new_pre.getPosition())
            .norm();
    residual += curr_res;
    if (curr_res < 0.2)
    {
      mean_old_no_outliers_newit += points_old_no_outliers.col(i);
      mean_new_no_outliers_newit += points_new_no_outliers.col(i);
      points_new_no_outliers_newit.col(num_points_new) =
          points_new_no_outliers.col(i);
      points_old_no_outliers_newit.col(num_points_new) =
          points_old_no_outliers.col(i);
      num_points_new++;
    }
  }
  VLOG(40) << "Average Residual Error " << residual / num_points_;
  mean_old_no_outliers_newit /= num_points_new;
  mean_new_no_outliers_newit /= num_points_new;
  mean_old = mean_old_no_outliers_newit;
  mean_new = mean_new_no_outliers_newit;
  num_points_ = num_points_new;

  if (num_points_ < 4)
  {
    return false;
  }

  // compute the transformation using the method of Horn (closed-form)
  Eigen::Matrix<FloatType, 3, 3> W1(Eigen::Matrix<FloatType, 3, 3>::Zero());
  for (int i = 0; i < num_points_; ++i)
  {
    W1 += (points_new_no_outliers_newit.col(i) - mean_new) *
          (points_old_no_outliers_newit.col(i) - mean_old).transpose();
  }
  W1 /= num_points_;

  //  Eigen::JacobiSVD<Eigen::Matrix<FloatType,3,3> > svd(
  //          W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::BDCSVD<Eigen::Matrix<FloatType, 3, 3> > svd1(
      W1, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix<FloatType, 3, 3> S1 =
      Eigen::Matrix<FloatType, 3, 3>::Identity();
  if (svd1.matrixU().determinant() * svd1.matrixV().determinant() < 0)
  {
    S1(2, 2) = -1;
  }
  Eigen::Matrix<FloatType, 3, 3> R1 =
      svd1.matrixU() * S1 * svd1.matrixV().transpose();

  T_old_new.getRotation() = Transformation::Rotation(R1);
  T_old_new.getPosition() = mean_new - R1 * mean_old;
  residual = 0;
  for (int i = 0; i < num_points_; ++i)
  {
    double curr_res =
        (points_new_no_outliers_newit.col(i) -
         R1 * points_old_no_outliers_newit.col(i) - T_old_new.getPosition())
            .norm();
    residual += curr_res;
  }
  VLOG(40) << "Average Residual " << residual / num_points_;

  if (residual / num_points_ > 0.4)
  {
    return false;
  }

  /*-------------------------------------------------------------------------------------------------------------------------*/
  return true;

  /** Sim3 alignment --------------------------------------------------------
   // zero center data
   for(int i = 0; i<num_points_; ++i)
   {
   points_old_.col(i) -= mean_old;
   points_new_.col(i) -= mean_new;
   }

   // compute correlation
   Eigen::Matrix<FloatType,3,3> C =
   points_old_*points_new_.transpose()/num_points_;
   double sigma2 = points_old_.array().square().sum()/num_points_;
   Eigen::JacobiSVD<Eigen::Matrix<FloatType,3,3> > svd2(
   C.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
   Eigen::DiagonalMatrix<FloatType,3> D(svd2.singularValues());
   Eigen::Matrix<FloatType,3,3> S2 = Eigen::Matrix<FloatType,3,3>::Identity();
   if(svd2.matrixU().determinant()*svd2.matrixV().determinant()<0)
   {
   S2(2,2) = -1;
   }
   Eigen::Matrix<FloatType,3,3> R2 =
   svd2.matrixU()*S2*svd2.matrixV().transpose();
   T_old_new.getRotation() = Transformation::Rotation(R2);
   double scale = (D* S2).trace()/sigma2;

   T_old_new.getPosition() = mean_new - scale* R2*mean_old;

   return true;
   */
}

void MapAlignmentSE3::getClosest4DOFTransformInPlace(Transformation& T)
{
  Eigen::Matrix<double, 3, 3> R = T.getRotationMatrix();
  double A = -R(0, 1) + R(1, 0);
  double B = R(0, 0) + R(1, 1);
  double theta = /*M_PI/2 - */ atan2(A, B);

  std::cout << "Theta " << theta << std::endl;

  Eigen::Matrix<double, 3, 3> R_z;
  R_z << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1;

  T.getRotation() = Transformation::Rotation(R_z);
}

bool MapAlignmentSE3::getTransformRansac(const int min_num_3d,
                                         const bool relax_thresh,
                                         Transformation* T_old_new,
                                         std::vector<int>* inlier_indices)
{
  CHECK_NOTNULL(inlier_indices);
  if (num_points_ransac_ < min_num_3d)
  {
    VLOG(40) << "Not enough points for ransac";
    return false;
  }
  /* create point cloud adapter */
  opengv::point_cloud::PointCloudAdapter adapter(points_vec_new_,
                                                 points_vec_old_);

  /* formulate ransac problem */
  opengv::sac::Ransac<opengv::sac_problems::point_cloud::PointCloudSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::point_cloud::PointCloudSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::point_cloud::PointCloudSacProblem(adapter));
  ransac.sac_model_ = relposeproblem_ptr;
  int min_points_thresh = ransac3d_min_pts_;
  if (relax_thresh)
  {
    ransac.threshold_ = 0.1;
    min_points_thresh = 6;
  }
  else
  {
    ransac.threshold_ = 0.03;
  }
  ransac.max_iterations_ = 400;

  ransac.computeModel(0);
  size_t num_inliers = ransac.inliers_.size();
  VLOG(40) << "The number of inliers is: " << num_inliers;
  ransac_inliers_.conservativeResize(points_vec_new_.size());
  ransac_inliers_ = Eigen::VectorXd::Zero(points_vec_new_.size());
  for (size_t i = 0; i < ransac.inliers_.size(); i++)
  {
    ransac_inliers_(ransac.inliers_[i]) = 1;
  }
  VLOG(40) << "Ransac inlier vector size " << ransac_inliers_.rows()
           << std::endl;
  VLOG(40) << "Ransac Needed " << ransac.iterations_ << " iterations."
           << std::endl;
  opengv::transformation_t transform = ransac.model_coefficients_;
  Eigen::Matrix<FloatType, 3, 3> R = transform.block<3, 3>(0, 0);

  T_old_new->getRotation() = Transformation::Rotation(R);
  T_old_new->getPosition() = transform.col(3);
  *inlier_indices = ransac.inliers_;

  double res = 0;
  for (size_t i = 0; i < num_inliers; i++)
  {
    res += (points_vec_new_[ransac.inliers_[i]] -
            R * points_vec_old_[ransac.inliers_[i]] - transform.col(3))
               .norm();
  }
  VLOG(40) << "Average Residual after ransac " << res / num_inliers;

  if ((double)num_inliers / num_points_ransac_ * 100 <
          ransac3d_inlier_percent_ ||
      num_inliers < static_cast<size_t>(min_points_thresh))
  {
    VLOG(40) << "Not enough inliers after ransac. Not closing Loop";
    return false;
  }
  return true;
}

bool MapAlignmentSE3::solveJointOptimisation(
    const std::vector<cv::Point3f>& landmarks_lc,
    const std::vector<cv::Point3f>& landmarks_cf,
    const std::vector<cv::Point2f>& keypoints_lc,
    const std::vector<cv::Point2f>& keypoints_cf,
    const std::vector<std::pair<size_t, size_t> >& keypoint_correspondences,
    const std::vector<std::pair<size_t, size_t> >& point_correspondences,
    const Eigen::MatrixXd& rel_pose, const Transformation& T_w_lc,
    const size_t& num_bow_features_lc, const size_t& num_bow_features_cf,
    const int& min_num_3d)
{
  ceres::Problem problem;
  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres::Solver::Summary summary;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;
  Eigen::Matrix3d R = rel_pose.block(0, 0, 3, 3);
  Eigen::Vector3d t = rel_pose.block(0, 3, 3, 1);
  Eigen::Quaterniond q(R);
  int count_3d = -1;
  for (const std::pair<size_t, size_t>& correspondence :
       keypoint_correspondences)
  {
    if (correspondence.first < num_bow_features_lc &&
        correspondence.second < num_bow_features_cf)
    {
      // 2d match, added as epipolar error
      Eigen::Matrix<double, 2, 1> p_1;
      p_1 << (double)keypoints_lc[correspondence.second].x,
          (double)keypoints_lc[correspondence.second].y;
      Eigen::Matrix<double, 2, 1> p_2;
      p_2 << (double)keypoints_cf[correspondence.first].x,
          (double)keypoints_cf[correspondence.first].y;
      ceres::CostFunction* cost_function =
          ceres_backend::EpipolarError::Create(1.0, p_1, p_2);
      problem.AddResidualBlock(cost_function, loss_function, t.data(),
                               q.coeffs().data());
    }
  }
  for (const std::pair<size_t, size_t>& correspondence : point_correspondences)
  {
    count_3d++;
    if (ransac_inliers_(count_3d) == 0)
    {
      continue;
    }
    Eigen::Matrix2d sqrt_information;
    sqrt_information.setIdentity();
    Eigen::Matrix<double, 2, 1> p;
    p << (double)keypoints_cf[correspondence.second + num_bow_features_cf].x,
        (double)keypoints_cf[correspondence.second + num_bow_features_cf].y;
    /* Convert the 3D world point in camera frame */
    Eigen::Matrix<double, 3, 1> P_w;
    P_w << (double)landmarks_lc[correspondence.first].x,
        (double)landmarks_lc[correspondence.first].y,
        (double)landmarks_lc[correspondence.first].z;
    Eigen::Matrix<double, 3, 1> P_c;
    P_c = T_w_lc.inverse() * P_w;
    ceres::CostFunction* cost_function =
        ceres_backend::ReprojectionErrorSimple::Create(sqrt_information, p,
                                                       P_c);
    problem.AddResidualBlock(cost_function, loss_function, t.data(),
                             q.coeffs().data());
  }
  problem.SetParameterization(q.coeffs().data(),
                              quaternion_local_parameterization);
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;
  t_rel_combined_.getRotation() = Transformation::Rotation(q);
  t_rel_combined_.getPosition() = t;
  return true;
}

Transformation MapAlignmentSE3::getTransformationCombined(
    Transformation& T_w_lc, Transformation& T_w_cf)
{
  Transformation T_new_old, T_w_new;
  T_w_new = T_w_lc * t_rel_combined_;
  /* Note that this T_new_old transforms the T_cf (current frame or sliding
   * window), to its new location and
   * orientation (still in world frame).
   * So, T_w_cf_new (new cf) = T_new_old * T_w_cf_
   */
  T_new_old = T_w_new * T_w_cf.inverse();
  VLOG(40) << "T new old";
  VLOG(40) << T_new_old;
  return T_new_old;
}

void MapAlignmentSE3::reset()
{
  points_old_.conservativeResize(Eigen::NoChange, max_num_points_);
  points_new_.conservativeResize(Eigen::NoChange, max_num_points_);

  points_vec_new_.clear();
  points_vec_old_.clear();

  num_points_ = 0;
  num_points_ransac_ = 0;

  t_rel_combined_.setIdentity();

  ransac_inliers_.conservativeResize(0);
}
}  // namespace svo
