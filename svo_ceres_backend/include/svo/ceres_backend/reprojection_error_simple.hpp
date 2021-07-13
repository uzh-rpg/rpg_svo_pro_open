/*********************************************************************************
 *
 *  Created on: Dec 4, 2018
 *      Author: Kunal Shrivastava (kunalshrivastava.iitr@gmail.com)
 *    Modified: Kunal Shrivastava
 *********************************************************************************/

/**
 * @file ceres_backend/reprojection_error_simple.hpp
 * @brief Header file for the ReprojectionErrorSimple class.
 * @author Kunal Shrivastava
 */

#include <ceres/ceres.h>
#include <math.h>

namespace svo
{
namespace ceres_backend
{
class ReprojectionErrorSimple
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionErrorSimple(const Eigen::Matrix2d& sqrt_information, const Eigen::Matrix<double, 2, 1>& p,
                          const Eigen::Matrix<double, 3, 1>& P) :
      p_(p), P_(P)
  {
    sqrt_information_ = sqrt_information;
  }
  ;

  ~ReprojectionErrorSimple()
  {
  }
  ;

  /*
   * Calculate the reprojection error in terms of the relative pose, using known 3d points and 2d feature locations.
   */

  template<typename T>
    bool operator()(const T* const t_rel, const T* const q_rel, T* e) const
    {
      Eigen::Map<const Eigen::Quaternion<T> > q_eig(q_rel);
      Eigen::Map<const Eigen::Matrix<T, 3, 1> > t_eig(t_rel);

      Eigen::Matrix<T, 4, 1> P_hom;
      P_hom << T(P_(0, 0)), T(P_(1, 0)), T(P_(2, 0)), T(1.0);

      Eigen::Matrix<T, 3, 3> R = q_eig.toRotationMatrix();

      Eigen::Matrix<T, 3, 4> T_C;
      T_C.block(0, 0, 3, 3) = R.transpose();
      T_C.block(0, 3, 3, 1) = -R.transpose() * t_eig;

      Eigen::Matrix<T, 3, 1> p_dash_hom_eig = T_C * P_hom;

      Eigen::Matrix<T, 2, 1> p_dash_eig;

      p_dash_eig << p_dash_hom_eig(0, 0) / p_dash_hom_eig(2, 0), p_dash_hom_eig(1, 0) / p_dash_hom_eig(2, 0);

      Eigen::Matrix<T, 2, 1> e_weighed = sqrt_information_.cast<T>() * (p_.cast<T>() - p_dash_eig);

      e[0] = e_weighed(0, 0);
      e[1] = e_weighed(1, 0);

      return true;
    }

  template<typename T>
    Eigen::Matrix<T, 3, 3> quatToRotMat(Eigen::Quaternion<T> q) const
    {
      Eigen::Matrix<T, 3, 3> R;

      R(0, 0) = q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z();
      R(1, 0) = 2.0 * q.w() * q.z() + 2.0 * q.x() * q.y();
      R(2, 0) = 2.0 * q.x() * q.z() - 2.0 * q.w() * q.y();

      R(0, 1) = 2.0 * q.x() * q.y() - 2.0 * q.w() * q.z();
      R(1, 1) = q.w() * q.w() - q.x() * q.x() + q.y() * q.y() - q.z() * q.z();
      R(2, 1) = 2.0 * q.w() * q.x() + 2.0 * q.y() * q.z();

      R(0, 2) = 2.0 * q.w() * q.y() + 2.0 * q.x() * q.z();
      R(1, 2) = 2.0 * q.y() * q.z() - 2.0 * q.w() * q.x();
      R(2, 2) = q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z();

      return R;
    }

  static ceres::CostFunction* Create(const Eigen::Matrix2d& sqrt_information, const Eigen::Matrix<double, 2, 1>& p,
                                     const Eigen::Matrix<double, 3, 1>& P)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionErrorSimple, 2, 3, 4>(
        new ReprojectionErrorSimple(sqrt_information, p, P));
  }

private:
  Eigen::Matrix2d sqrt_information_;
  Eigen::Matrix<double, 2, 1> p_;
  Eigen::Matrix<double, 3, 1> P_;
};
}
}
