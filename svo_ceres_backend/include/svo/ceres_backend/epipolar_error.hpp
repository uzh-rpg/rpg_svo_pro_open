/*********************************************************************************
 *
 *  Created on: Dec 4, 2018
 *      Author: Kunal Shrivastava (kunalshrivastava.iitr@gmail.com)
 *    Modified: Kunal Shrivastava
 *********************************************************************************/

/**
 * @file ceres_backend/epipolar_error.hpp
 * @brief Header file for the EpipolarError class.
 * @author Kunal Shrivastava
 */

#pragma once

#include <ceres/ceres.h>
#include <math.h>

namespace svo
{
namespace ceres_backend
{

class EpipolarError
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  EpipolarError(const double& sqrt_information, const Eigen::Matrix<double, 2, 1> p_1, const Eigen::Matrix<double, 2, 1> p_2):p_1_(p_1), p_2_(p_2)
  {
    sqrt_information_=sqrt_information;
  }

  ~EpipolarError(){}

  /*
   * Calculate the epipolar error in terms of the 2d feature coordinates and the relative pose.
   */

  template<typename T>
    bool operator()(const T* const t_rel, const T* const q_rel, T* e) const
    {
      Eigen::Map<const Eigen::Quaternion<T> > q_eig(q_rel);
      Eigen::Map<const Eigen::Matrix<T, 3, 1> > t_eig(t_rel);

      Eigen::Matrix < T, 3, 1 > p_1_hom;
      p_1_hom << T(p_1_(0, 0)), T(p_1_(1, 0)), T(1.0);
      Eigen::Matrix < T, 3, 1 > p_2_hom;
      p_2_hom << T(p_2_(0, 0)), T(p_2_(1, 0)), T(1.0);

      Eigen::Matrix<T, 3, 3> R = q_eig.toRotationMatrix();

      Eigen::Matrix<T, 3, 3> tx;
      tx << T(0), -t_eig(2,0), t_eig(1,0),
            t_eig(2,0), T(0), -t_eig(0,0),
            -t_eig(1,0), t_eig(0,0), T(0);

      T e_1 = p_2_hom.transpose()*tx*R*p_1_hom;
      T e_2 = p_1_hom.transpose()*R.transpose()*tx.transpose()*p_2_hom;
      e[0] = sqrt(pow(e_1,2) + pow(e_2,2));

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

  static ceres::CostFunction* Create(const double& sqrt_information, const Eigen::Matrix<double, 2, 1>& p_1, const Eigen::Matrix<double, 2, 1>& p_2)
  {
    return new ceres::AutoDiffCostFunction<EpipolarError, 1, 3, 4>(
            new EpipolarError(sqrt_information, p_1, p_2));
  }

private:
  Eigen::Matrix<double, 2, 1> p_1_;
  Eigen::Matrix<double, 2, 1> p_2_;
  double sqrt_information_;
};
}
}
