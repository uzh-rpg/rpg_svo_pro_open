#pragma once

#include <cstdlib> // size_t, fabs
#define _USE_MATH_DEFINES // for M_PI
#include <cmath>   // sin, cos
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <kindr/minimal/quat-transformation.h>

namespace vk {

using Transformation = kindr::minimal::QuatTransformation;
using Quaternion = kindr::minimal::RotationQuaternion;

using Bearings = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor>;

using std::size_t;
using std::uint8_t;

//TODO(cfo) template, take eigen::Ref, and work with minkindr trafo
Eigen::Vector3d triangulateFeatureNonLin(
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const Eigen::Vector3d& feature1,
    const Eigen::Vector3d& feature2);

/// Assumes the bearing vectors f_c and f_r are on the epipolar plane, i.e.
/// perfect triangulation without noise!
bool depthFromTriangulationExact(
    const Eigen::Matrix3d& R_r_c,
    const Eigen::Vector3d& t_r_c,
    const Eigen::Vector3d& f_r,
    const Eigen::Vector3d& f_c,
    double& depth_in_r,
    double& depth_in_c);

double reprojError(
    const Eigen::Vector3d& f1,
    const Eigen::Vector3d& f2,
    double error_multiplier2);

double computeInliers(
    const Bearings& features1, ///< c1
    const Bearings& features2, ///< c2
    const Eigen::Matrix3d& R, ///< R_c1_c2
    const Eigen::Vector3d& t, ///< c1_t
    const double reproj_thresh,
    double error_multiplier2,
    Bearings& xyz_vec, ///< in frame c1
    std::vector<int>& inliers,
    std::vector<int>& outliers
);

void computeInliersOneView(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& feature_sphere_vec,
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& xyz_vec,
    const Eigen::Matrix3d &R,
    const Eigen::Vector3d &t,
    const double reproj_thresh,
    const double error_multiplier2,
    std::vector<int>& inliers,
    std::vector<int>& outliers);

/// Direct Cosine Matrix to Roll Pitch Yaw
Eigen::Vector3d dcm2rpy(const Eigen::Matrix3d &R);

/// Roll Pitch Yaw to Direct Cosine Matrix
Eigen::Matrix3d rpy2dcm(const Eigen::Vector3d &rpy);

/// Angle Axis parametrization to Quaternion
Eigen::Quaterniond angax2quat(const Eigen::Vector3d& n, const double& angle);

/// Angle Axis parametrization to Matrix representation
Eigen::Matrix3d angax2dcm(const Eigen::Vector3d& n, const double& angle);

/// Spherical linear interpolation. t should be in [0,1]
Quaternion slerp(const Quaternion& R0, const Quaternion& R1, double t);

/// Sampsonus score
double sampsonDistance(
    const Eigen::Vector3d& f_cur,
    const Eigen::Matrix3d& E_cur_ref,
    const Eigen::Vector3d& f_ref);

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v)
{
  Eigen::Matrix3d v_sqew;
  v_sqew << 0, -v[2], v[1],
            v[2], 0, -v[0],
            -v[1], v[0], 0;
  return v_sqew;
}

/// From GTSAM
inline Eigen::Matrix3d rightJacobianExpMapSO3(const Eigen::Vector3d& x)
{
  // x is the axis-angle representation (exponential coordinates) for a rotation
  const double normx = x.norm(); // rotation angle
  Eigen::Matrix3d Jr;
  if (normx < 10e-8){
    Jr = Eigen::Matrix3d::Identity();
  }
  else{
    const Eigen::Matrix3d X = vk::skew(x); // element of Lie algebra so(3): X = x^
    Jr = Eigen::Matrix3d::Identity() - ((1-cos(normx))/(normx*normx)) * X +
        ((normx-sin(normx))/(normx*normx*normx)) * X * X; // right Jacobian
  }
  return Jr;
}

/// From GTSAM
inline Eigen::Matrix3d rightJacobianExpMapSO3inverse(const Eigen::Vector3d& x)
{
  // x is the axis-angle representation (exponential coordinates) for a rotation
  const double normx = x.norm(); // rotation angle
  Eigen::Matrix3d Jrinv;
  if (normx < 10e-8)
  {
    Jrinv = Eigen::Matrix3d::Identity();
  }
  else
  {
    const Eigen::Matrix3d X = vk::skew(x); // element of Lie algebra so(3): X = x^
    Jrinv = Eigen::Matrix3d::Identity() +
        0.5 * X + (1/(normx*normx) - (1+cos(normx))/(2*normx * sin(normx))   ) * X * X;
  }
  return Jrinv;
}

inline double norm_max(const Eigen::VectorXd & v)
{
  double max = -1;
  for (int i=0; i<v.size(); i++)
  {
    double abs = std::fabs(v[i]);
    if(abs>max){
      max = abs;
    }
  }
  return max;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 2, 1> project2(const Eigen::MatrixBase<Derived>& v)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 1);
  return v.template head<2>() / v(2);
}

inline Eigen::Vector3d unproject2d(const Eigen::Vector2d& v)
{
  return Eigen::Vector3d(v[0], v[1], 1.0);
}

inline Eigen::Vector3d project3d(const Eigen::Vector4d& v)
{
  return v.head<3>()/v[3];
}

inline Eigen::Vector4d unproject3d(const Eigen::Vector3d& v)
{
  return Eigen::Vector4d(v[0], v[1], v[2], 1.0);
}

template<class T>
T getMedian(std::vector<T>& data_vec)
{
  assert(!data_vec.empty());
  typename std::vector<T>::iterator it = data_vec.begin()+std::floor(data_vec.size()/2);
  std::nth_element(data_vec.begin(), it, data_vec.end());
  return *it;
}

inline double pyrFromZero_d(double x_0, int level)
{
  return x_0/(1<<level); // = 1 / 2^level
}

inline Eigen::Vector2d pyrFromZero_2d(const Eigen::Vector2d& uv_0, int level)
{
  return Eigen::Vector2d(pyrFromZero_d(uv_0[0], level),
                         pyrFromZero_d(uv_0[1], level));
}

template<class T>
inline T normPdf(const T x, const T mean, const T sigma)
{
  T exponent = x - mean;
  exponent *= -exponent;
  exponent /= 2 * sigma * sigma;
  T result = std::exp(exponent);
  result /= sigma * std::sqrt(2 * M_PI);
  return result;
}

/// Frame jacobian for projection of 3D point in (f)rame coordinate to
/// unit plane coordinates uv (focal length = 1).
inline void jacobianFrame_xyz2uv(
    const Eigen::Vector3d& xyz_in_f,
    Eigen::Matrix<double,2,6>& J)
{
  const double x = xyz_in_f[0];
  const double y = xyz_in_f[1];
  const double z_inv = 1./xyz_in_f[2];
  const double z_inv_2 = z_inv*z_inv;
  J(0,0) = -z_inv;              // -1/z
  J(0,1) = 0.0;                 // 0
  J(0,2) = x*z_inv_2;           // x/z^2
  J(0,3) = y*J(0,2);            // x*y/z^2
  J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
  J(0,5) = y*z_inv;             // y/z
  J(1,0) = 0.0;                 // 0
  J(1,1) = -z_inv;              // -1/z
  J(1,2) = y*z_inv_2;           // y/z^2
  J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
  J(1,4) = -J(0,3);             // -x*y/z^2
  J(1,5) = -x*z_inv;            // x/z
}

/// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
inline void jacobianPoint_xyz2uv(
    const Eigen::Vector3d& p_in_f,
    const Eigen::Matrix3d& R_f_w,
    Eigen::Matrix<double,2,3>& J)
{
  const double z_inv = 1.0/p_in_f[2];
  const double z_inv_sq = z_inv*z_inv;
  J(0,0) = z_inv;
  J(0,1) = 0.0;
  J(0,2) = -p_in_f[0] * z_inv_sq;
  J(1,0) = 0.0;
  J(1,1) = z_inv;
  J(1,2) = -p_in_f[1] * z_inv_sq;
  J = - J * R_f_w;
}

} // end namespace vk
