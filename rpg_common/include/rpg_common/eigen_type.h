// handy typedefs of Eigen matrices and vectors, adapted from
// https://github.com/zurich-eye/ze_oss/blob/master/ze_common/include/ze/common/types.hpp

// You can now use
// rpg::Matrix66 for a 6x6 matrix
// rpg::Matrix72 for a 7x2 matrix
// Valid for arbitrary combinations from 1-9.

#pragma once

#include <Eigen/Core>

namespace rpg_common
{
#define ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(SIZE, SUFFIX)                            \
  using Matrix##SUFFIX = Eigen::Matrix<double, SIZE, SIZE>;                    \
  using Matrix1##SUFFIX = Eigen::Matrix<double, 1, SIZE>;                      \
  using Matrix2##SUFFIX = Eigen::Matrix<double, 2, SIZE>;                      \
  using Matrix3##SUFFIX = Eigen::Matrix<double, 3, SIZE>;                      \
  using Matrix4##SUFFIX = Eigen::Matrix<double, 4, SIZE>;                      \
  using Matrix5##SUFFIX = Eigen::Matrix<double, 5, SIZE>;                      \
  using Matrix6##SUFFIX = Eigen::Matrix<double, 6, SIZE>;                      \
  using Matrix7##SUFFIX = Eigen::Matrix<double, 7, SIZE>;                      \
  using Matrix8##SUFFIX = Eigen::Matrix<double, 8, SIZE>;                      \
  using Matrix9##SUFFIX = Eigen::Matrix<double, 9, SIZE>;                      \
  using Matrix##SUFFIX##X = Eigen::Matrix<double, SIZE, Eigen::Dynamic>;       \
  using MatrixX##SUFFIX = Eigen::Matrix<double, Eigen::Dynamic, SIZE>;         \
  static const Eigen::MatrixBase<Matrix##SUFFIX>::IdentityReturnType           \
      I_##SUFFIX##x##SUFFIX = Matrix##SUFFIX::Identity();                      \
  static const Eigen::MatrixBase<Matrix##SUFFIX>::ConstantReturnType           \
      Z_##SUFFIX##x##SUFFIX = Matrix##SUFFIX::Zero()

ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(1, 1);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(2, 2);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(3, 3);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(4, 4);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(5, 5);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(6, 6);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(7, 7);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(8, 8);
ZE_MAKE_EIGEN_MATRIX_TYPEDEFS(9, 9);

#undef ZE_MAKE_EIGEN_MATRIX_TYPEDEFS

// Typedef arbitary length vector and arbitrary sized matrix.
using VectorX = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using MatrixX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXi = Eigen::VectorXi;

// Commonly used fixed size vectors.
using Vector1 = Eigen::Matrix<double, 1, 1>;
using Vector2 = Eigen::Matrix<double, 2, 1>;
using Vector3 = Eigen::Matrix<double, 3, 1>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Vector5 = Eigen::Matrix<double, 5, 1>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Vector7 = Eigen::Matrix<double, 7, 1>;
using Vector8 = Eigen::Matrix<double, 8, 1>;
using Vector9 = Eigen::Matrix<double, 9, 1>;
using Vector2i = Eigen::Vector2i;
}
namespace rpg = rpg_common;
