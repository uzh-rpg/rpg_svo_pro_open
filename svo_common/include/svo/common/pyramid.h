#pragma once

#include <cassert>
#include <Eigen/Core>

namespace svo {
namespace pyramid {

// halfSample() averages a 2x2 block. A level-l pixel p therefore represents
// 2^l*p + (2^l-1)/2 in level zero. Only positions need this centre offset;
// displacements and Jacobians still scale by powers of two.
// These conversions do not apply to OpenCV's optical-flow pyramids.
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 2, 1> pixelToLevel(
    const Eigen::MatrixBase<Derived>& pixel, const int level)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
  static_assert(!Eigen::NumTraits<typename Derived::Scalar>::IsInteger,
                "Pixel coordinates must use floating point");
  assert(level >= 0 && level < 31);
  using Scalar = typename Derived::Scalar;
  const Scalar scale = static_cast<Scalar>(1 << level);
  return ((pixel.array() - (scale - Scalar(1)) / Scalar(2)) / scale).matrix();
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 2, 1> pixelToLevelZero(
    const Eigen::MatrixBase<Derived>& pixel, const int level)
{
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
  static_assert(!Eigen::NumTraits<typename Derived::Scalar>::IsInteger,
                "Pixel coordinates must use floating point");
  assert(level >= 0 && level < 31);
  using Scalar = typename Derived::Scalar;
  const Scalar scale = static_cast<Scalar>(1 << level);
  return (pixel.array() * scale + (scale - Scalar(1)) / Scalar(2)).matrix();
}

} // namespace pyramid
} // namespace svo
