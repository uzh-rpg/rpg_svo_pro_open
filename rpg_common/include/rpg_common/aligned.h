#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <glog/logging.h>

#include "rpg_common/eigen_hash.h"

// See https://eigen.tuxfamily.org/dox-devel/group__TopicStlContainers.html
namespace rpg_common {

// e.g. rpg::Aligned<std::vector, Eigen::Vector2d>
// Consider using Matrix2Xd instead in such cases.
template <template <typename Type, typename Allocator> class StlContainer,
typename EigenType>
using Aligned = StlContainer<
    EigenType, Eigen::aligned_allocator<EigenType>>;

namespace aligned {

template <typename KeyType, typename EigenType>
using Map =
    std::map<KeyType, EigenType, std::less<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, EigenType>>>;

template <typename KeyType, typename EigenType>
using UnorderedMap =
    std::unordered_map<KeyType, EigenType, std::hash<KeyType>,
    std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, EigenType>>>;

template <typename EigenType, typename ValueType>
using UnorderedMapEigenKey =
    std::unordered_map<EigenType, ValueType, std::hash<EigenType>,
    std::equal_to<EigenType>,
    Eigen::aligned_allocator<std::pair<const EigenType, ValueType>>>;

template <typename EigenType>
using UnorderedSet =
    std::unordered_set<EigenType, std::hash<EigenType>,
    std::equal_to<EigenType>, Eigen::aligned_allocator<EigenType>>;

// See cols_vec.h for the inverse.
template <typename Scalar, int Rows>
Eigen::Matrix<Scalar, Rows, Eigen::Dynamic> toMat(
    const Aligned<std::vector, Eigen::Matrix<Scalar, Rows, 1>>& container)
{
  Eigen::Matrix<Scalar, Rows, Eigen::Dynamic> result(Rows, container.size());
  for (size_t i = 0u; i < container.size(); ++i)
  {
    result.col(i) = container[i];
  }
  return result;
}

}  // namespace aligned
}  // namespace rpg_common
namespace rpg = rpg_common;
