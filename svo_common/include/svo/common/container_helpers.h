#pragma once

#include <algorithm>
#include <vector>
#include <algorithm>

#include <glog/logging.h>

namespace svo {
namespace common {
namespace container_helpers {

template <typename Type>
void eraseIndicesFromVector(const std::vector<size_t>& sorted_indices_to_remove,
                            std::vector<Type>* vector)
{
  CHECK_NOTNULL(vector);
  std::vector<Type> to_swap;
  to_swap.reserve(vector->size() - sorted_indices_to_remove.size());

  size_t in_begin = 0u;

  auto copy_until_and_skip = [&](const size_t end_index) {
    CHECK_GE(end_index, in_begin) << "Indices to remove must be sorted!";
    // TODO(tcies) Potential optimization: Use move iterators. Can't use memcpy
    // because that will e.g. fail on shared pointers.
    to_swap.insert(to_swap.end(), vector->begin() + in_begin,
                   vector->begin() + end_index);
    in_begin = end_index + 1;
  };

  // Copy blocks before indices to remove.
  for (const size_t index_to_remove : sorted_indices_to_remove)
  {
    copy_until_and_skip(index_to_remove);
  }

  // Copy block after last index to remove.
  copy_until_and_skip(vector->size());

  vector->swap(to_swap);
}


// Solution from:
// http://stackoverflow.com/questions/7571937/how-to-delete-items-from-a-stdvector-given-a-list-of-indices
template<typename ElementType, typename Allocator>
inline std::vector<ElementType, Allocator> eraseIndicesFromVector_DEPRECATED(
    const std::vector<ElementType, Allocator>& data,
    const std::vector<size_t>& indices_to_delete) {
  if (indices_to_delete.empty()) {
    return data;
  }
  std::vector<size_t> mutable_indices_to_delete = indices_to_delete;
  std::sort(mutable_indices_to_delete.begin(), mutable_indices_to_delete.end());
  CHECK_LT(mutable_indices_to_delete.back(), data.size());

  std::vector<ElementType, Allocator> reduced_vector;
  CHECK_GE(data.size(), mutable_indices_to_delete.size());
  reduced_vector.reserve(data.size() - mutable_indices_to_delete.size());

  // Copy blocks from the input vector to the output vector.
  typename std::vector<ElementType, Allocator>::const_iterator it_block_begin = data.begin();

  for (typename std::vector<size_t>::const_iterator it = mutable_indices_to_delete.begin();
      it != mutable_indices_to_delete.end(); ++it) {
    typename std::vector<ElementType, Allocator>::const_iterator it_block_end = data.begin() + *it;
    if (it_block_begin != it_block_end) {
      std::copy(it_block_begin, it_block_end, std::back_inserter(reduced_vector));
    }
    it_block_begin = it_block_end + 1;
  }

  // Copy the last block.
  if (it_block_begin != data.end()) {
    std::copy(it_block_begin, data.end(), std::back_inserter(reduced_vector));
  }
  return reduced_vector;
}

}  // namespace container_helpers
}  // namespace common
}  // namespace svo
