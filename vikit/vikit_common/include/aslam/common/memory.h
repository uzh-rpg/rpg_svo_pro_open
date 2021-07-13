#ifndef ASLAM_MEMORY_HELPERS_H_
#define ASLAM_MEMORY_HELPERS_H_

#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace aslam {

template<template<typename, typename> class Container, typename Type>
struct Aligned {
  typedef Container<Type, Eigen::aligned_allocator<Type> > type;
};

template<typename KeyType, typename ValueType>
struct AlignedUnorderedMap {
  typedef std::unordered_map<KeyType, ValueType,
      std::hash<KeyType>, std::equal_to<KeyType>,
      Eigen::aligned_allocator<std::pair<const KeyType, ValueType> > > type;
};

template<typename KeyType, typename ValueType>
struct AlignedMap {
  typedef std::map<KeyType, ValueType,
      std::less<KeyType>,
      Eigen::aligned_allocator<std::pair<const KeyType, ValueType> > > type;
};

/// \brief Aligned allocator to be used like std::make_shared
///
/// To be used like:
///   std::shared_ptr my_ptr = aligned_shared<aslam::RadTanDistortion>(params);
template<typename Type, typename ... Arguments>
inline std::shared_ptr<Type> aligned_shared(Arguments&&... arguments) {
  typedef typename std::remove_const<Type>::type TypeNonConst;
  return std::allocate_shared<Type>(Eigen::aligned_allocator<TypeNonConst>(),
                                    std::forward<Arguments>(arguments)...);
}

} // namespace aslam

#endif  // ASLAM_MEMORY_HELPERS_H_

