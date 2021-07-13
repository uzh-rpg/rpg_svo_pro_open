#ifndef ASLAM_COMMON_MACROS_H_
#define ASLAM_COMMON_MACROS_H_

#include <memory>

#define ASLAM_DISALLOW_EVIL_CONSTRUCTORS(TypeName)     \
  TypeName(const TypeName&) = delete;                  \
  void operator=(const TypeName&) = delete

#define ASLAM_POINTER_TYPEDEFS(TypeName)               \
  typedef std::unique_ptr<TypeName> UniquePtr;         \
  typedef std::shared_ptr<TypeName> Ptr;               \
  typedef std::shared_ptr<const TypeName> ConstPtr

/// Extract the type from an expression which wraps a type inside braces. This
/// is done to protect the commas in some types.
template<typename T> struct ArgumentType;
template<typename T, typename U> struct ArgumentType<T(U)> { typedef U Type; };
#define GET_TYPE(TYPE) ArgumentType<void(TYPE)>::Type

#endif  // ASLAM_COMMON_MACROS_H_
