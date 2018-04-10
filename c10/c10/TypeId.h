#pragma once

#include <cstdint>

// A compact identifier which stores all of the information necessary to
// carry out a dispatch on a type.  This is NOT NECESSARILY in one-to-one
// correspondence with the type hierarchy of TensorImpl, because we may decide
// that we want to refine dispatch on a runtime property of a tensor which is
// NOT reflected by the class hierarchy.
//
// Other note: there is no NATIVE notion of a subtyping relationship between
// these TypeIds.  We are planning to design one but we haven't decided on
// its specifics yet.
//
//    ezyang: CC smessmer; I know you wanted to have this line up exactly
//    with the concrete TensorImpl subclasses, but I don't want to commit
//    to that at the moment
//
// TODO: Does this also contain per Tensor properties, like contiguity?
//
// NB: NO default constructor
class TypeId final {
  const int64_t id_;
  explicit constexpr TypeId(int64_t id) noexcept : id_(id) {}
  friend class TypeIds;
};

class TypeIds final {
public:
  // These are just here for illustrative purposes
  static constexpr TypeId Undefined = TypeId(0);
  static constexpr TypeId CPUTensor = TypeId(1);
  static constexpr TypeId StridedCPUTensor = TypeId(2);
};
