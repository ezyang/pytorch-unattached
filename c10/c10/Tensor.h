#pragma once

#include "guts/Retainable.h"
#include "ArrayRef.h"

namespace c10 { namespace guts {

  class TensorImpl;
  class UndefinedTensorImpl;

}}

namespace c10 {

// Design notes:
//  - Manual retain/release instead of shared_ptr. Reasons:
//      - PRIMARY: It's possible to work with the underlying retained object using
//        a C API, which is basically impossible to do with shared_ptr because
//        it doesn't expose a manual retain()/release() API
//      - SECONDARY: A true intrusive reference count has some nice properties
//        which you don't get from use of std::make_shared (to put the refcount
//        metadata next to the regular dynamic allocation) and
//        std::enabled_shared_from_this (which generally needs to store a weak pointer
//        to the control block).
// - guts::UndefinedTensorImpl instead of null pointer. Reasons:
//      - We originally had a null pointer in ATen, but this meant that when we
//        incorrectly attempted to use such a null pointer, we would segfault and
//        crash, which is very unfriendly for our Python users.  Using an guts::UndefinedTensorImpl
//        as our default constructor is much better for us.
// - Fixed the mismatch between PyTorch and C++ methods
//      - sizes() is now size()
//
// Tensor x = ...;
// Tensor y = x;  // NO COPY



// SUMMING UP
// 1. There will NOT be a retain/release on the Tensor class.  There might be
// some unsafe mechanism to retain/release (because C API bindings need it), but it won't
// be a method with an easy name to spell.
// 2. virtual issue, DELAYED (it's OK for now)
// 3. Undefined tensor... can we make illegal states unrepresentable?
// 4. Because of ArrayRef, we will need to define code style guide (ArrayRef disagrees)

// NB: This is publically inherited, but only to conveniently bring the public methods
// of Retainable into scope.  If this is causing bad error messages, make it private
// again and explicitly 'using' each of the public methods you want to propagate.
class Tensor final : public guts::Retainable<Tensor, guts::TensorImpl, guts::UndefinedTensorImpl> {
  using TensorBase = guts::Retainable<Tensor, guts::TensorImpl, guts::UndefinedTensorImpl>;

public:
  // Normal constructors
  // TODO: I don't know if it's safe to replace this with = default here... godbolt time...
  Tensor()  = default;
  Tensor(const Tensor &rhs) = default;
  Tensor(Tensor &&rhs) noexcept = default;

  // These methods are SO important, they are currently implemented via virtual dispatch
  // via our implementation classes.  Most non-core methods should be implemented by
  // the generic dispatch mechanism.

  // The definitions of these live in TensorMethods.h
  inline int64_t dim() const;
  inline ArrayRef<int64_t> size() const;
  inline ArrayRef<int64_t> stride() const;
  inline void* data_ptr() const;

  inline int64_t Tensor::ndimension() const {
    return dim();
  }

  template<typename T>
  inline T *data() const {
    return static_cast<T *>(data_ptr());
  }

  // The "well known" Tensor functions will call into the dispatch mechanism (yet to be
  // implemented)
};

} // namespace c10