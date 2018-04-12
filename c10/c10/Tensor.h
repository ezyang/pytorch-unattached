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
//        as our default constructor is much better for us. This approach is similar to
//        allowing nullptr dispatch in Obj-C
// - Fixed the mismatch between PyTorch and C++ methods
//      - sizes() is now size()
//
// Tensor x = ...;
// Tensor y = x;  // NO COPY


// Note [Why int64_t?]
// ~~~~~~~~~~~~~~~~~~~
// We need a general purpose numeric type to represent things like sizes, strides
// and other things.  Along the way, there are a lot of hazards which you have to
// watch out for:
//
//    - size_t, the type used by most containers, is UNSIGNED, which means that
//      it is a footgun waiting to happen when you accidentally mix it up with
//      a signed quantity.
//    - int, on 64-bit systems, is still 32-bit, for backwards compatibility
//    - ssize_t is not actually signed on Windows, isn't part of the standard,
//      and only guarantees that -1 is representable
//    - long is still 32-bit on 64-bit Windows systems
//    - ptrdiff_t is hypothetically OK, but it's not an intuitive name
//
// So, we have two choices: (1) we could define our OWN integer type (synonym
// to ptrdiff_t), or (2) we can always use int64_t and eat the performance cost on
// 32-bit systems.  We have chosen (2).
//
// See also http://en.cppreference.com/w/cpp/language/types



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
class Tensor final {
  using TensorBase = guts::Retainable<guts::TensorImpl, guts::UndefinedTensorImpl>;
  TensorBase impl_;

  Tensor(TensorBase impl) : impl_(impl) {};

public:
  static Tensor _fromImpl(guts::TensorImpl* impl) { return Tensor(TensorBase(impl)); };

  // Normal constructors
  // TODO: I don't know if it's safe to replace this with = default here... godbolt time...
  Tensor()  = default;
  Tensor(const Tensor &rhs) = default;
  Tensor(Tensor &&rhs) noexcept = default;

  // These methods are SO important, they are currently implemented via virtual dispatch
  // via our implementation classes.  Most non-core methods should be implemented by
  // the generic dispatch mechanism.

  // The definitions of these live in TensorMethods.h
  // dzhulgakov: nit - is it widely used? I'd prefer ndimension as below or rank. In C2 it's a function returning particular dimension
  inline int64_t dim() const;
  // dzhulgakov: nit - why `size` and not `sizes`? In C2 the size is number of elements - I bet it will cause confusion
  inline ArrayRef<int64_t> size() const;
  inline ArrayRef<int64_t> stride() const;
  inline void* data_ptr() const;
  inline int64_t ndimension() const;

  // dzhulgakov: what are the semantics of it? i.e. how do I change type of the elements stored in a tensor? Or is it passed only in the constructor?
  template<typename T>
  inline T *data() const;

  // The "well known" Tensor functions will call into the dispatch mechanism (yet to be
  // implemented)

  void resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride);

  // To be something like:
  // Tensor add(Tensor x, Tensor y) { guts::dispatch("add", x, y); }

};

} // namespace c10
