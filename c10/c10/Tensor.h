#pragma once

#include "guts/Retainable.h"
#include "guts/TensorImpl.h"

#include "ArrayRef.h"
#include "DataType.h"
#include "Utils.h"

namespace c10 {

// Design notes:
//  - Manual retain/release instead of shared_ptr. Reasons:
//      - PRIMARY: It's possible to work with the underlying retained object using
//        a C API, which is basically impossible to do with shared_ptr because
//        it doesn't expose a manual retain()/release() API
//      - SECONDARY: A true intrusive reference count avoids the need to store
//        a weak pointer to the control block (as is the case for
//        std::enabled_shared_from_this).
// - guts::UndefinedTensorImpl instead of null pointer. Reasons:
//      - We originally had a null pointer in ATen, but this meant that when we
//        incorrectly attempted to use such a null pointer, we would segfault and
//        crash, which is very unfriendly for our Python users.  Using an guts::UndefinedTensorImpl
//        as our default constructor is much better for us. This approach is similar to
//        allowing nullptr dispatch in Obj-C
// - Fixed the mismatch between PyTorch and C++ methods
//      - sizes() is now sizes()
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
//    - ptrdiff_t is allowed to have 2**15-1 as its max value in C
//
// So, we have two choices: (1) we could define our OWN integer type (typedef'ed
// to be a sane thing on all platforms), or (2) we can always use int64_t and eat
// the performance cost on 32-bit systems.  We have chosen (2).
//
// See also http://en.cppreference.com/w/cpp/language/types


// Note [Cult of the dot]
// ~~~~~~~~~~~~~~~~~~~~~
// In Python, method invocation is very simple: you write x.f()
// We wish to preserve this simplicity in C++.  To achieve this, most of our
// classes are implemented in the PIMPL pattern (there is an implementation class,
// TensorImpl, which actually contains the data and implementations of functions,
// and a wrapper class Tensor, which is just a pointer to TensorImpl), where the
// wrapper class is written to act as a pass-by-value pointer, with direct methods
// which forward to the implementation.
//
// There are a few downsides to this strategy, which we enumerate here:
//
//   - It's difficult to do const-correctness in this regime, because doing so
//     correctly requires *two* wrapper classes for the const and non-const
//     version (const Tensor doesn't cut the mustard, because it says that the
//     pointer is const, not that we have a (non-const) pointer to const data.)
//     We have opted not introduce another Tensor type, but the meaning of
//     const Tensor& is perpetually confusing to C++ experts who attempt to
//     use our Tensor type.)
//
//   - Static members that used to be pointers don't work correctly.  In particular,
//     you can't do something like this:
//
//        class Tensor {
//          static const Tensor EMPTY = {...};
//        }
//
//     At the time C++ is laying out the static member, Tensor is an incomplete type,
//     so the static Tensor EMPTY declaration is illegal.  The workaround we employ
//     in this codebase is to first predeclare a class that has all of the interesting
//     members, and then inherit from it the actual class name that now has the static
//     members (and we can implicitly convert from the pre-class.)  Not the prettiest,
//     but it gets us the desired API.


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


/**
 * Tensor is a "generic" object holding a pointer to guts::TensorImpl, managing
 * reference counts for it.  In this way, it is similar to boost::intrusive_ptr.
 *
 * For example:
 *
 *      void func(Tensor a) {
 *          Tensor b = a;
 *          ...
 *      }
 *
 * In this example, when we say Tensor b = a, we are creating a new object that points to the
 * same underlying guts::TensorImpl, and bumps its reference count. When b goes out of scope, the
 * destructor decrements the reference count by calling release() on the guts::TensorImpl it points to.
 * The existing constructors, operator overloads, etc. take care to implement the correct semantics.
 * You are not expected to ever interact directly with guts::TensorImpl.
 */
class Tensor final {
  using TensorBase = guts::Retainable<guts::TensorImpl, guts::UndefinedTensorImpl>;
  TensorBase impl_;

  Tensor(TensorBase impl) : impl_(impl) {};

public:
  // Steals the reference.  (In old ATen, there was an optional retain which also bumped
  // the refcount while you were at it.)
  // TODO: Figure out a safer way to expose this to relevant sites
  // I've forgotten how to use this safely, so it's
  // not a good API. :)
  static Tensor _from_impl(guts::TensorImpl *impl) { return Tensor(TensorBase(impl)); };

  // TODO: Figure out a safer way to expose this to relevant sites
  guts::TensorImpl* _to_impl() const { return impl_.get(); }

  // Normal constructors
  Tensor() = default;
  Tensor(const Tensor &rhs) = default;
  Tensor(Tensor &&rhs) noexcept = default;
  Tensor& operator=(const Tensor &rhs) = default;
  Tensor& operator=(Tensor &&rhs) = default;

  // These methods are SO important, they are currently implemented via virtual dispatch
  // via our implementation classes.  Most non-core methods should be implemented by
  // the generic dispatch mechanism.

  // dzhulgakov: nit - is it widely used? I'd prefer ndimension as below or rank. In C2 it's a function returning particular dimension
  int64_t dim() const {
    return impl_->dim();
  }

  // NB: PyTorch Python API calls this size(), but this is confusing in C++ for two reasons:
  //    - C++ conventionally has size() meaning "number of elements"
  //    - Caffe2 has size() meaning "number of elements"
  // We could also introduce shape() as a way to get at this name
  ArrayRef<int64_t> sizes() const {
    return impl_->sizes();
  }

  ArrayRef<int64_t> strides() const {
    return impl_->strides();
  }

  // TODO: This is inconsistent with ATen naming, which uses the overload sizes(int64_t) and strides(int64_t)

  int64_t size(int64_t dim) const {
    return impl_->sizes().at(static_cast<size_t>(dim));
  }

  int64_t stride(int64_t dim) const {
    return impl_->strides().at(static_cast<size_t>(dim));
  }

  // smessmer to @ezyang: Do we want to try honoring const-ness for the underlying data?
  //          i.e. const T* data() const {} and T* data() {} ?
  //          not sure if it's a good idea, but we should consider it.
  // ezyang to @smessmer: This is difficult to do without adding more user-visible 'Tensor' types.
  //          Back story is at https://github.com/zdevito/ATen/issues/27
  void *data_ptr() const {
    return impl_->data_ptr();
  }

  int64_t storage_offset() const {
    return impl_->storage_offset();
  }

  int64_t numel() const {
    return impl_->numel();
  }

  int64_t ndimension() const {
    return dim();
  }

  DataType dtype() const {
    return impl_->dtype();
  }

  TypeId type_id() const {
    return impl_->type_id();
  }

  /**
   * Is the tensor contiguous?
   *
   * Contiguous memory layout is often referred to as "C order."
   *
   * @note It is easy to compute a canonical set of contiguous strides
   * given some tensor size, but it is not always the case that this
   * equation holds:
   *
   *        x.strides() == contiguous_strides(x.sizes())
   *
   * This is because tensors with dimensions of zero or one size have
   * unlimited degrees of freedom in strides while maintaining contiguity
   * (since the stride "never matters" in this case.)
   *
   * @return true if the tensor is contiguous, false otherwise
   */
  bool is_contiguous() const {
    return impl_->is_contiguous();
  }

  /**
   * Set this tensor to share storage with another tensor, at a given offset, size and stride.
   *
   * @param src Tensor to share storage with.  Must be contiguous and have same dtype and type_id.
   * @param storage_offset
   * @param size
   * @param stride
   */
  void set_(const Tensor& src, int64_t storage_offset, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) const {
    // TODO: TensorImpl*? Really??
    return impl_->_set(src._to_impl(), storage_offset, size, stride);
  }

  // dzhulgakov: what are the semantics of it? i.e. how do I change type of the elements stored in a tensor? Or is it passed only in the constructor?
  // ezyang: invocation of data() is only well-defined if the type T matches the internal type T of the tensor.
  // This function has nothing to do with casting.
  template<typename T>
  inline T *data() const {
    // dzhulgakov: also, if tensor doesn't support raw pointers - is it expected to throw?
    // ezyang: yes.  Not implemented yet.
    // clion hates me (scalar_type is ambiguous)
    C10_ASSERT(c10::dtype<T>() == impl_->dtype(),
               "data: requested dtype ", c10::dtype<T>(),
               " via template parameter does not match dtype of tensor ", impl_->dtype());
    return static_cast<T *>(data_ptr());
  }

  // The "well known" Tensor functions will call into the dispatch mechanism (yet to be
  // implemented)

  // NB: Const is a lie!  See also Note [Cult of the dot]

  // TODO: this is a sharp-edged API, which we are planning to replace
  void legacy_pytorch_resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) const;
  void legacy_pytorch_resize_(ArrayRef<int64_t> size) const {
    legacy_pytorch_resize_(size, contiguous_strides(size));
  }
  void legacy_pytorch_resize_as_(const Tensor& other) const;

  void legacy_caffe2_resize_(ArrayRef<int64_t> size) const;

  void zero_() const;

  // Hmmmmm, does the void* violate our dispatch data model?  OTOH, we are probably going to
  // need ways to create tensors from void* pointers
  /**
   * Copy the contents of a pointer into this tensor.
   *
   * @warning Prefer using the templated `copy_` instead, which is more type-safe.
   *
   * @param dtype       The type of the elements to be copied
   * @param p           Pointer to the elements to be copied
   * @param size_bytes  The size in bytes to copy
   */
  void copy_(DataType dtype, const void* p, int64_t size_bytes) const;

  // NB: This is an instance of the design pattern, where we cannot (and will not) dispatch
  // templated functions.  So this has an inline definition which goes straight to the
  // actual implementation which is dynamically dispatched.
  /**
   * Copy the contents of an ArrayRef into this tensor.
   *
   * @tparam T      The type of the elements to be copied
   * @param arr     The array of elements to be copied
   */
  template <typename T>
  void copy_(ArrayRef<T> arr) const {
    copy_(c10::dtype<T>(), arr.data(), arr.size() * sizeof(T));
  }

  /**
   * @brief Extends the outer-most dimension of this tensor by `num` elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * `growthPct`. This ensures that extend runs on an amortized O(1) time
   * complexity.
   *
   * @warning growthPct is denominated in percent points (so 20 is 20%)
   *
   * @note Corresponds to Caffe2 `Tensor::Extend`, but `growthPct` is a double
   * rather than a float, following ATen's conventions.
   *
   * @todo Possibly give growthPct a default argument?
   *
   * @param num Size to extend the outer-most dimension by
   * @param growthPct Minimum growth rate, to ensure amortized execution
   */
  void extend_(int64_t num, double growthPct) const;

  /**
   * Reserve enough space to hold a tensor of `new_size` in the underlying storage
   *
   * If the current reserved space (in storage) is insufficient, a reallocation
   * will occur.  This operation no-ops if the reservation request is equal to
   * or smaller than the current reserved space.
   *
   * @note Corresponds to Caffe2 `Tensor::Reserve`
   *
   * @param new_size Size to reserve enough space to hold.
   */
  void reserve_(ArrayRef<int64_t> new_size) const;

  void shrink_(int64_t outer_dim_new_size) const;

  void view_(ArrayRef<int64_t> size) const;

  void clone() const;

  bool equal(const Tensor& other) const;

  // To be something like:
  // Tensor add(Tensor x, Tensor y) { guts::dispatch("add", x, y); }

};

} // namespace c10
