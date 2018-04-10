#pragma once

#include "guts/TensorImpl.h"

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

class Tensor final {
  guts::TensorImpl *pImpl;

  // This is a relatively unsafe constructor which you should avoid using if you
  // don't need it.  The retain parameter specifies whether or not this constructor
  // takes ownership of the passed Impl or not (when retain = true, the caller retains
  // their reference.)
  Tensor(guts::TensorImpl *self)
      : pImpl(self) {
    if (pImpl == nullptr) {
      throw std::runtime_error("Tensor with nullptr not supported");
    }
  }

  guts::TensorImpl *get() const {
    return pImpl;
  }

  guts::TensorImpl *detach() {
    guts::TensorImpl *ret = pImpl;
    pImpl = guts::UndefinedTensorImpl::singleton();
    return ret;
  }

  // smessmer to @ezyang: I think it makes sense to pull out refcounting functionality into
  //          a separate helper class.
  // ezyang to @smessmer: I'm back to wondering, why don't we just have these methods on the impl?
  //          we can make their method names longer...
  // Refcounting kit
  void retain() {
    if (pImpl == guts::UndefinedTensorImpl::singleton()) return;
    ++pImpl->refcount_;
  }

  void release() {
    if (pImpl == guts::UndefinedTensorImpl::singleton()) return;
    if (--pImpl->refcount_ == 0) {
      delete pImpl;
    }
  }

public:
  // Normal constructors
  Tensor() : Tensor(guts::UndefinedTensorImpl::singleton()) {}

  Tensor(const Tensor &rhs)
      : pImpl(rhs.pImpl) {
    if (pImpl != guts::UndefinedTensorImpl::singleton())
      retain();
  }

  Tensor(Tensor &&rhs) noexcept
      : pImpl(rhs.pImpl) {
    rhs.pImpl = guts::UndefinedTensorImpl::singleton();
  }

  // Destructor
  ~Tensor() {
    if (pImpl != guts::UndefinedTensorImpl::singleton())
      release();
  }

  // Copy assignment
  Tensor &operator=(Tensor &&rhs) &noexcept {
    // smessmer to @ezyang: I'd explicitly set rhs to undefined for better debugability.
    // ezyang to @smessmer: That's a bunch of extra refcount bumps though, isn't it?
    rhs.swap(*this);
    return *this;
  }

  Tensor &operator=(Tensor const &rhs) &{
    //TensorBase ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally TensorBase dtor releases rhs.pImpl, which was originally this->pImpl
    Tensor(rhs).swap(*this);
    return *this;
  }

  void reset() {
    Tensor().swap(*this);
  }

  void swap(Tensor &rhs) noexcept {
    guts::TensorImpl *tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  bool defined() const {
    return pImpl != guts::UndefinedTensorImpl::singleton();
  }

  // These methods are SO important, they are currently implemented via virtual dispatch
  // via our implementation classes.  Most non-core methods should be implemented by
  // the generic dispatch mechanism.

  int64_t dim() const {
    return pImpl->dim();
  }

  int64_t ndimension() const {
    return dim();
  }

  ArrayRef<int64_t> size() const {
    return pImpl->size();
  }

  ArrayRef<int64_t> stride() const {
    return pImpl->stride();
  }
  /*
  TypeId type_id() const {
    return pImpl->type_id();
  }
   */
  // smessmer to @ezyang: Do we want to try honoring const-ness for the underlying data?
  //          i.e. const T* data() const {} and T* data() {} ?
  //          not sure if it's a good idea, but we should consider it.
  // ezyang to @smessmer: This is difficult to do without adding more user-visible 'Tensor' types.
  //          Back story is at https://github.com/zdevito/ATen/issues/27
  template<typename T>
  T *data() const {
    return static_cast<T *>(pImpl->data_ptr());
  }


  // TODO: work out the type() situation

  // TODO: work out the guts() situation

  // The "well known" Tensor functions will call into the dispatch mechanism (yet to be
  // implemented)
};

} // namespace c10
