#pragma once

#include <atomic>
#include <stdexcept>

namespace c10 { namespace guts {

/**
 * Base class for intrusive refcounting.
 */
class IntrusivePtrTarget {
  std::atomic<int> refcount_;

  template <typename T, typename NullType>
  friend class IntrusivePtr;

public:
  virtual ~IntrusivePtrTarget() = default;

protected:
  IntrusivePtrTarget() : refcount_(1) {}
};

template <typename TTarget, typename NullType>
class IntrusivePtr final {
  TTarget *ptr_;

  void retain() noexcept {
    if (ptr_ == NullType::singleton()) return;
    ++ptr_->refcount_;
  }

  void release() noexcept {
    if (ptr_ == NullType::singleton()) return;
    if (--ptr_->refcount_ == 0) {
      delete ptr_;
      ptr_ = NullType::singleton();
    }
  }

public:

  IntrusivePtr() : ptr_(NullType::singleton()) {}

  // NB: invariant: if self == nullptr, then nullptr == NullType::singleton()
  IntrusivePtr(TTarget *self) : ptr_(self) {}

  IntrusivePtr(IntrusivePtr &&rhs) noexcept
      : ptr_(rhs.ptr_) {
    rhs.ptr_ = NullType::singleton();
  }

  IntrusivePtr(const IntrusivePtr &rhs)
      : ptr_(rhs.ptr_) {
    retain();
  }

  ~IntrusivePtr() {
    release();
  }

  TTarget *get() const noexcept {
    return ptr_;
  }

  TTarget* operator*() const noexcept {
    return ptr_;
  }

  TTarget* operator->() const noexcept {
    return **this;
  }

  // Copy assignment
  IntrusivePtr &operator=(IntrusivePtr &&rhs) &noexcept {
    // smessmer to @ezyang: I'd explicitly set rhs to undefined for better debugability.
    // ezyang to @smessmer: That's a bunch of extra refcount bumps though, isn't it?
    // smessmer to @ezyang: Only if *this is valid. And in that case, you probably want these bumps because you want to free the memory held by *this. I added it.
    release();
    rhs.swap(*this);
    return *this;
  }

  IntrusivePtr &operator=(const IntrusivePtr &rhs) &noexcept {
    //TensorBase ctor retains original rhs.ptr_
    //then rhs.ptr_ is swapped with this->ptr_
    //finally TensorBase dtor releases rhs.ptr_, which was originally this->ptr_
    IntrusivePtr(rhs).swap(*this);
    return *this;
  }

  void reset() noexcept {
    IntrusivePtr().swap(*this);
  }

  void swap(IntrusivePtr &rhs) noexcept {
    TTarget *tmp = ptr_;
    ptr_ = rhs.ptr_;
    rhs.ptr_ = tmp;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  bool defined() const noexcept {
    return ptr_ != NullType::singleton();
  }
};

}} // namespace c10::guts
