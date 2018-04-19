#pragma once

#include <atomic>
#include <stdexcept>

namespace c10 { namespace guts {

/**
 * Base class for intrusive refcounting.
 */
class RetainableImpl {
  std::atomic<int> refcount_;

  template <typename T, typename NullType>
  friend class Retainable;

public:
  virtual ~RetainableImpl() = default;

protected:
  RetainableImpl() : refcount_(1) {}
};

template <typename TImpl, typename NullType>
class Retainable final {
  TImpl *pImpl;

  void retain() noexcept {
    if (pImpl == NullType::singleton()) return;
    ++pImpl->refcount_;
  }

  void release() noexcept {
    if (pImpl == NullType::singleton()) return;
    if (--pImpl->refcount_ == 0) {
      delete pImpl;
      pImpl = NullType::singleton();
    }
  }

public:

  Retainable() : pImpl(NullType::singleton()) {}

  // NB: invariant: if self == nullptr, then nullptr == NullType::singleton()
  Retainable(TImpl *self) : pImpl(self) {}

  Retainable(Retainable &&rhs) noexcept
      : pImpl(rhs.pImpl) {
    rhs.pImpl = NullType::singleton();
  }

  Retainable(const Retainable &rhs)
      : pImpl(rhs.pImpl) {
    retain();
  }

  ~Retainable() {
    release();
  }

  TImpl *get() const noexcept {
    return pImpl;
  }

  TImpl* operator*() const noexcept {
    return pImpl;
  }

  TImpl* operator->() const noexcept {
    return **this;
  }

  // Copy assignment
  Retainable &operator=(Retainable &&rhs) &noexcept {
    // smessmer to @ezyang: I'd explicitly set rhs to undefined for better debugability.
    // ezyang to @smessmer: That's a bunch of extra refcount bumps though, isn't it?
    // smessmer to @ezyang: Only if *this is valid. And in that case, you probably want these bumps because you want to free the memory held by *this. I added it.
    release();
    rhs.swap(*this);
    return *this;
  }

  Retainable &operator=(const Retainable &rhs) &noexcept {
    //TensorBase ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally TensorBase dtor releases rhs.pImpl, which was originally this->pImpl
    Retainable(rhs).swap(*this);
    return *this;
  }

  void reset() noexcept {
    Retainable().swap(*this);
  }

  void swap(Retainable &rhs) noexcept {
    TImpl *tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  bool defined() const noexcept {
    return pImpl != NullType::singleton();
  }
};

}} // namespace c10::guts
