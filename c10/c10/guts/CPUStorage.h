#pragma once

#include "Retainable.h"

#include <cstddef>
#include <memory>

namespace c10 { namespace guts {

class CPUStorage;

// CPUStorage is NOT part of the public API
//
// Every Tensor is backed by a storage; multiple tensors may share the same storage
// (which is why we need an indirection.)
//
// This is implemented as a regular shared_ptr to reduce implementation
// cost.  Since it's internal we don't mind if it's a little clunky to use.
//
// In Caffe2, this was previously implemented directly as a std::shared_ptr.  Doing
// it this means that realloc is not possible because a std::shared_ptr only
// records the free() pointer.
class CPUStorageImpl : public RetainableImpl {
  void* data_;
  // NB: This is not BYTES, this is number of elements!
  std::size_t size_;
  std::size_t element_size_;
  // TODO: pack this boolean flag?
  // Is this storage resizable?  If it comes externally, or has been
  // shared to some external system, it may not be.  Corresponds to
  // TH_STORAGE_RESIZABLE.
  // TODO: Maybe
  bool resizable_;
  // NB: I axed TH_STORAGE_FREEMEM; the expectation is you install a no-op
  // deallocator if this is what you want.
  // NB: I axed TH_STORAGE_VIEW; it makes things complicated for not a good enough
  // reason
  // NB: I axed TH_STORAGE_REFCOUNTED; it seems to always be turned on.  If we do want
  // this, itw ould be a good fit for the Retainable class.

  // The general expectation is that the user of the library has installed some global variable
  // including the allocator you want, so we don't want to splat copies of the allocator everywhere,
  // just indirect to the correct one.
  // TODO: Maybe reconsider this
  Allocator* allocator_;

  friend class CPUStorage;

public:
  ~CPUStorageImpl() override {
    allocator_->free(data_);
  }

  static constexpr CPUStorageImpl* singleton() {
    return nullptr;
  }
};

// This is currently written in a decidedly non-PIMPL-y way.  We won't the benefits of separate
// compilation writing it this way.  If we need it, you need to move these methods around and
// write a little more boilerplate.  The current style is to reduce boilerplate.

class CPUStorage : public Retainable<CPUStorage, CPUStorageImpl, CPUStorageImpl> {
  using StorageBase = Retainable<CPUStorage, CPUStorageImpl, CPUStorageImpl>;

public:
  CPUStorage(const CPUStorage &rhs) = default;
  CPUStorage(CPUStorage &&rhs) noexcept = default;

  CPUStorage(std::size_t element_size, std::size_t size, Allocator* allocator) : StorageBase(new CPUStorageImpl()) {
    auto* impl = get();
    impl->data_ = allocator->malloc(element_size * size);
    impl->element_size_ = element_size;
    impl->size_ = size;
    impl->resizable_ = true;
    impl->allocator_ = allocator;
  }

  void resize() {
    auto* impl = get();
    if (!impl->resizable_) {
      throw std::runtime_error("trying to resize storage that is not resizable");
    }
    // This has one more virtual dispatch than we might have had, if realloc
  }

  // Straight up reimplementation of the ATen CPUStorage API

  template <typename T>
  inline const T* data() const {
    return static_cast<T*>(get()->data_);
  }

  template <typename T>
  inline T* data() {
    return static_cast<T*>(get()->data_);
  }

  inline const void* data_ptr() const {
    return get()->data_;
  }

  inline void* data_ptr() {
    return get()->data_;
  }

  // THStorage_(size)
  inline std::size_t size() const {
    return get()->size_;
  }

  // THStorage_(elementSize) ???
  inline std::size_t elementSize() const {
    return get()->element_size_;
  }




};

}} // namespace c10::guts
