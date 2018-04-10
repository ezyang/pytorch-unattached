#pragma once

#include "Retainable.h"

#include <cstddef>
#include <memory>

namespace c10 { namespace guts {

// Use cases of allocators:
//  - I'm inside an operator, I need to allocate some helper buffer or
//    similar for my operation.
//  - Entirely inside the backend, its hidden away, the CPU tensor needs
//    a way to call malloc/GPU needs a way to call GPU malloc (which
//    could be a completely different implementation

// Some design constraints:
//  - We want a single "allocator" struct which contains all of the information for
//    malloc, realloc and free

// Would be more accurate to call this CPUAllocator/CPUStorage
// CPU is easy; GPU you want to know what device you're allocating
// memory on.  Not enough to know that it's GPU memory.  There may
// be trickier versions.

// Corresponds to THAllocator, but in C++ style.
class Allocator {
public:
  // Tentatively malloc/realloc are not needed
  virtual void* malloc(std::size_t) = 0;
  virtual void* realloc(void* p, std::size_t s) {
    void* np = malloc(s);
    // memcpy();
    return nullptr;
  };
  virtual void* free(void*) = 0;
};

// TODO: Need a default allocator...

class Storage;

// Storage is NOT part of the public API
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
class StorageImpl : public RetainableImpl {
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

  friend class Storage;

public:
  ~StorageImpl() override {
    allocator_->free(data_);
  }

  static constexpr StorageImpl* singleton() {
    return nullptr;
  }
};

// This is currently written in a decidedly non-PIMPL-y way.  We won't the benefits of separate
// compilation writing it this way.  If we need it, you need to move these methods around and
// write a little more boilerplate.  The current style is to reduce boilerplate.

class Storage : public Retainable<Storage, StorageImpl, StorageImpl> {
  using StorageBase = Retainable<Storage, StorageImpl, StorageImpl>;

public:
  Storage(const Storage &rhs) = default;
  Storage(Storage &&rhs) noexcept = default;

  Storage(std::size_t element_size, std::size_t size, Allocator* allocator) : StorageBase(new StorageImpl()) {
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

  // Straight up reimplementation of the ATen Storage API

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
