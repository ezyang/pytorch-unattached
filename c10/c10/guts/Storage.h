#pragma once

#include "Retainable.h"

#include <cstddef>
#include <memory>

namespace c10 { namespace guts {

// Some design constraints:
//  - We want a single "allocator" struct which contains all of the information for
//    malloc, realloc and free

// Corresponds to THAllocator
struct Allocator {
  void* (*malloc)(void*, ptrdiff_t);
  void* (*realloc)(void*, void*, ptrdiff_t);
  void* (*free)(void*, void*);
};


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
  ptrdiff_t size_;
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

  // The general expectation is that the user of the library has installed some global variable
  // including the allocator you want, so we don't want to splat copies of the allocator everywhere,
  // just indirect to the correct one.
  // TODO: Maybe reconsider this
  Allocator* allocator_;
  void* allocator_context_;

public:
  static constexpr StorageImpl* singleton() {
    return nullptr;
  }
};

class Storage : public Retainable<Storage, StorageImpl, StorageImpl> {
  using StorageBase = Retainable<Storage, StorageImpl, StorageImpl>;

public:
  Storage() : StorageBase() {}
  Storage(const Storage &rhs) : StorageBase(rhs) {}
  Storage(Storage &&rhs) noexcept : StorageBase(std::move(rhs)) {}
};

}} // namespace c10::guts
