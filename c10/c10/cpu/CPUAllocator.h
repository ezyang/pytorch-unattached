#pragma once

#include <functional>

namespace c10 { namespace cpu {

struct CPUAllocator {
  using malloc_ret_t = std::pair<void*, std::function<void(void*)>>;

  CPUAllocator() = default;
  // In case the allocator wants to manage some internal state that needs
  // to be freed later
  virtual ~CPUAllocator() noexcept = default;

  // NB: It would be safer to return a unique_ptr directly.  Why don't we?
  // There is no way to move a deleter out of a unique pointer (we can make
  // a copy via get_deleter() but that, in general, would entail a copy of
  // any state associated with the deleter.)  And we need to do this when
  // we implement placement-delete as a layer on top of the allocator.
  // Pair it is.  (Caffe2 already does this correctly.)
  virtual malloc_ret_t malloc(int64_t) = 0;
  // NB: Dropped getDeleter() for now
};

struct SimpleCPUAllocator : public CPUAllocator {
  malloc_ret_t malloc(int64_t size_bytes) override {
    return malloc_ret_t(std::malloc(static_cast<size_t>(size_bytes)), [](void* p) { std::free(p); });
  }
};

}}