#pragma once

#include <functional>

namespace c10 { namespace cpu {

/**
 * An interface CPU memory allocators.
 */
struct CPUAllocator {
  // TODO: Stop using std::function here
  using malloc_ret_t = std::pair<void*, std::function<void(void*)>>;

  CPUAllocator() = default;
  // In case the allocator wants to manage some internal state that needs
  // to be freed later
  virtual ~CPUAllocator() noexcept = default;

  /**
   * Allocate `size_bytes` bytes of CPU memory, and return a pointer to this memory
   * and a function pointer that, when invoked with this pointer, deallocates
   * this memory.
   *
   * @param size_bytes Size in bytes to allocate
   *
   * @note It would be safer to return a unique_ptr directly, as it would ensure
   * call-sites always remember to dispose of the pointer.  Why don't we?
   * There is no way to move a deleter out of a unique pointer (we can make
   * a copy via `get_deleter()` but that, in general, would entail a copy of
   * any state associated with the deleter.)  And we need to do this when
   * we implement placement-delete as a layer on top of the allocator.
   * Pair it is.  (We copied this pattern from Caffe2)
   */
  virtual malloc_ret_t malloc(int64_t size_bytes) = 0;

  // NB: Dropped getDeleter() for now

  // TODO: consider adding back realloc()
};

/**
 * A simple CPU memory allocator that uses `malloc` and `free`.
 */
struct SimpleCPUAllocator : public CPUAllocator {
  malloc_ret_t malloc(int64_t size_bytes) override {
    return malloc_ret_t(std::malloc(static_cast<size_t>(size_bytes)), [](void* p) { std::free(p); });
  }
};

/**
 * Retrieve the global instance of the simple CPU allocator.
 */
CPUAllocator* getSimpleCPUAllocator();

}}