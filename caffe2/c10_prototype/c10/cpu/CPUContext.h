#pragma once

#include "caffe2/c10_prototype/c10/cpu/CPUAllocator.h"
#include "caffe2/utils/Optional.h"

namespace c10 { namespace cpu {

// TODO: a lot of indirections here...
// For example, to get at the allocator in the current design, we have to:
//    Pull the cache line for CPUContext object
//    Pull the cache line for CPUAllocator object
//    Pull the cache line for CPUAllocator's vtable
//    Pull the instructions for the allocation function
// If CPUAllocator were a struct of functions we could embed it directly
// into CPUContext, eliminating one cache line pull

/**
 * A set of global state associated with the CPU backend.  These settings are global to
 * all threads.
 *
 * Some discussion can be found at https://fb.quip.com/w9G9AlEXbPlq
 */
class CPUContext final {
  CPUAllocator* allocator_;

  // This comes from Caffe2; it corresponds to FLAGS_caffe2_keep_on_shrink
  // and FLAGS_caffe2_max_keep_on_shrink_memory
  // TODO: Put in global context, probably
  optional<int64_t> max_keep_on_shrink_bytes_ = nullopt;

public:
  CPUContext() : allocator_(getSimpleCPUAllocator()) {}

  /**
   * The CPU allocator, from which you can `malloc()` data.
   */
  CPUAllocator *getCPUAllocator() { return allocator_; }
  void setCPUAllocator(CPUAllocator* allocator) { allocator_ = allocator; }

  /**
   * Controls reallocation behavior when a memory resize shrinks a tensor.
   *
   * If `nullopt`, resizes which reduce the size of a tensor never reallocate
   * the tensor.  Otherwise, a reallocation occurs if the amount of memory that
   * would be freed is greater than the recorded `maxKeepOnShrinkBytes`.
   */
  optional<int64_t> maxKeepOnShrinkBytes() { return max_keep_on_shrink_bytes_; }
  void setMaxKeepOnShrinkBytes(optional<int64_t> s) { max_keep_on_shrink_bytes_ = s; }
};

CPUContext &globalCPUContext();

}}
