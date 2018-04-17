#pragma once

#include "CPUAllocator.h"

namespace c10 { namespace cpu {

// TODO: a lot of indirections here...
// For example, to get at the allocator in the current design, we have to:
//    Pull the cache line for CPUContext object
//    Pull the cache line for CPUAllocator object
//    Pull the cache line for CPUAllocator's vtable
//    Pull the instructions for the allocation function
// If CPUAllocator were a struct of functions we could embed it directly
// into CPUContext, eliminating one cache line pull

class CPUContext final {
  CPUAllocator* allocator_;
  // TODO: condense booleans into a bitmask
  // These come from Caffe2; correspond to FLAGS_caffe2_keep_on_shrink
  // and FLAGS_caffe2_max_keep_on_shrink_memory
  bool keep_on_shrink_;
  int64_t max_keep_on_shrink_bytes_;
public:
  CPUContext() : allocator_(getSimpleCPUAllocator()) {}
  CPUAllocator *getCPUAllocator() { return allocator_; }
  void setCPUAllocator(CPUAllocator* allocator) { allocator_ = allocator; }
  bool keepOnShrink() { return keep_on_shrink_; }
  void setKeepOnShrink(bool b) { keep_on_shrink_ = b; }
  size_t maxKeepOnShrinkBytes() { return max_keep_on_shrink_bytes_; }
  void setMaxKeepOnShrinkBytes(int64_t s) { max_keep_on_shrink_bytes_ = s; }
};

CPUContext &globalCPUContext();

}}
