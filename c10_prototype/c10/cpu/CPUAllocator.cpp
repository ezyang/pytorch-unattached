#include <c10/cpu/CPUAllocator.h>

namespace c10 { namespace cpu {

CPUAllocator* getSimpleCPUAllocator() {
  static SimpleCPUAllocator allocator;
  return &allocator;
}

}}