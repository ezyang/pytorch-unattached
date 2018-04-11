#include "CPUAllocator.h"

namespace c10 { namespace cpu {

static std::unique_ptr<CPUAllocator> globalCPUAllocator(new SimpleCPUAllocator());

CPUAllocator* getCPUAllocator() {
  return globalCPUAllocator.get();
}

void setCPUAllocator(CPUAllocator* allocator) {
  globalCPUAllocator.reset(allocator);
}

}}