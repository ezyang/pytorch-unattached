#pragma once

#include <functional>

namespace c10 { namespace cpu {

struct CPUAllocator {
  using data_t = std::unique_ptr<void, std::function<void(void*)>>;

  CPUAllocator() {}
  // In case the allocator wants to manage some internal state that needs
  // to be freed later
  virtual ~CPUAllocator() noexcept {}
  virtual data_t malloc(std::size_t) = 0;
  // NB: Dropped getDeleter() for now
};

struct SimpleCPUAllocator : public CPUAllocator {
  data_t malloc(std::size_t size) override {
    return data_t(std::malloc(size), [](void* p) { std::free(p); });
  }
};

// Now that I've written these functions, I remembered that Zach suggested that
// we should use a context stored in our objects which will let us avoid an indirection

CPUAllocator* getCPUAllocator();
void setCPUAllocator(CPUAllocator* allocator);

}}