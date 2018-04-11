#pragma once

#include <functional>

struct CPUAllocator {
  CPUAllocator() {}
  // In case the allocator wants to manage some internal state that needs
  // to be freed later
  virtual ~CPUAllocator() noexcept {}
  virtual std::unique_ptr<void, std::function<void(void*)>> malloc() {}
  // virtual void realloc() {}
};

