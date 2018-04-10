#pragma once

#include <atomic>

namespace c10 { namespace guts {

// Base for intrusive refcounting
class Retainable {
  std::atomic<int> refcount_;

  friend class c10::Tensor;
};

}} // namespace c10::guts
