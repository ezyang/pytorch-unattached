#pragma once

// Base for intrusive refcounting
class Retainable {
  // Refcounting
  std::atomic<int> refcount_;
};


