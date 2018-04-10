#pragma once

#include <cstddef>

namespace c10 { namespace guts {

// Storage is NOT part of the public API
class Storage {
private:
  void* data_;
  ptrdiff_t size_;
  // TODO: pack these boolean flags
  // Is this storage resizable?  If it comes externally, or has been
  // shared to some external system, it may not be.
  bool resizable_;
  // Should we free
  bool
};

}} // namespace c10::guts
