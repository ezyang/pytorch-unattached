#pragma once

#include <c10/DimVector.h>
#include <c10/ArrayRef.h>
#include "caffe2/core/typeid.h"

#include <numeric>
#include <cinttypes>

// Sin bin to dump functions which we don't have good places to put yet

namespace c10 {

// TODO: These shouldn't actually be inline; the inline is just here to appease the linker

// Return the strides corresponding to a contiguous layout of sizes.
inline DimVector contiguous_strides(ArrayRef<int64_t> size) {
  DimVector v(size.size());
  int64_t total_size = 1;
  for (int64_t d = static_cast<int64_t>(size.size()) - 1; d >= 0; d--) {
    v[static_cast<size_t>(d)] = total_size;
    total_size *= size[static_cast<size_t>(d)];
  }
  return v;  // RVO
}

// Given sizes and strides, return the lowest and highest indices (inclusive-exclusive) which may
// be accessed with them.
// TODO: Refactor this into a utility header file
inline std::pair<int64_t, int64_t> compute_extent(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  // Watermarks are inclusive.  NB: watermarks can be negative! Careful!
  // NB: when sizes is empty, we get {0, 1}; that is to say,
  // there is ONE valid location.  This is correct!
  int64_t low_watermark = 0; // inclusive
  int64_t high_watermark = 1; // exclusive
  for (int64_t d = static_cast<int64_t>(size.size()) - 1; d >= 0; d--) {
    // TODO: This special case is so irritating.  But if we don't apply it,
    // this function returns {0, 1} when you pass it sizes {0} strides {0}.
    if (size[static_cast<size_t>(d)] == 0) return {0, 0};
    C10_ASSERT(size[static_cast<size_t>(d)] > 0, "size = ", size);
    if (stride[static_cast<size_t>(d)] >= 0) {
      high_watermark += (size[static_cast<size_t>(d)] - 1) * stride[static_cast<size_t>(d)];
    } else {
      low_watermark += (size[static_cast<size_t>(d)] - 1) * stride[static_cast<size_t>(d)];
    }
  }
  return {low_watermark, high_watermark};
}

inline int64_t required_new_storage_size(
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    int64_t storage_offset) {
  int64_t low_watermark, high_watermark;
  std::tie(low_watermark, high_watermark) = compute_extent(size, stride);
  if (low_watermark + storage_offset < 0) {
    C10_CHECK(0, "The given size ", size, " and stride ", stride, " can result in a negative index ",
    "as large as ", low_watermark, ", but this could result in an underflow as your storage ",
    "begins at element ", storage_offset);
  }
  return high_watermark + storage_offset;
}

inline int64_t product(ArrayRef<int64_t> xs) {
  return std::accumulate(xs.begin(), xs.end(), 1, std::multiplies<int64_t>());
}

}
