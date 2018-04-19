#pragma once

#include <c10/ArrayRef.h>
#include <c10/DataType.h>

#include <cstdint>

namespace c10 {
  class Tensor;
}

// TODO: It's not clear if we're going to have a separate file per operator,
// multiple operators in a file (and thus some organization scheme from grouping them.)
// Up until then, we're dumping all the operators in this file.

namespace c10 { namespace cpu { namespace op {

// Factory functions
Tensor empty(ArrayRef<int64_t> sizes, DataType dtype);
Tensor zeros(ArrayRef<int64_t> sizes, DataType dtype);
void zero_(const Tensor& self);
Tensor tensor(void* data, ArrayRef<int64_t> sizes, DataType dtype);

// Caffe2's in-place operations
void copy_(const Tensor& self, DataType dtype, const void* p, int64_t size_bytes);
void resize_(const Tensor& self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride, bool keep_data);
void reserve_(const Tensor& self, ArrayRef<int64_t> new_size);
void extend_(const Tensor& self, int64_t num, double growthPct);

// Minimal operations we needed for testing
bool equal(const Tensor& self, const Tensor& other);

}}} // namespace c10::cpu::op
