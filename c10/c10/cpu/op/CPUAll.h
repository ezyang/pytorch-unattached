#pragma once

#include <c10/ArrayRef.h>
#include <c10/guts/TypeId.h>

#include <cstdint>

namespace c10 {
  class Tensor;
}

// TODO: It's not clear if we're going to have a separate file per operator,
// multiple operators in a file (and thus some organization scheme from grouping them.)
// Up until then, we're dumping all the operators in this file.

namespace c10 { namespace cpu { namespace op {

// Factory functions
Tensor empty(ArrayRef<int64_t> sizes, TypeMeta dtype);
Tensor zeros(ArrayRef<int64_t> sizes, TypeMeta dtype);
void zero_(const Tensor& self);
Tensor tensor(const void* data, ArrayRef<int64_t> sizes, TypeMeta dtype);

// Caffe2's in-place operations
void copy_(const Tensor& self, TypeMeta dtype, const void* p, int64_t size_bytes);
void legacy_pytorch_resize_(const Tensor &self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride);
void legacy_caffe2_resize_(const Tensor &self, ArrayRef<int64_t> new_size);
void reserve_(const Tensor& self, ArrayRef<int64_t> new_size);
void extend_(const Tensor& self, int64_t num, double growthPct);

// Minimal operations we needed for testing
bool equal(Tensor self, Tensor other);

}}} // namespace c10::cpu::op
