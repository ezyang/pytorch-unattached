#pragma once

#include "c10/Tensor.h"

// TODO: It's not clear if we're going to have a separate file per operator,
// multiple operators in a file (and thus some organization scheme from grouping them.)
// Up until then, we're dumping all the operators in this file.

namespace c10 { namespace cpu { namespace op {

/// c10::tensor
Tensor tensor(DataType dtype);
void copy_(const Tensor& self, DataType dtype, const void* p, int64_t size_bytes);
void resize_(const Tensor& self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride, bool keep_data);
void reserve_(const Tensor& self, ArrayRef<int64_t> new_size);
void extend_(const Tensor& self, int64_t num, double growthPct);

}}} // namespace c10::cpu::op
