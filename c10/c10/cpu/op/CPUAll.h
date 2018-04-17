#pragma once

#include "c10/Tensor.h"

namespace c10 { namespace cpu { namespace op {

Tensor tensor(DataType dtype);
void copy_(const Tensor& self, DataType dtype, const void* p, int64_t size_bytes);
void resize_(const Tensor& self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride, bool keep_data);

}}} // namespace c10::cpu::op
