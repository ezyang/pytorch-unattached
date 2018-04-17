#pragma once

#include "c10/Tensor.h"

namespace c10 { namespace cpu { namespace op {

Tensor tensor(DataType dtype);
void copy_(const Tensor& self, DataType dtype, const void* p, int64_t size_bytes);

}}} // namespace c10::cpu::op
