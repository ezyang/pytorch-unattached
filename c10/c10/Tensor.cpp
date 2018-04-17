#include "Tensor.h"

// NB: This implementation file is strictly temporary; in the real implementation we can
// inline these in Tensor.h because they won't refer to actual implementations (breaking
// the circular dependency)

// TODO: Strictly temporary: these polymorphic functions should still go through the dispatcher
#include "c10/op/All.h"

// TODO: Strictly temporary, hardcoded CPU
#include "c10/cpu/op/CPUAll.h"

namespace c10 {

void Tensor::resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride, bool keep_data) const {
  // TODO: Use the dynamic dispatcher instead
  cpu::op::resize_(*this, size, stride, keep_data);
}

void Tensor::copy_(DataType dtype, const void* p, int64_t size_bytes) const {
  // TODO: Use the dynamic dispatcher instead
  cpu::op::copy_(*this, dtype, p, size_bytes);
}

void Tensor::extend_(int64_t num, double growthPct) const {
  // TODO: Use the dynamic dispatcher instead
  cpu::op::extend_(*this, num, growthPct);
}

void Tensor::reserve_(ArrayRef<int64_t> new_size) const {
  // TODO: Use the dynamic dispatcher instead
  cpu::op::reserve_(*this, new_size);
}

void Tensor::shrink_(int64_t outer_dim_new_size) const {
  op::shrink_(*this, outer_dim_new_size);
}

void Tensor::resize_as_(c10::Tensor other) const {
  op::resize_as_(*this, other);
}

void Tensor::view_(c10::ArrayRef<int64_t> size) const {
  op::view_(*this, size);
}

}
