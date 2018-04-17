#include <c10/c10.h>

#include "All.h"

namespace c10 { namespace op {

Tensor tensor(DataType dtype, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  auto r = c10::tensor(dtype);
  r.resize_(size, stride);
  return r;
}

void shrink_(const Tensor& self, int64_t outer_dim_new_size) {
  self._to_impl()->_shrink(outer_dim_new_size);
}

}} // namespace c10::op
