#include <c10/c10.h>

#include "All.h"

namespace c10 { namespace ops {

Tensor tensor(DataType dtype, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  auto r = c10::tensor(dtype);
  r.resize_(size, stride);
  return r;
}

}} // namespace c10::op
