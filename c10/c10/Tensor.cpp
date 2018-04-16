#include "Tensor.h"

// STRICTLY TEMPORARY
#include "guts/TensorImpl.h"

namespace c10 {

void Tensor::resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  impl_->HACK_resize_(size, stride);
}

void Tensor::copy_(ScalarType s, const void* p, int64_t size_bytes) {
  impl_->HACK_copy_(s, p, size_bytes);
}

}
