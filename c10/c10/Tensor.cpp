#include "Tensor.h"

namespace c10 {

void Tensor::resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  //impl_->HACK_resize_(size, stride);
}

void Tensor::copy_(DataType dtype, const void* p, int64_t size_bytes) {
  //impl_->HACK_copy_(dtype, p, size_bytes);
}

void Tensor::extend_(int64_t num, double growthPct) {
  //impl_->HACK_extend_(num, growthPct);
}

void Tensor::reserve_(ArrayRef<int64_t> new_size) {
  //impl_->HACK_reserve_(new_size);
}

}
