#include "Tensor.h"

// STRICTLY TEMPORARY
#include "guts/TensorImpl.h"

namespace c10 {

void Tensor::resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  impl_->HACK_resize_(size, stride);
}

}
