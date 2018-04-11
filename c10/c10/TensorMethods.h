#pragma once

#include "Tensor.h"
#include "guts/TensorImpl.h"

// These inline method definitions have to live in a separate file than Tensor.h
// because we need a complete definition of Tensor to work with from TensorImpl
// (temporarily, anyway) and we need to break the cycle.

namespace c10 {

inline int64_t Tensor::dim() const {
  return impl_->dim();
}

inline ArrayRef<int64_t> Tensor::size() const {
  return impl_->size();
}

inline ArrayRef<int64_t> Tensor::stride() const {
  return impl_->stride();
}

// smessmer to @ezyang: Do we want to try honoring const-ness for the underlying data?
//          i.e. const T* data() const {} and T* data() {} ?
//          not sure if it's a good idea, but we should consider it.
// ezyang to @smessmer: This is difficult to do without adding more user-visible 'Tensor' types.
//          Back story is at https://github.com/zdevito/ATen/issues/27
inline void *Tensor::data_ptr() const {
  return impl_->data_ptr();
}

} // namespace c10
