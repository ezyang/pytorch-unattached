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

inline int64_t Tensor::size(int64_t dim) const {
  return impl_->size().at(dim);
}

inline int64_t Tensor::stride(int64_t dim) const {
  return impl_->stride().at(dim);
}

// smessmer to @ezyang: Do we want to try honoring const-ness for the underlying data?
//          i.e. const T* data() const {} and T* data() {} ?
//          not sure if it's a good idea, but we should consider it.
// ezyang to @smessmer: This is difficult to do without adding more user-visible 'Tensor' types.
//          Back story is at https://github.com/zdevito/ATen/issues/27
inline void *Tensor::data_ptr() const {
  return impl_->data_ptr();
}

inline int64_t Tensor::storage_offset() const {
  return impl_->storage_offset();
}

inline int64_t Tensor::numel() const {
  return impl_->numel();
}

inline int64_t Tensor::ndimension() const {
  return dim();
}

template<typename T>
inline T *Tensor::data() const {
  // dzhulgakov: also, if tensor doesn't support raw pointers - is it expected to throw?
  // ezyang: yes.  Not implemented yet.
  // clion hates me (scalar_type is ambiguous)
  C10_ASSERT(scalar_type<T>() == impl_->scalar_type());
  return static_cast<T *>(data_ptr());
}

} // namespace c10
