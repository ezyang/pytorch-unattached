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

// Caffe2 Tensor::ResizeLike
// PyTorch resize_as_
// NB: This DESTROYS stride information and resizes as if it were
// contiguous
void resize_as_(const Tensor& self, const Tensor& other) {
  self.resize_(other.sizes(), contiguous_strides(other.sizes()));
}

// Caffe2 Tensor::Reshape (out-of-place)
// PyTorch view
//
// Should this be called view() or reshape()?  There is some funny business
// going on here when the input tensor is not necessarily contiguous.
// Numpy reshape() ALWAYS works; if it can't figure out how
Tensor view(const Tensor& self, ArrayRef<int64_t> new_size) {
  //return reshape(self, new_size);
}

// THTensor_(reshape)
// Numpy reshape
// Renaming authorized by sgross in https://discuss.pytorch.org/t/equivalent-of-np-reshape-in-pytorch/144
/*
void reshape(const Tensor& self, ArrayRef<int64_t> new_size) {

}
 */


}} // namespace c10::op
