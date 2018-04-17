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
// PyTorch "in-place" view
//
// Should this be called view() or reshape()?  There is some funny business
// going on here when the input tensor is not necessarily contiguous.
// Numpy reshape() ALWAYS works; if it can't figure out how to restride so
// that no allocation occurs, it just reallocates the thing.  view() in
// Torch-land will never do this: it will restride, or raise an error.
void view_(const Tensor& self, ArrayRef<int64_t> new_sizes) {
  // TODO: Generalize this to work on more stride situations.  I don't
  // need this for Caffe2, which is the current thrust, so I didn't
  // implement it
  C10_CHECK(self.is_contiguous());
  C10_CHECK(self.numel() == product(new_sizes));
  auto* impl = self._to_impl();
  impl->_set_sizes_and_strides(new_sizes, contiguous_strides(new_sizes));
}

// TODO
// THTensor_(reshape)
// Numpy reshape
/*
void reshape(const Tensor& self, ArrayRef<int64_t> new_size) {

}
 */


}} // namespace c10::op
