#include <c10.h>

#include "All.h"

namespace c10 { namespace op {

void shrink_(const Tensor& self, int64_t outer_dim_new_size) {
  self._to_impl()->_shrink(outer_dim_new_size);
}

// TODO:
// Caffe2 Tensor::ResizeLike

// PyTorch resize_as_
// NB: This DESTROYS stride information and resizes as if it were
// contiguous
void legacy_pytorch_resize_as_(const Tensor& self, const Tensor& other) {
  self.legacy_pytorch_resize_(other.sizes(), contiguous_strides(other.sizes()));
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
  C10_CHECK(self.is_contiguous(), "cannot view on non-contiguous tensors (TODO: this is too restrictive)");
  C10_CHECK(self.numel() == product(new_sizes),
            "number of elements in original size ", self.sizes(), " and new size ", new_sizes,
            " must have same number of elements (", self.numel(), " != ", product(new_sizes), ")");
  auto* impl = self._to_impl();
  impl->_set_sizes_and_strides(new_sizes, contiguous_strides(new_sizes));
}

// THTensor_(newClone)
// NB: this is not an "exact" clone, because the resulting tensor is contiguous.
// However, if you just wanted a contiguous tensor, you should use contiguous()
// instead.
Tensor clone(const Tensor& /*self*/) {
  /*
  auto r = self.tensor(self.sizes(), contiguous_strides(self.sizes()));
  r.copy_(self);
  return r;
  */
  C10_ASSERT(0, "not yet implemented");
}

// TODO
// THTensor_(reshape)
// Numpy reshape
/*
void reshape(const Tensor& self, ArrayRef<int64_t> new_size) {

}
 */


}} // namespace c10::op
