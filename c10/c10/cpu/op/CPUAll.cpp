#include <c10/cpu/op/CPUAll.h>

#include <c10.h>
#include <c10/cpu/CPUTensorImpl.h>

namespace c10 { namespace cpu { namespace op {

// The actual implementations

// TCB
static CPUTensorImpl* _cpu_impl(const Tensor& self) {
  C10_ASSERT(self.type_id() == TypeIds::CPUTensor, "type_id = ", self.type_id());
  return static_cast<CPUTensorImpl*>(self._to_impl());
}

void zero_(const Tensor& self) {
  // TODO: This is wrong
  C10_ASSERT(self.is_contiguous(), "TODO: non-contiguous not supported yet (sizes = ", self.sizes(), ", strides = ", self.strides(), ")")
  std::memset(self.data_ptr(), 0, static_cast<size_t>(self.numel() * self.dtype().itemsize()));
}

// TCB
Tensor empty(ArrayRef<int64_t> sizes, DataType dtype) {
  auto r = Tensor::_from_impl(new CPUTensorImpl(dtype));
  // Please do not copy paste the line below, it relies on the invariant that
  // a fresh storage was allocated
  _cpu_impl(r)->cpu_storage()->resize_(product(sizes) * dtype.itemsize(), /*keep data*/ false);
  _cpu_impl(r)->_set_sizes_and_strides(sizes, contiguous_strides(sizes));
  return r;
}

Tensor zeros(ArrayRef<int64_t> sizes, DataType dtype) {
  auto r = op::empty(sizes, dtype);  // nonvirtual
  r.zero_();
  return r;
}

// Channeling Caffe2 Tensor::Tensor(const T& value, Context* context)
void copy_(const Tensor& self, DataType dtype, const void* p, int64_t size_bytes) {
  C10_CHECK(dtype == self.dtype());
  _cpu_impl(self)->cpu_storage()->copy_(p, size_bytes);
}

Tensor tensor(const void* data, ArrayRef<int64_t> sizes, DataType dtype) {
  auto r = op::empty(sizes, dtype); // nonvirtual
  op::copy_(r, dtype, data, r.numel() * dtype.itemsize()); // nonvirtual
  return r;
}

// TODO: There's some fishy business going on here, because the Caffe2 implementations
// of these functions are backend independent.  These implementations are not independent,
// but that's only because the storage methods are nonvirtual (so you have to be at the
// correct type when you invoke them.)

// This function implements, bug-for-bug, the resize logic of PyTorch resize_
// a.k.a. THTensor_(resizeNd)
//
// TODO: Try to deprecate this as much as possible...
// Channeling THTensor_(resizeNd)
// If aggressive = true, we will always try to free up old memory (this means
// we always have to do a reallocation).  Torch default behavior was to
// keep the old data around; Caffe2's behavior is to do a full reallocate.
// NB: This code is GENERIC for all strided tensors.
// When strides is not set, it is assumed you wanted to preserve the original strides
// NB: resizeNd used to accept NULL strides, in which case contiguous strides are
// assumed.  To keep this function simple, we FORCE the callee to pass new_stride;
// it's a simple matter to compute what the appropriate contiguous strides for a
// tensor are.
// WARNING: BC-breaking change; previously a negative number was assumed to mean
// "compute whatever the appropriate contiguous strides is."  But this didn't even
// work in all cases; when determining if sizes/strides had changed, resizeNd would
// incorrectly assume the original tensor was
// contiguously strided for every negative index, even when it was not.
// See also https://github.com/pytorch/pytorch/issues/229
void legacy_pytorch_resize_(const Tensor &self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride) {
  C10_ASSERT(new_size.size() == new_stride.size(), "new_size = ", new_size, "; new_stride = ", new_stride);
  bool unchanged = new_size.equals(self.sizes()) && new_stride.equals(self.strides());
  if (unchanged) return;
  // TODO: This is an error-prone API call.  Might be safer to just pass self directly
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, new_stride, self.storage_offset() * self.dtype().itemsize());
  auto impl = _cpu_impl(self);
  auto cpu_storage = impl->cpu_storage();
  if (new_size_bytes > cpu_storage->size_bytes() ) {
    cpu_storage->resize_(new_size_bytes, true);
  }
  impl->_set_sizes_and_strides(new_size, new_stride);
}

void legacy_caffe2_resize_(const Tensor &self, ArrayRef<int64_t> new_size) {
  C10_CHECK(self.is_contiguous(), "Caffe2-style resize not supported for non-contiguous tensors");
  if (new_size.equals(self.sizes())) return;
  // TODO: This is an error-prone API call.  Might be safer to just pass self directly
  auto new_stride = contiguous_strides(new_size);
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, new_stride, self.storage_offset() * self.dtype().itemsize());
  auto impl = _cpu_impl(self);
  auto cpu_storage = impl->cpu_storage();
  bool needs_resize =
      // not enough space, OR
      new_size_bytes > cpu_storage->size_bytes() ||
      // we shrunk greater than the maximum "keep on shrink" bytes.
      cpu_storage->size_bytes() - new_size_bytes > globalCPUContext().maxKeepOnShrinkBytes().value_or(INT64_MAX);
  if (needs_resize) {
    // Caffe2 resize never keeps data
    cpu_storage->resize_(new_size_bytes, /* keep data */ false);
  }
  impl->_set_sizes_and_strides(new_size, new_stride);
}


// Channeling Caffe2 Tensor::Reserve(const std::vector<T>& newCapacity, ContextForCopy* context)
// TODO: Consider also having a direct "numels" variant.
// Note that this version accounts correctly for strides
void reserve_(const Tensor& self, ArrayRef<int64_t> new_size) {
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, self.strides(), self.storage_offset() * self.dtype().itemsize());
  auto cpu_storage = _cpu_impl(self)->cpu_storage();
  if (new_size_bytes > cpu_storage->size_bytes()) {
    // NB: Size of this tensor is unchanged!
    cpu_storage->resize_(new_size_bytes, true);
  }
}

// Channeling Caffe2 Tensor::Extend(TIndex num, float growthPct, ContextForCopy* context)
void extend_(const Tensor& self, int64_t num, double growthPct) {
  C10_CHECK(self.dim() >= 1);

  auto* impl =_cpu_impl(self);
  auto cpu_storage = impl->cpu_storage();

  // Compute initialize size increase
  DimVector new_size{self.sizes()};
  new_size[0] += num;
  auto tentative_new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, self.strides(), self.storage_offset() * self.dtype().itemsize());
  if (tentative_new_size_bytes <= cpu_storage->size_bytes()) {
    // Cheap! Do the quick and easy thing
    impl->_set_sizes_and_strides(new_size, self.strides());
    return;
  }

  // Compute the true size increase, to ensure extend() amortizes correctly
  new_size[0] = std::max(new_size[0], static_cast<int64_t>(std::ceil(self.sizes()[0] * (growthPct + 100) / 100)));
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, self.strides(), self.storage_offset() * self.dtype().itemsize());
  cpu_storage->resize_(new_size_bytes, /* keep data */ true);
  impl->_set_sizes_and_strides(new_size, self.strides());
}


// THTensor_(equal)
bool equal(const Tensor& self, const Tensor& other) {
  C10_ASSERT(self.type_id() == TypeIds::CPUTensor, "self.type_id() = ", self.type_id());
  C10_ASSERT(other.type_id() == TypeIds::CPUTensor, "other.type_id() = ", other.type_id());
  C10_ASSERT(self.dtype() == other.dtype(), "self.dtype() = ", self.dtype(), "; other.dtype() = ", other.dtype())
  if (!self.sizes().equals(other.sizes())) return false;
  if (self.is_contiguous() && other.is_contiguous()) {
    // TODO: This is WRONG for floating point
    return std::memcmp(self.data_ptr(), other.data_ptr(), static_cast<size_t>(self.numel() * self.dtype().itemsize())) == 0;
  } else {
    C10_ASSERT(false, "non-contiguous equality not supported yet");
  }
}

/*
// Channeling Caffe2 Tensor::CopyFrom(const Tensor<SrcContext>& src, ContextForCopy* context)
// and Tensor::CopyFrom(const Tensor<SrcContext>& src)
// This function is deferred until multiple dispatch is online, as it can only be conveniently
// implemented inside the multiple dispatch framework
void HACK_copy_(Tensor src) {
}
 */

}}} // namespace c10::cpu::op
