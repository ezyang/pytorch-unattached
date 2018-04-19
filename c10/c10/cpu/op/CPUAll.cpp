#include <c10/cpu/op/CPUAll.h>

#include <c10.h>
#include <c10/cpu/CPUTensorImpl.h>

namespace c10 { namespace cpu { namespace op {

// The actual implementations

// PRIVATE PRIVATE PRIVATE!!!
static CPUTensorImpl* _cpu_impl(const Tensor& self) {
  C10_ASSERT(self.type_id() == TypeIds::CPUTensor, "type_id = ", self.type_id());
  return static_cast<CPUTensorImpl*>(self._to_impl());
}

void zero_(const Tensor& self) {
  // TODO: This is wrong
  C10_ASSERT(self.is_contiguous(), "TODO: non-contiguous not supported yet")
  std::memset(self.data_ptr(), 0, self.numel() * self.dtype().itemsize());
}

Tensor empty(ArrayRef<int64_t> sizes, DataType dtype) {
  auto r = Tensor::_from_impl(new CPUTensorImpl(dtype));
  r.resize_(sizes);
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

Tensor tensor(void* data, ArrayRef<int64_t> sizes, DataType dtype) {
  auto r = op::empty(sizes, dtype); // nonvirtual
  op::copy_(r, dtype, data, r.numel() * dtype.itemsize()); // nonvirtual
  return r;
}

// TODO: There's some fishy business going on here, because the Caffe2 implementations
// of these functions are backend independent.  These implementations are not independent,
// but that's only because the storage methods are nonvirtual (so you have to be at the
// correct type when you invoke them.)

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
//
// TODO: This will probably be deprecated in favor of safer APIs
void resize_(const Tensor& self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride, bool keep_data) {
  C10_ASSERT(new_size.size() == new_stride.size(), "new_size = ", new_size, "; new_stride = ", new_stride);
  bool unchanged = new_size.equals(self.sizes()) && new_stride.equals(self.strides());
  if (unchanged) return;
  // TODO: This is an error-prone API call.  Might be safer to just pass self directly
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, new_stride, self.storage_offset() * self.dtype().itemsize());
  auto impl = _cpu_impl(self);
  impl->_set_sizes_and_strides(new_size, new_stride);
  // NB: In the old TH code, it was permissible for Storage to be a nullptr at this point.
  // We have tightened the internal invariants.  I put the ASSERT back in where the old
  // test for storage_ being nullptr would have been.
  auto cpu_storage = impl->cpu_storage();
  C10_ASSERT(cpu_storage, "cpu_storage is null");
  bool needs_resize =
      // not enough space, OR
      new_size_bytes > cpu_storage->sizeBytes() ||
      // we're not allowed to keep the old storage on a shrink, OR
      !globalCPUContext().keepOnShrink() ||
      // we shrunk greater than the maximum "keep on shrink" bytes.
      cpu_storage->sizeBytes() - new_size_bytes > globalCPUContext().maxKeepOnShrinkBytes();
  if (needs_resize) {
    cpu_storage->resize_(new_size_bytes, keep_data);
  }
}

// Channeling Caffe2 Tensor::Reserve(const std::vector<T>& newCapacity, ContextForCopy* context)
// TODO: Consider also having a direct "numels" variant.  Note that this version accounts
// correctly for strides
void reserve_(const Tensor& self, ArrayRef<int64_t> new_size) {
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, self.strides(), self.storage_offset() * self.dtype().itemsize());
  auto cpu_storage = _cpu_impl(self)->cpu_storage();
  if (new_size_bytes > cpu_storage->sizeBytes()) {
    // NB: Size of this tensor is unchanged!
    cpu_storage->resize_(new_size_bytes, true);
  }
}

// Channeling Caffe2 Tensor::Extend(TIndex num, float growthPct, ContextForCopy* context)
void extend_(const Tensor& self, int64_t num, double growthPct) {
  C10_CHECK(self.dim() >= 1);
  DimVector new_size{self.sizes()};
  new_size[0] += num;
  // NB: Do not need to test for storage_ == nullptr as it is assumed to
  // have been initialized
  auto tentative_new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, self.strides(), self.storage_offset() * self.dtype().itemsize());
  auto* impl =_cpu_impl(self);
  auto cpu_storage = impl->cpu_storage();
  if (tentative_new_size_bytes <= cpu_storage->sizeBytes()) {
    impl->_set_sizes_and_strides(new_size, self.strides());
    return;
  }
  // Compute the true sizes increase, to ensure extend() amortizes correctly
  new_size[0] = std::max(new_size[0], static_cast<int64_t>(std::ceil(self.sizes()[0] * (growthPct + 100) / 100)));
  // Short-circuit dynamic dispatch.
  resize_(self, new_size, self.strides(), true);
}


// THTensor_(equal)
bool equal(const Tensor& self, const Tensor& other) {
  C10_ASSERT(self.type_id() == TypeIds::CPUTensor, "self.type_id() = ", self.type_id());
  C10_ASSERT(other.type_id() == TypeIds::CPUTensor, "other.type_id() = ", other.type_id());
  C10_ASSERT(self.dtype() == other.dtype(), "self.dtype() = ", self.dtype(), "; other.dtype() = ", other.dtype())
  if (self.sizes().equals(other.sizes())) return false;
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
