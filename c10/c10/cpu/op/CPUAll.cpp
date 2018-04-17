#include <c10/c10.h>

#include <c10/cpu/CPUTensorImpl.h>
#include "CPUAll.h"

namespace c10 { namespace cpu { namespace op {

// The actual implementations

// Hmm... this is still a bit awkward...  using private _fromImpl to get
// the implementation going???  How do we actually want to write this?
Tensor tensor(DataType dtype) {
  auto storage = std::make_shared<CPUStorageImpl>(dtype);
  return Tensor::_fromImpl(new CPUTensorImpl(dtype, storage));
}

// PRIVATE PRIVATE PRIVATE!!!
static CPUTensorImpl* _cpu_impl(const Tensor& self) {
  C10_ASSERT(self.type_id() == TypeIds::CPUTensor);
  return static_cast<CPUTensorImpl*>(self._toImpl());
}

// Channeling Caffe2 Tensor::Tensor(const T& value, Context* context)
void copy_(const Tensor& self, DataType dtype, const void* p, int64_t size_bytes) {
  C10_CHECK(dtype == self.dtype());
  _cpu_impl(self)->cpu_storage()->copy_(p, size_bytes);
}

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
void resize_(const Tensor& self, ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride, bool keep_data) {
  C10_ASSERT(new_size.size() == new_stride.size());
  bool unchanged = new_size.equals(self.sizes()) && new_stride.equals(self.strides());
  if (unchanged) return;
  auto new_size_bytes = required_new_storage_size_bytes(self.dtype(), new_size, new_stride, self.storage_offset());
  auto impl = _cpu_impl(self);
  /*
  impl->sizes_.assign(new_size.begin(), new_size.end());
  impl->stride_.assign(new_stride.begin(), new_stride.end());
  // NB: In the old TH code, it was permissible for Storage to be a nullptr at this point.
  // We have tightened the internal invariants.  I put the ASSERT back in where the old
  // test for storage_ being nullptr would have been.
  C10_ASSERT(storage_);
  bool needs_resize =
      // not enough space, OR
      new_size_bytes > storage_->sizeBytes() ||
      // we're not allowed to keep the old storage on a shrink, OR
      !globalCPUContext().keepOnShrink() ||
      // we shrunk greater than the maximum "keep on shrink" bytes.
      storage_->sizeBytes() - new_size_bytes > globalCPUContext().maxKeepOnShrinkBytes();
  if (needs_resize) {
    cpu_storage()->resize_(new_size_bytes, keep_data);
  }
   */
}

#if 0

// Channeling Caffe2 Tensor::Reserve(const std::vector<T>& newCapacity, ContextForCopy* context)
// TODO: Consider also having a direct "numels" variant.  Note that this version accounts
// correctly for strides
void reserve_(ArrayRef<int64_t> new_size) override {
  auto new_size_bytes = required_new_storage_size_bytes(scalar_type(), new_size, stride(), storage_offset_bytes_);
  if (new_size_bytes > storage_->sizeBytes()) {
    // NB: Size of this tensor is unchanged!
    cpu_storage()->resize_(new_size_bytes, true);
  }
}

// Channeling Caffe2 Tensor::Extend(TIndex num, float growthPct, ContextForCopy* context)
void extend_(int64_t num, double growthPct) override {
  C10_CHECK(dim() >= 1);
  DimVector new_size{size()};
  new_size[0] += num;
  // NB: Do not need to test for storage_ == nullptr as it is assumed to
  // have been initialized
  auto tentative_new_size_bytes = required_new_storage_size_bytes(scalar_type_, new_size, strides(), storage_offset_bytes_);
  if (tentative_new_size_bytes <= storage_->sizeBytes()) {
    sizes_ = new_size;
    return;
  }
  // Compute the true sizes increase, to ensure extend() amortizes correctly
  new_size[0] = std::max(new_size[0], static_cast<int64_t>(std::ceil(sizes()[0] * (growthPct + 100) / 100)));
  HACK_resize_(new_size, strides(), true);
}

/*
// Channeling Caffe2 Tensor::CopyFrom(const Tensor<SrcContext>& src, ContextForCopy* context)
// and Tensor::CopyFrom(const Tensor<SrcContext>& src)
// This function is deferred until multiple dispatch is online, as it can only be conveniently
// implemented inside the multiple dispatch framework
void HACK_copy_(Tensor src) {
}
 */

#endif

}}} // namespace c10::cpu::op
