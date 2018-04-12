#include <c10/DimVector.h>
#include "c10/guts/TensorImpl.h"
#include "c10/Optional.h"

#include "CPUStorage.h"
#include "CPUAllocator.h"

namespace c10 { namespace cpu {

DimVector contiguous_strides(ArrayRef<int64_t> size) {
  DimVector v(size.size());
  ssize_t total_size = 1;
  for (ssize_t d = size.size() - 1; d >= 0; d--) {
    v[d] = total_size;
    total_size *= size[d];
  }
  return v;  // RVO
}

// TODO: Refactor this into a utility header file
std::pair<ssize_t, ssize_t> compute_extent(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  // Watermarks are inclusive.  NB: watermarks can be negative! Careful!
  ssize_t high_watermark = 0;
  ssize_t low_watermark = 0;
  for (ssize_t d = size.size() - 1; d >= 0; d--) {
    if (stride[d] >= 0) {
      high_watermark += (size[d] - 1) * stride[d];
    } else {
      low_watermark += (size[d] - 1) * stride[d];
    }
  }
  return {low_watermark, high_watermark};
};

// Everything is ssize_t to prevent us from accidentally doing a signed-unsigned operation
// which is basically never what you want.  But using ssize_t instead of int64_t shuts
// up the compiler about size_t conversions from standard library.

class CPUTensorImpl final : public guts::TensorImpl {
  ssize_t element_size_bytes_;
  // Note: storage->size() may be greater than the recorded size of the tensor
  // ezyang to @smessmer: Maybe we should consider using a never-null pointer
  CPUStorage storage_;
  // Note: In Torch this can be nonzero, because we support views into the
  // inside of tensors.  In historic Caffe2 this was always zero.
  // NB: This is BYTES!!!  Different from TH historically, which was scalar size.
  ssize_t storage_offset_bytes_;
  // NB: shares_data from Caffe2 was axed, because it is SOLELY used to determine
  // check what the overall tensor usage is.  We can rewrite that code to
  // keep a mapping of storage base pointers that it has seen (these all
  // "count" the same), and perhaps add a bit to storage which tells us if
  // it is "external" or "internal" (external storages don't count for accounting
  // purposes.)
  // NB: reserved from Caffe2 axed; as there are TWO sizes, we can easily
  // implement the reserved pattern by having the storage be larger than the
  // size recorded in a Tensor.  Hooray!
  // TODO: Move this to the parent class
  // Reminder: The way stride works is:
  //    size[0]*stride[0] + size[1]*stride[1] + ...
  // This means you can end up in weird situations.  Make sure to think about:
  //    stride[i] == 0 (broadcasting)
  //    stride[i] < 0 (negative strides)
  //    size[i] == 0 (useful to maintain size information!)
  //    stride[i] % size[i-1] != 0 (rolling window strides / not "embeddable")
  //    len(size) == 0 (scalars)
  // See also https://ezyang.github.io/stride-visualizer/index.html
  DimVector stride_;
public:
  CPUTensorImpl(ssize_t element_size_bytes, const CPUStorage& storage)
  : TensorImpl(TypeIds::CPUTensor)
      , element_size_bytes_(element_size_bytes)
      , storage_(storage)
  {};

  void *data_ptr() const override {
    if (!storage_) return nullptr;
    return static_cast<void*>(static_cast<char*>(storage_->data_ptr()) + storage_offset_bytes_);
  }

  // Hacked up operators

  static Tensor HACK_tensor(std::size_t element_size, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
    auto storage = std::make_shared<CPUStorageImpl>();
    Tensor r = Tensor::_fromImpl(new CPUTensorImpl(element_size, storage));
    r.resize_(size, stride);
    return r;
  }

  // Channeling THTensor_(resizeNd)
  // NB: This code is GENERIC for all strided tensors.
  // When stride is not set, it is assumed you wanted to preserve the original stride
  // NB: resizeNd used to accept NULL stride, in which case contiguous strides are
  // assumed.  To keep this function simple, we FORCE the callee to pass new_stride;
  // it's a simple matter to compute what the appropriate contiguous strides for a
  // tensor are.
  // WARNING: BC-breaking change; previously a negative number was assumed to mean
  // "compute whatever the appropriate contiguous stride is."  But this didn't even
  // work in all cases; when determining if sizes/strides had changed, resizeNd would
  // incorrectly assume the original tensor was
  // contiguously strided for every negative index, even when it was not.
  // See also https://github.com/pytorch/pytorch/issues/229
  // NB: This does NOT deallocate memory when you resize to something smaller.
  void HACK_resize_(ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride) override {
    C10_ASSERT(new_size.size() == new_stride.size());
    bool unchanged = new_size.equals(size()) && new_stride.equals(stride());
    if (unchanged) return;
    ssize_t low_watermark, high_watermark;
    std::tie(low_watermark, high_watermark) = compute_extent(new_size, new_stride);
    // NB: Actually, we should be able to support resizing the left-side of the tensor; we
    // simply need to support the case when the pointer doesn't point to the beginning
    // of the allocated region; this means we need to store an offset to "undo" the
    // shift later.  Should be simple but we don't implement it for now.
    if (low_watermark * element_size_bytes_ + storage_offset_bytes_ < 0) {
      throw std::runtime_error("Cannot resize past beginning of tensor");
    }
    size_.assign(new_size.begin(), new_size.end());
    stride_.assign(new_stride.begin(), new_stride.end());
    if (high_watermark * element_size_bytes_ + storage_offset_bytes_ > 0) {
      if (!storage_) {
        storage_ = std::make_shared<CPUStorageImpl>();
      }
      auto new_size_bytes = high_watermark * element_size_bytes_ + storage_offset_bytes_;
      if (new_size_bytes > storage_->sizeBytes()) {
        storage_->resize_(new_size_bytes);
      }
    }
  }
};

}} // namespace c10::cpu
