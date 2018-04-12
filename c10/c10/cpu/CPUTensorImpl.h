#include "c10/guts/TensorImpl.h"
#include "c10/Optional.h"

#include "CPUStorage.h"
#include "CPUAllocator.h"

namespace c10 { namespace cpu {

class CPUTensorImpl final : public guts::TensorImpl {
  std::size_t element_size_;
  // Note: storage->size() may be greater than the recorded size of the tensor
  CPUStorage storage_;
  // Note: In Torch this can be nonzero, because we support views into the
  // inside of tensors.  In historic Caffe2 this was always zero.
  // NB: This is BYTES!!!  Different from TH historically, which was scalar size.
  std::size_t storage_offset_;
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
  SmallVector<int64_t, 5> stride_;
public:
  CPUTensorImpl(std::size_t element_size, const CPUStorage& storage)
  : TensorImpl(TypeIds::CPUTensor)
      , element_size_(element_size)
      , storage_(storage)
  {};

  void *data_ptr() const override {
    return static_cast<void*>(static_cast<char*>(storage_->data_ptr()) + storage_offset_);
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
  void HACK_resize_(ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride) override {
    C10_ASSERT(new_size.size() == new_stride.size());
    bool unchanged = new_size.equals(size()) && new_stride.equals(stride());
    if (unchanged) return;
    auto compute_extent = [](ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
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
      return std::pair<ssize_t, ssize_t>(low_watermark, high_watermark);
    };
    ssize_t low_watermark, high_watermark;
    std::tie(low_watermark, high_watermark) = compute_extent(new_size, new_stride);
    size_.assign(new_size.begin(), new_size.end());
    stride_.assign(new_stride.begin(), new_stride.end());
  }
};

}} // namespace c10::cpu
