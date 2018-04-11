#include "c10/guts/TensorImpl.h"

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
  SmallVector<int64_t> stride_;
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
  void HACK_resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) override {
    C10_ASSERT(size.size() == stride.size());
  }
};

}} // namespace c10::cpu
