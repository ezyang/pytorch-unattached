#include "c10/guts/TensorImpl.h"

#include "CPUStorage.h"
#include "CPUAllocator.h"

namespace c10 { namespace cpu {

class CPUTensorImpl final : public guts::TensorImpl {
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
  // TODO: Add strides
public:
  CPUTensorImpl(const CPUStorage& storage) : TensorImpl(TypeIds::CPUTensor), storage_(storage) {};

  inline void *data_ptr() const {
    return static_cast<void*>(static_cast<char*>(storage_->data_ptr()) + storage_offset_);
  }
};

}} // namespace c10::cpu
