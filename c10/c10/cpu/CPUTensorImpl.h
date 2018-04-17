#include <c10/DimVector.h>
#include "c10/guts/TensorImpl.h"
#include "c10/Optional.h"

#include "CPUStorage.h"

#include <numeric>
#include <cmath>

namespace c10 { namespace cpu {

// Everything is int64_t to prevent us from accidentally doing a signed-unsigned operation
// which is basically never what you want.  But using int64_t instead of int64_t shuts
// up the compiler about size_t conversions from standard library.

class CPUTensorImpl final : public guts::TensorImpl {

  // NB: reserved from Caffe2 axed; as there are TWO sizes, we can easily
  // implement the reserved pattern by having the storage be larger than the
  // sizes recorded in a Tensor.  Hooray!
  // dzhulgakov: superlike! :)
  // TODO: Move this to the parent class
  // Reminder: The way stride works is:
  //    sizes[0]*stride[0] + sizes[1]*stride[1] + ...
  // This means you can end up in weird situations.  Make sure to think about:
  //    stride[i] == 0 (broadcasting)
  //    stride[i] < 0 (negative strides)
  //    sizes[i] == 0 (useful to maintain sizes information!)
  //    stride[i] % sizes[i-1] != 0 (rolling window strides / not "embeddable")
  //    len(sizes) == 0 (scalars)
  // dzhulgakov: how much "stride analysis" do implementations usually do in TH?
  // See also https://ezyang.github.io/stride-visualizer/index.html
  DimVector stride_;

public:
  CPUTensorImpl(DataType dtype, const CPUStorage& storage)
  : TensorImpl(TypeIds::CPUTensor, dtype, storage)
  {
    C10_ASSERT(storage);
  };


  CPUStorage cpu_storage() {
    return std::static_pointer_cast<CPUStorageImpl>(storage_);
  }
};

}} // namespace c10::cpu
