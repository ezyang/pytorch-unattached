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
