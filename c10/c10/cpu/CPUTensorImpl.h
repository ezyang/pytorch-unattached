#include <c10/cpu/CPUStorage.h>
#include <c10/guts/TensorImpl.h>
#include <c10/DimVector.h>
#include <c10/Optional.h>
#include <c10/dispatch/Dispatcher.h>

#include <numeric>
#include <cmath>

namespace c10 { namespace cpu {

C10_DECLARE_TENSOR_TYPE(CPU_TENSOR);

/**
 * Specialization of TensorImpl for CPU tensors.  Data layout is the same but we can make
 * extra assumptions about the types of some members.
 */
class CPUTensorImpl final : public guts::TensorImpl {
public:
  explicit CPUTensorImpl(DataType dtype)
  : TensorImpl(CPU_TENSOR(), dtype, {0}, {1}, std::make_shared<CPUStorageImpl>(dtype), 0)
  {};

  CPUStorage cpu_storage() {
    return std::static_pointer_cast<CPUStorageImpl>(storage_);
  }
};

}} // namespace c10::cpu
