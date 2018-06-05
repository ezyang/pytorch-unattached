#include "OpSchema.h"
#include <c10/dispatch/TensorTypeId.h>

namespace c10 {

C10_DEFINE_TENSOR_TYPE(CAFFE2_CPU_TENSOR)
C10_DEFINE_TENSOR_TYPE(CAFFE2_CUDA_TENSOR)

}
