#include "OpSchema.h"
#include "caffe2/core/dispatch/TensorTypeId.h"

namespace c10 {

C10_DEFINE_TENSOR_TYPE(CAFFE2_CPU_TENSOR)
C10_DEFINE_TENSOR_TYPE(CAFFE2_CUDA_TENSOR)

}
