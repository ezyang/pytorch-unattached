#include "c10_sigmoid_op.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"

using caffe2::Tensor;
using caffe2::CPUContext;

C10_DEFINE_OP_SCHEMA(caffe2::SigmoidOp);
C10_DEFINE_OP_SCHEMA(caffe2::SigmoidOp2);

namespace caffe2 {
    REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(SigmoidOp, C10Sigmoid)
    REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_2(SigmoidOp2, C10Sigmoid2)
}
