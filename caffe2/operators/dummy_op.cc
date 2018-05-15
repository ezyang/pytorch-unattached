#include "caffe2/core/operator_c10wrapper.h"
#include <c10/dispatch/OpSchemaRegistration.h>
#include <c10/dispatch/OpRegistration.h>

using caffe2::Tensor;
using caffe2::CPUContext;

struct DummyOp final {
  using Signature = Tensor<CPUContext>(Tensor<CPUContext>);

  static constexpr std::array<const char*, 1> parameter_names = {"input"};
};

C10_DEFINE_OP_SCHEMA(DummyOp);

Tensor<CPUContext> dummy_op_impl(Tensor<CPUContext> arg) {
  Tensor<CPUContext> result;
  result.CopyFrom(arg);
  result.mutable_data<float>()[0] += 1;
  return result;
}

namespace c10 {
C10_REGISTER_OP(DummyOp)
  .kernel(&dummy_op_impl)
  .dispatchKey({});
}
namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(DummyOp, DummyOp)
}
