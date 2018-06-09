#include "caffe2/core/operator_c10wrapper.h"
#include <c10/dispatch/OpSchemaRegistration.h>
#include <c10/dispatch/KernelRegistration.h>
#include "caffe2/utils/Array.h"

using caffe2::Tensor;
using caffe2::CPUContext;

struct DummyOp final {
  using Signature = Tensor<CPUContext>(Tensor<CPUContext>);

  static constexpr c10::guts::array<const char*, 1> parameter_names = {"input"};
};

C10_DEFINE_OP_SCHEMA(DummyOp);

Tensor<CPUContext> dummy_op_int_impl(Tensor<CPUContext> arg) {
  Tensor<CPUContext> result;
  result.CopyFrom(arg);
  result.mutable_data<int>()[0] += 10;
  return result;
}

Tensor<CPUContext> dummy_op_float_impl(Tensor<CPUContext> arg) {
  Tensor<CPUContext> result;
  result.CopyFrom(arg);
  result.mutable_data<float>()[0] += 20;
  return result;
}

namespace c10 {
C10_REGISTER_KERNEL(DummyOp)
  .kernel(&dummy_op_int_impl)
  .dispatchKey({c10::CAFFE2_CPU_TENSOR(), c10::TypeMeta::Id<int>()});
C10_REGISTER_KERNEL(DummyOp)
  .kernel(&dummy_op_float_impl)
  .dispatchKey({c10::CAFFE2_CPU_TENSOR(), c10::TypeMeta::Id<float>()});
}
namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(DummyOp, DummyOp)
}
