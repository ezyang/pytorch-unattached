#include <c10.h>
#include <utility>
#include <c10/dispatch/OpSchemaRegistration.h>
#include <c10/dispatch/OpRegistration.h>
#include <c10/cpu/CPUTensorImpl.h>
#include <c10/stack_bindings/StackBasedOperatorRegistry.h>
#include <c10/stack_bindings/CallStack.h>
#include <iostream>

using namespace c10;
using cpu::CPU_TENSOR;

namespace ops {
struct conditional final {
  using Signature = Tensor(bool, const Tensor&, Tensor);

  static constexpr std::array<const char*, 3> parameter_names = {
    "condition", "lhs", "rhs"
  };
};
}
C10_DEFINE_OP_SCHEMA(::ops::conditional);

Tensor conditional_op(bool condition, const Tensor& thenTensor, Tensor elseTensor) {
  if (condition) {
    return thenTensor;
  } else {
    return elseTensor;
  }
}

C10_REGISTER_OP(::ops::conditional)
  .kernel(&conditional_op)
  .dispatchKey({c10::details::TensorParameterDispatchKey{CPU_TENSOR(), TypeMeta::Id<int>()}, c10::details::TensorParameterDispatchKey{CPU_TENSOR(), TypeMeta::Id<int>()}});

int main() {
  CallStack callStack;

  callStack.push(tensor<int>({5}));
  callStack.push(tensor<int>({10}));
  callStack.push(true);

  auto op = StackBasedOperatorRegistry()->Create("::ops::conditional");
  (*op)(&callStack);

  Tensor result = callStack.pop<Tensor>();
  
  std::cout << "Result is " << result.data<int>()[0] << std::endl;
}