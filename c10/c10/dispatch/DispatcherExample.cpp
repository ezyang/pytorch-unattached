#include "Dispatcher.h"
#include <c10.h>
#include <c10/cpu/CPUTensorImpl.h>
#include <c10/dispatch/OpRegistration.h>
#include <c10/dispatch/OpSchemaRegistration.h>

using namespace c10;
using c10::cpu::CPU_TENSOR;

namespace ops {
struct conditional final {
  using Signature = Tensor(bool, const Tensor&, Tensor);
};
struct add_notensor final {
  using DispatchKey = c10::DispatchKey<0>;
  using Signature = int(int, int);
};
}
C10_DEFINE_OP_SCHEMA(ops::conditional);
C10_DEFINE_OP_SCHEMA(ops::add_notensor);

// TODO Need some test case that tensors can be passed as ref and by value
Tensor conditional_op(bool condition, const Tensor& thenTensor, Tensor elseTensor) {
  if (condition) {
    return thenTensor;
  } else {
    return elseTensor;
  }
}

C10_REGISTER_OP().define<ops::conditional>(&conditional_op, {CPU_TENSOR(), CPU_TENSOR()});

int add_notensor_op(int lhs, int rhs) {
  return lhs + rhs;
}

C10_REGISTER_OP().define<ops::add_notensor>(&add_notensor_op);

int main() {
  Tensor t1 = tensor<int>({5});
  Tensor t2 = tensor<int>({10});
  Tensor t3 = Dispatcher::call<ops::conditional>(false, t1, t2);
   std::cout << "Result is " << t3.data<int>()[0] << std::endl;
  // outputs "Result is 10"

  std::cout << "Result is " << Dispatcher::call<ops::add_notensor>(3, 6) << std::endl;
  // outputs "Result is 9"
}
