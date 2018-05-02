#include "Dispatcher.h"
#include <c10.h>
#include <c10/cpu/CPUTensorImpl.h>
#include <c10/dispatch/OpRegistration.h>

using namespace c10;
using c10::cpu::CPU_TENSOR;

namespace ops {
struct conditional final {
  static constexpr OpId op_id() {return OpId{1000}; }
  using Signature = Tensor(bool, Tensor, Tensor);
};
struct add_notensor final {
  static constexpr OpId op_id() {return OpId{1001};}
  using Signature = int(int, int);
};
}

Tensor conditional_op(bool condition, Tensor thenTensor, Tensor elseTensor) {
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
  Dispatcher& d = Dispatcher::singleton();

  Tensor t1 = tensor<int>({5});
  Tensor t2 = tensor<int>({10});
  Tensor t3 = d.call<ops::conditional>(false, t1, t2);
   std::cout << "Result is " << t3.data<int>()[0] << std::endl;
  // outputs "Result is 10"

  std::cout << "Result is " << d.call<ops::add_notensor>(3, 6) << std::endl;
  // outputs "Result is 9"
}
