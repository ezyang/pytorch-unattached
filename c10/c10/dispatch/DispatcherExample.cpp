#include "Dispatcher.h"
#include <c10.h>
#include <c10/cpu/CPUTensorImpl.h>

using namespace c10;

namespace ops {
struct conditional final {
  static constexpr OpId op_id() {return OpId{0}; }
  using Signature = Tensor(bool, Tensor, Tensor);
};
struct add_notensor final {
  static constexpr OpId op_id() {return OpId{1};}
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

int add_notensor_op(int lhs, int rhs) {
  return lhs + rhs;
}

using c10::cpu::CPU_TENSOR;

int main() {
  Dispatcher& d = Dispatcher::singleton();
  d.registerOp<ops::conditional>(&conditional_op, {CPU_TENSOR(), CPU_TENSOR()});
  d.registerOp<ops::add_notensor>(&add_notensor_op);

  Tensor t1 = tensor<int>({5});
  Tensor t2 = tensor<int>({10});
  Tensor t3 = d.call<ops::conditional>(false, t1, t2);
   std::cout << "Result is " << t3.data<int>()[0] << std::endl;
  // outputs "Result is 10"

  std::cout << "Result is " << d.call<ops::add_notensor>(3, 6) << std::endl;
  // outputs "Result is 9"
}
