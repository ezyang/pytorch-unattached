#include "caffe2/core/dispatch/Dispatcher.h"
#include "caffe2/c10_prototype/c10.h"
#include "caffe2/c10_prototype/c10/cpu/CPUTensorImpl.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/utils/Array.h"

using namespace c10;

namespace ops {
struct conditional final {
  static constexpr const char* name = "conditional";

  using Signature = Tensor(bool, const Tensor&, Tensor);

  static constexpr guts::array<const char*, 3> parameter_names = {{
    "condition", "lhs", "rhs"
  }};
};
struct add_notensor final {
  static constexpr const char* name = "add_notensor";

  using Signature = int(int, int);

  static constexpr guts::array<const char*, 2> parameter_names = {{
    "lhs", "rhs"
  }};

  static std::string dispatch_key(int, int) {
    return "bla";
  }
};
}
C10_DEFINE_OP_SCHEMA(::ops::conditional);
C10_DEFINE_OP_SCHEMA(::ops::add_notensor);

// TODO Need some test case that tensors can be passed as ref and by value
Tensor conditional_op(bool condition, const Tensor& thenTensor, Tensor elseTensor) {
  if (condition) {
    return thenTensor;
  } else {
    return elseTensor;
  }
}

// TODO Commented out because I removed c10::Tensor compatibility from OpSchema
//C10_REGISTER_KERNEL(::ops::conditional)
//  .kernel(&conditional_op)
//  .dispatchKey({c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<int>()}, c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<int>()}});

int add_notensor_op(int lhs, int rhs) {
  return lhs + rhs;
}

C10_REGISTER_KERNEL(::ops::add_notensor)
  .kernel(&add_notensor_op)
  .dispatchKey("bla");

struct equals final {
  static constexpr const char* name = "equals";
  using Signature = bool(Tensor, Tensor);
  static constexpr guts::array<const char*, 2> parameter_names = {{
    "lhs", "rhs"
  }};
};

C10_DEFINE_OP_SCHEMA(::equals);

bool equals_impl(Tensor, Tensor) {
  return true;
}

// TODO Commented out because I removed c10::Tensor compatibility from OpSchema
//C10_REGISTER_KERNEL(::equals)
//  .kernel(&equals_impl)
//  .dispatchKey({c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<int>()}, c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<int>()}});

int main() {
  Tensor t1 = tensor<int>({5});
  Tensor t2 = tensor<int>({10});
  Tensor t3 = Dispatcher<::ops::conditional>::call(false, t1, t2);
   std::cout << "Result is " << t3.data<int>()[0] << std::endl;
  // outputs "Result is 10"

  std::cout << "Result is " << Dispatcher<::ops::add_notensor>::call(3, 6) << std::endl;
  // outputs "Result is 9"
}
