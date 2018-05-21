#include "caffe2/core/operator_c10wrapper.h"
#include <c10/dispatch/OpSchemaRegistration.h>
#include <c10/dispatch/OpRegistration.h>
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;

struct SigmoidOp2 final {
  using Signature = bool(const Tensor<CPUContext>&, Tensor<CPUContext>*);

  static constexpr std::array<const char*, 2> parameter_names = {"input", "output"};
};

C10_DEFINE_OP_SCHEMA(SigmoidOp2);


template<class DataType>
bool sigmoid_op_cpu_impl_2(const Tensor<CPUContext>& input, Tensor<CPUContext>* output) {
  //Tensor<CPUContext> output;
  output->ResizeLike(input);

  caffe2::ConstEigenVectorArrayMap<DataType> xM(input.data<DataType>(), input.size());
  caffe2::EigenVectorArrayMap<DataType>(output->mutable_data<DataType>(), input.size()) = 1. / (1. + (-xM).exp());

  //return output;
  return true;
}

namespace c10 {
C10_REGISTER_OP(SigmoidOp2)
  .kernel(&sigmoid_op_cpu_impl_2<float>)
  .dispatchKey({c10::CAFFE2_CUDA_TENSOR(), c10::TypeMeta::Id<int>()});
C10_REGISTER_OP(SigmoidOp2)
  .kernel(&sigmoid_op_cpu_impl_2<float>)
  .dispatchKey({c10::CAFFE2_CUDA_TENSOR(), c10::TypeMeta::Id<float>()});
C10_REGISTER_OP(SigmoidOp2)
  .kernel(&sigmoid_op_cpu_impl_2<float>)
  .dispatchKey({c10::CAFFE2_CPU_TENSOR(), c10::TypeMeta::Id<int>()});
C10_REGISTER_OP(SigmoidOp2)
  .kernel(&sigmoid_op_cpu_impl_2<float>)
  .dispatchKey({c10::CAFFE2_CPU_TENSOR(), c10::TypeMeta::Id<float>()});
}
namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_2(SigmoidOp2, C10Sigmoid2)
}


struct SigmoidOp final {
    using Signature = Tensor<CPUContext>(const Tensor<CPUContext>&);

    static constexpr std::array<const char*, 1> parameter_names = {"input"};
};

C10_DEFINE_OP_SCHEMA(SigmoidOp);


template<class DataType>
Tensor<CPUContext> sigmoid_op_cpu_impl(const Tensor<CPUContext>& input) {
    Tensor<CPUContext> output;
    output.ResizeLike(input);

    caffe2::ConstEigenVectorArrayMap<DataType> xM(input.data<DataType>(), input.size());
    caffe2::EigenVectorArrayMap<DataType>(output.mutable_data<DataType>(), input.size()) = 1. / (1. + (-xM).exp());

    return output;
}

namespace c10 {
C10_REGISTER_OP(SigmoidOp)
        .kernel(&sigmoid_op_cpu_impl<float>)
        .dispatchKey({c10::CAFFE2_CPU_TENSOR(), c10::TypeMeta::Id<float>()});
}
namespace caffe2 {
    REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(SigmoidOp, C10Sigmoid)
}
