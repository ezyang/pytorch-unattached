#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/Array.h"

using caffe2::Tensor;
using caffe2::CPUContext;

struct ElementwiseLinearOp final {
  static constexpr const char* name = "elementwise_linear";

  using Signature = Tensor<CPUContext>(const Tensor<CPUContext>&, const Tensor<CPUContext>&, const Tensor<CPUContext>&);

  static constexpr c10::guts::array<const char*, 3> parameter_names = {{"X", "a", "b"}};
};

C10_DEFINE_OP_SCHEMA(ElementwiseLinearOp);


Tensor<CPUContext> elementwise_linear_op_cpu_impl(const Tensor<CPUContext>& X, const Tensor<CPUContext>& a, const Tensor<CPUContext>& b) {
    // TODO Take axis as argument
    constexpr int axis_ = 1;
    Tensor<CPUContext> Y;

    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int N = X.size_to_dim(canonical_axis);
    const int D = X.size_from_dim(canonical_axis);

    CAFFE_ENFORCE_EQ(a.ndim(), 1, a.ndim());
    CAFFE_ENFORCE_EQ(a.dim(0), D, a.ndim());
    CAFFE_ENFORCE_EQ(b.ndim(), 1, b.ndim());
    CAFFE_ENFORCE_EQ(b.dim(0), D, b.ndim());

    Y.ResizeLike(X);

    const float* X_data = X.data<float>();
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* Y_data = Y.mutable_data<float>();

    int p = 0;
    for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
            Y_data[p] = X_data[p] * a_data[d] + b_data[d];
            p++;
        }
    }
    return Y;
}

namespace c10 {
C10_REGISTER_KERNEL(ElementwiseLinearOp)
  .kernel(&elementwise_linear_op_cpu_impl)
  .dispatchKey({c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()}, c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()}, c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()}});
}
namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(ElementwiseLinearOp, C10ElementwiseLinear)
}
