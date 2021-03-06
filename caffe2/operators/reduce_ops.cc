#include "caffe2/operators/reduce_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
void ComputeReduceMinMaxGradient(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data) {
  const int dX_size = std::accumulate(
      dX_dims.cbegin(), dX_dims.cend(), 1, std::multiplies<int>());
  const int ndim = dX_dims.size();
  std::vector<int> index(ndim, 0);
  for (int dX_index = 0; dX_index < dX_size; ++dX_index) {
    const int dY_index =
        math::utils::GetIndexFromDims(ndim, dY_dims.data(), index.data());
    dX_data[dX_index] =
        Y_data[dY_index] == X_data[dX_index] ? dY_data[dY_index] : T(0);
    math::utils::IncreaseIndexInDims(ndim, dX_dims.data(), index.data());
  }
}

} // namespace

template <>
template <typename T>
bool MinReducer<CPUContext>::Backward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CPUContext* /* context */) const {
  ComputeReduceMinMaxGradient(
      dY_dims, dX_dims, dY_data, X_data, Y_data, dX_data);
  return true;
}

template <>
template <typename T>
bool MaxReducer<CPUContext>::Backward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CPUContext* /* context */) const {
  ComputeReduceMinMaxGradient(
      dY_dims, dX_dims, dY_data, X_data, Y_data, dX_data);
  return true;
}

REGISTER_CPU_OPERATOR(
    ReduceMin,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MinReducer<CPUContext>>);
REGISTER_CPU_OPERATOR(
    ReduceMinGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MinReducer<CPUContext>>);

OPERATOR_SCHEMA(ReduceMin)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Computes the min of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal True.
  If keepdims equal false, then the resulted tensor have the reduced dimension
  pruned.
)DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default True keeps the reduced "
        "dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

OPERATOR_SCHEMA(ReduceMinGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    ReduceMax,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MaxReducer<CPUContext>>);
REGISTER_CPU_OPERATOR(
    ReduceMaxGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MaxReducer<CPUContext>>);

OPERATOR_SCHEMA(ReduceMax)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Computes the max of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal True.
  If keepdims equal false, then the resulted tensor have the reduced dimension
  pruned.
)DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default True keeps the reduced "
        "dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

OPERATOR_SCHEMA(ReduceMaxGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    ReduceSum,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        SumReducer<CPUContext>>);
REGISTER_CPU_OPERATOR(
    ReduceSumGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        SumReducer<CPUContext>>);

OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the **sum** of the input tensor's elements along the provided `axes`. The resulting tensor has the same rank as the input if the `keepdims` argument equals 1 (default). If `keepdims` is set to 0, then the `axes` dimensions are pruned.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceSum",
    ["X"],
    ["Y"],
    axes=(0,1),
    keepdims=0
)

workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[[[5. 3. 7. 9. 5.]
   [4. 5. 1. 8. 3.]
   [1. 0. 9. 7. 6.]
   [7. 5. 0. 3. 1.]
   [6. 4. 4. 8. 3.]]

  [[8. 9. 6. 7. 7.]
   [5. 5. 4. 7. 0.]
   [9. 7. 6. 6. 7.]
   [7. 5. 2. 4. 2.]
   [4. 5. 1. 9. 4.]]]]
Y:
[[13. 12. 13. 16. 12.]
 [ 9. 10.  5. 15.  3.]
 [10.  7. 15. 13. 13.]
 [14. 10.  2.  7.  3.]
 [10.  9.  5. 17.  7.]]

```

</details>

)DOC")
    .Arg("axes", "(*Tuple(int)*): list of axes to reduce")
    .Arg(
        "keepdims",
        "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Output(0, "Y", "(*Tensor`<float>`*): reduced tensor");

OPERATOR_SCHEMA(ReduceSumGradient).NumInputs(3).NumOutputs(1);

REGISTER_CPU_OPERATOR(
    ReduceMean,
    ReduceOp<TensorTypes<float>, CPUContext, MeanReducer<CPUContext>>);
REGISTER_CPU_OPERATOR(
    ReduceMeanGradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, MeanReducer<CPUContext>>);

OPERATOR_SCHEMA(ReduceMean)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes the **mean** of the input tensor's elements along the provided `axes`. The resulting tensor has the same rank as the input if the `keepdims` argument equals 1 (default). If `keepdims` is set to 0, then the `axes` dimensions are pruned.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "ReduceMean",
    ["X"],
    ["Y"],
    axes=(0,1),
    keepdims=0
)

workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
print("X:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))

```

**Result**

```

X:
[[[[9. 0. 3. 6. 0.]
   [3. 4. 5. 0. 9.]
   [6. 9. 1. 1. 5.]
   [6. 2. 3. 7. 7.]
   [3. 1. 1. 0. 1.]]

  [[4. 3. 9. 8. 1.]
   [8. 2. 0. 4. 0.]
   [8. 9. 9. 0. 2.]
   [7. 2. 5. 8. 9.]
   [5. 9. 1. 9. 0.]]]]
Y:
[[6.5 1.5 6.  7.  0.5]
 [5.5 3.  2.5 2.  4.5]
 [7.  9.  5.  0.5 3.5]
 [6.5 2.  4.  7.5 8. ]
 [4.  5.  1.  4.5 0.5]]

```

</details>


)DOC")
    .Arg("axes", "(*Tuple(int)*): list of axes to reduce")
    .Arg(
        "keepdims",
        "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
    .Input(0, "X", "(*Tensor`<float>`*): input tensor")
    .Output(0, "Y", "(*Tensor`<float>`*): reduced tensor");

OPERATOR_SCHEMA(ReduceMeanGradient).NumInputs(3).NumOutputs(1);

namespace {

class GetReduceGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        std::vector<string>{GO(0), I(0), O(0)},
        std::vector<string>{GI(0)});
  }
};

} // namespace

REGISTER_GRADIENT(ReduceMin, GetReduceGradient);
REGISTER_GRADIENT(ReduceMax, GetReduceGradient);
REGISTER_GRADIENT(ReduceSum, GetReduceGradient);
REGISTER_GRADIENT(ReduceMean, GetReduceGradient);

} // namespace caffe2
