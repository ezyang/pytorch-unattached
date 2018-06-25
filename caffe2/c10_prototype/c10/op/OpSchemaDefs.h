#pragma once

#include <c10/Tensor.h>
#include "caffe2/utils/Array.h"

namespace c10 { namespace ops {

//A(Tensor x, Tensor y);

template <typename... Args>
constexpr guts::array<const char*, sizeof...(Args)> parameters(Args... args) {
  return {args...};
};

struct equals final {
  using Signature = bool(Tensor, Tensor);
  static constexpr auto parameter_names = parameters("lhs", "rhs");
};

struct zeros final {
  using Signature = Tensor(ArrayRef<int64_t>, caffe2::TypeMeta);
  // ezyang: This is really long...
  static constexpr guts::array<const char*, 2> parameter_names = {{
    "shape", "dtype"
  }};
};

struct legacy_pytorch_resize_ final {
  using Signature = void(ArrayRef<int64_t> size, ArrayRef<int64_t> stride);

};

}}
