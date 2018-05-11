#pragma once

#include <c10/dispatch/OpId.h>
#include <c10/Tensor.h>

namespace c10 { namespace ops {

//A(Tensor x, Tensor y);

template <typename... Args>
constexpr std::array<const char*, sizeof...(Args)> parameters(Args... args) {
  return {args...};
};

struct equals final {
  using Signature = bool(Tensor, Tensor);
  static constexpr auto parameter_names = parameters("lhs", "rhs");
};

struct zeros final {
  using Signature = Tensor(ArrayRef<int64_t>, TypeMeta);
  // ezyang: This is really long...
  static constexpr std::array<const char*, 2> parameter_names = {
    "shape", "dtype"
  };
};

struct legacy_pytorch_resize_ final {
  using Signature = void(ArrayRef<int64_t> size, ArrayRef<int64_t> stride);

};

}}
