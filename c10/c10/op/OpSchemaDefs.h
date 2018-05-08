#pragma once

#include <c10/dispatch/OpId.h>
#include <c10/Tensor.h>

namespace c10 { namespace ops {

struct equals final {
  using Signature = bool(Tensor, Tensor);
  static constexpr std::array<const char*, 2> parameter_names = {
    "lhs", "rhs"
  };
};

struct zeros final {
  using Signature = Tensor(ArrayRef<int64_t>, DataType);
  static constexpr std::array<const char*, 2> parameter_names = {
    "shape", "dtype"
  };
};

}}
