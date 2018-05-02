#pragma once

#include <c10/dispatch/OpId.h>
#include <c10/Tensor.h>

namespace c10 { namespace ops {

struct equals final {
  static constexpr OpId op_id() {return OpId{0}; }
  using Signature = bool(Tensor, Tensor);
};

struct zeros final {
  static constexpr OpId op_id() {return OpId{1}; }
  using Signature = Tensor(ArrayRef<int64_t>, DataType);
};

}}
