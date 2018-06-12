#pragma once

#include "caffe2/core/typeid.h"
#include "Tensor.h"

namespace c10 {

/**
 * This class supports a fixed set of universal keyword arguments which commonly occur in many
 * functions in C10.  We can implement it efficiently because it only supports certain arguments.
 *
 * This class is IMMUTABLE.
 *
 * This class is NOT intended to be a dumping ground for arbitrary keyword arguments (it's stack allocated,
 * so you pay in stack space for every argument you pass this way).
 *
 * This class MAY participate in dispatch.
 */
class KeywordArgs final {
  TypeMeta dtype_;
  Tensor out_;
public:
  KeywordArgs() = default;
  KeywordArgs(const KeywordArgs &rhs) = default;
  KeywordArgs(KeywordArgs &&rhs) noexcept = default;
  KeywordArgs& operator=(KeywordArgs &&rhs) = default;
  KeywordArgs& operator=(const KeywordArgs &rhs) = default;

  TypeMeta dtype() const { return dtype_; }
  Tensor out() const { return out_; }

  // Modern compilers are able to optimize away the copies.  See:
  // https://godbolt.org/g/fqErkU

  KeywordArgs dtype(TypeMeta dtype) const {
    KeywordArgs args(*this);
    args.dtype_ = dtype;
    return args;
  }

  KeywordArgs out(Tensor out) const {
    KeywordArgs args(*this);
    args.out_ = out;
    return args;
  }
};

} // namespace c10
