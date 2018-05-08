#pragma once

#include <c10/dispatch/TensorTypeId.h>

#include <vector>
#include <functional>

namespace c10 {

/**
 * The dispatch key encodes the runtime type identity of a function call arguments,
 * specifying what aspects of this identity can be dynamically dispatched on.
 *
 * Intuitively, given a function signature like f(Tensor, int), a valid dispatch
 * key for the arguments might be [CPUFloatTensor] (notice that 'f' is NOT included
 * in the dispatch key, and the runtime type of 'int' is NOT considered for dispatch
 * (since it is trivial).
 *
 * Dispatch keys permit equality tests and are hashable.
 *
 * @tparam num_dispatch_args The number of dispatchable arguments
 */
// ezyang to smessmer: You originally called this num_dispatch_args, but we might
// include things like dtype in this dispatch key, so I renamed it
template<size_t num_dispatch_args>
struct DispatchKey final {
  std::array<TensorTypeId, num_dispatch_args> argTypes;
};

template<size_t num_dispatch_args>
inline constexpr bool operator==(const DispatchKey<num_dispatch_args> &lhs, const DispatchKey<num_dispatch_args>& rhs) {
  // TODO: Use AVX instructions to perform this equality test more quickly
  return lhs.argTypes == rhs.argTypes;
}

}

namespace std {
  template<size_t num_dispatch_args>
  struct hash<c10::DispatchKey<num_dispatch_args>> final {
    // TODO constexpr hashing
    size_t operator()(const c10::DispatchKey<num_dispatch_args>& obj) const {
      size_t hash_value = 0;
      for (const auto& typeId : obj.argTypes) {
        hash_value *= 10883; // prime
        hash_value += std::hash<c10::TensorTypeId>()(typeId);
      }
      return hash_value;
    }
  };
}
