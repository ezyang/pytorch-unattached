#pragma once

#include "../TensorTypeId.h"

#include <vector>
#include <functional>

namespace c10 {

template<size_t num_tensor_args>
struct DispatchKey final {
  std::array<TensorTypeId, num_tensor_args> argTypes;
};

template<size_t num_tensor_args>
inline constexpr bool operator==(const DispatchKey<num_tensor_args> &lhs, const DispatchKey<num_tensor_args>& rhs) {
  return lhs.argTypes == rhs.argTypes;
}

}


namespace std {
  template<size_t num_tensor_args>
  struct hash<c10::DispatchKey<num_tensor_args>> final {
    // TODO constexpr hashing
    size_t operator()(const c10::DispatchKey<num_tensor_args>& obj) const {
      size_t hash = 0;
      for (const auto& typeId : obj.argTypes) {
        hash *= 10883; // prime
        hash += std::hash<c10::TensorTypeId>()(typeId);
      }
      return hash;
    }
  };
}
