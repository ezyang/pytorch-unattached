#pragma once

#include "OpId.h"
#include "TensorTypeId.h"

#include <vector>
#include <functional>

namespace c10 {

// TODO Implement DispatchKey in a better way
struct DispatchKey final {
  // TODO
  OpId opId_;
  std::vector<TensorTypeId> argTypes_;
};

inline bool operator==(const DispatchKey &lhs, const DispatchKey& rhs) {
  return lhs.opId_ == rhs.opId_ && lhs.argTypes_ == rhs.argTypes_;
}

}


namespace std {
  template<>
  struct hash<c10::DispatchKey> final {
    size_t operator()(const c10::DispatchKey& obj) const {
      size_t hash = std::hash<c10::OpId>()(obj.opId_);
      for (const auto& typeId : obj.argTypes_) {
        hash *= 10883; // prime
        hash += std::hash<c10::TensorTypeId>()(typeId);
      }
      return hash;
    }
  };
}
