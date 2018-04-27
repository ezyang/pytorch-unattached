#pragma once

#include <functional>

namespace c10 {

// TODO Implement DispatchTypeId in a better way
struct DispatchTypeId final {
  int id_;
};

constexpr inline bool operator==(DispatchTypeId lhs, DispatchTypeId rhs) {
  return lhs.id_ == rhs.id_;
}

}

namespace std {
  template<>
  struct hash<c10::DispatchTypeId> final {
    size_t operator()(c10::DispatchTypeId obj) const {
      return std::hash<decltype(obj.id_)>()(obj.id_);
    }
  };
}
