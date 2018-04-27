#pragma once

#include <functional>

namespace c10 {

// TODO Implement OpId in a better way
struct OpId final {
  int id_;
};

inline bool operator==(OpId lhs, OpId rhs) {
  return lhs.id_ == rhs.id_;
}

}

namespace std {
  template<>
  struct hash<c10::OpId> final {
    size_t operator()(c10::OpId obj) const {
      return std::hash<decltype(obj.id_)>()(obj.id_);
    }
  };
}
