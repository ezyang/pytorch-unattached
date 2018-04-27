#pragma once

#include <cstdint>
#include <ostream>
#include <functional>

namespace c10 {

// A compact identifier which stores all of the information necessary to
// carry out a dispatch on a type.  This is NOT NECESSARILY in one-to-one
// correspondence with the type hierarchy of TensorImpl, because we may decide
// that we want to refine dispatch on a runtime property of a tensor which is
// NOT reflected by the class hierarchy.
//
// Other note: there is no NATIVE notion of a subtyping relationship between
// these TypeIds.  We are planning to design one but we haven't decided on
// its specifics yet.
//
//    ezyang: CC smessmer; I know you wanted to have this line up exactly
//    with the concrete TensorImpl subclasses, but I don't want to commit
//    to that at the moment
//
// TODO: Does this also contain per Tensor properties, like contiguity?
//
// NB: NO default constructor
//
// dzhulgakov: have we considered just reusing TypeMeta directly? It allows for
// extensible backends more easily and is a pretty battle-tested piece of code.
class TypeId final {
  int64_t id_;
  const char* name_;

  explicit constexpr TypeId(int64_t id, const char* name) noexcept : id_(id), name_(name) {}

  friend class TypeIds;
public:
  // TODO: Not sure if this is done exactly correctly
  constexpr TypeId() noexcept : id_(0), name_("Undefined") {}
  constexpr TypeId(const TypeId& other) noexcept : id_(other.id_), name_(other.name_) {}
  constexpr TypeId(TypeId&& other) noexcept : id_(other.id_), name_(other.name_) {}
  constexpr TypeId& operator=(const TypeId& other) { id_ = other.id_; name_ = other.name_; return *this; }
  constexpr TypeId& operator=(TypeId&& other) { id_ = other.id_; name_ = other.name_; return *this; }

  constexpr int64_t id() const {
    return id_;
  }
  constexpr const char* name() const {
    return name_;
  }
};

constexpr inline bool operator ==(TypeId self, TypeId other) { return self.id() == other.id(); };

inline std::ostream& operator<<(std::ostream& out, TypeId id) {
  out << id.name();
  return out;
}

class TypeIds final {
public:
  // These are just here for illustrative purposes
  // NB: In C++11 you have to add these in TypeId.cpp too
  static constexpr TypeId Undefined = TypeId(0, "Undefined");
  static constexpr TypeId CPUTensor = TypeId(1, "CPUTensor");
  static constexpr TypeId CUDATensor = TypeId(2, "CUDATensor");
};

} // namespace c10

namespace std {
template<>
struct hash<c10::TypeId> final {
  size_t operator()(c10::TypeId obj) const {
    return std::hash<decltype(obj.id())>()(obj.id());
  }
};
}
