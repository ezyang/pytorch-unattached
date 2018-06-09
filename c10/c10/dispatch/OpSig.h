#pragma once

#include <typeinfo>
#include <array>
#include <utility>
#include "caffe2/utils/C++17.h"
#include "caffe2/utils/Array.h"

namespace c10 {

template <typename T>
constexpr int type_meta();

template <>
constexpr int type_meta<int>() { return 1; }

using TypeMeta = int;

struct ArgSig final {
  constexpr ArgSig() : ty(0), name("") {}
  constexpr ArgSig(TypeMeta ty_, const char* name_) : ty(ty_), name(name_) {};
  TypeMeta ty;
  const char* name;
};

// TODO: Did we already have a copy of this?
template <typename T, std::size_t N, std::size_t... I>
constexpr guts::array<T, N + 1>
append_aux(guts::array<T, N> a, T t, guts::index_sequence<I...>) {
  return guts::array<T, N + 1>{ a[I]..., t };
}
template <typename T, std::size_t N>
constexpr guts::array<T, N + 1> append(guts::array<T, N> a, T t) {
  return append_aux(a, t, guts::make_index_sequence<N>());
}

template <std::size_t N = 0>
struct OpSig final {
  guts::array<ArgSig, N> args;
  constexpr OpSig() {}
  constexpr OpSig(guts::array<ArgSig, N> args_) : args(std::move(args_)) {}
  template <typename T>
  constexpr OpSig<N+1> arg(const char* name) {
    return OpSig<N+1>(append(args, ArgSig(type_meta<T>(), name)));
  };
};

}
