#pragma once

#include <type_traits>
#include <array>
#include <functional>
#include "TypeList.h"

namespace c10 { namespace guts {


/**
 * Compile-time operations on std::array.
 * Only call these at compile time, they're slow if called at runtime.
 * Examples:
 *  equals({2, 3, 4}, {2, 3, 4}) == true  // needed because operator==(std::array, std::array) isn't constexpr
 *  tail({2, 3, 4}) == {3, 4}
 *  prepend(2, {3, 4}) == {2, 3, 4}
 */
namespace details {
template<class T, size_t N, size_t... I> struct eq__ {};
template<class T, size_t N, size_t IHead, size_t... ITail> struct eq__<T, N, IHead, ITail...> {
 static constexpr bool call(const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
   return std::get<IHead>(lhs) == std::get<IHead>(rhs) && eq__<T, N, ITail...>::call(lhs, rhs);
 }
};
template<class T, size_t N> struct eq__<T, N> {
 static constexpr bool call(const std::array<T, N>& /*lhs*/, const std::array<T, N>& /*rhs*/) {
   return true;
 }
};
template<class T, size_t N, size_t... I>
constexpr inline bool eq_(const std::array<T, N>& lhs, const std::array<T, N>& rhs, std::index_sequence<I...>) {
 return eq__<T, N, I...>::call(lhs, rhs);
}
}
template<class T, size_t N>
constexpr inline bool eq(const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
 return details::eq_(lhs, rhs, std::make_index_sequence<N>());
}

namespace details {
template<class T, size_t N, size_t... I>
constexpr inline std::array<T, N-1> tail_(const std::array<T, N>& arg, std::index_sequence<I...>) {
  static_assert(sizeof...(I) == N-1, "invariant");
  return {{std::get<I+1>(arg)...}};
}
}
template<class T, size_t N>
constexpr inline std::array<T, N-1> tail(const std::array<T, N>& arg) {
  static_assert(N > 0, "Can only call tail() on an std::array with at least one element");
  return details::tail_(arg, std::make_index_sequence<N-1>());
}

namespace details {
template<class T, size_t N, size_t... I>
constexpr inline std::array<T, N+1> prepend_(T head, const std::array<T, N>& tail, std::index_sequence<I...>) {
  return {{std::move(head), std::get<I>(tail)...}};
}
}
template<class T, size_t N>
constexpr inline std::array<T, N+1> prepend(T head, const std::array<T, N>& tail) {
  return details::prepend_(std::move(head), tail, std::make_index_sequence<N>());
}



/**
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::argument_typle_type // A == tuple<float, double>
 */
template<class Func> struct function_traits {
  static_assert(!std::is_same<Func, Func>::value, "Can only use function_traits on function types");
};
template<class Result, class... Args>
struct function_traits<Result (Args...)> {
  using func_type = Result (Args...);
  using return_type = Result;
  using parameter_types = typelist::typelist<Args...>;
};



/**
 * Convert a C array into a std::array.
 * Example:
 *   int source[3] = {2, 3, 4};
 *   std::array<int, 3> target = to_std_array(source);
 */

namespace details {
template<class T, size_t N, size_t... I>
constexpr std::array<T, N> to_std_array_(const T (&arr)[N], std::index_sequence<I...>) {
  return {{arr[I]...}};
}
}

template<class T, size_t N>
constexpr std::array<T, N> to_std_array(const T (&arr)[N]) {
  return details::to_std_array_(arr, std::make_index_sequence<N>());
}



/**
 * Use extract_arg_by_filtered_index to return the i-th argument whose
 * type fulfills a given type trait. The argument itself is perfectly forwarded.
 *
 * Example:
 * std::string arg1 = "Hello";
 * std::string arg2 = "World";
 * std::string&& result = extract_arg_by_filtered_index<is_string, 1>(0, arg1, 2.0, std::move(arg2));
 *
 * Warning: Taking the result by rvalue reference can cause segfaults because ownership will not be passed on
 *          from the original reference. The original reference dies after the expression and the resulting
 */
namespace details {
template<template <class> class Condition, size_t index, class Enable, class... Args> struct extract_arg_by_filtered_index_;
template<template <class> class Condition, size_t index, class Head, class... Tail>
struct extract_arg_by_filtered_index_<Condition, index, std::enable_if_t<!Condition<Head>::value>, Head, Tail...> {
  static decltype(auto) call(Head&& /*head*/, Tail&&... tail) {
    return extract_arg_by_filtered_index_<Condition, index, void, Tail...>::call(std::forward<Tail>(tail)...);
  }
};
template<template <class> class Condition, size_t index, class Head, class... Tail>
struct extract_arg_by_filtered_index_<Condition, index, std::enable_if_t<Condition<Head>::value && index != 0>, Head, Tail...> {
  static decltype(auto) call(Head&& /*head*/, Tail&&... tail) {
    return extract_arg_by_filtered_index_<Condition, index-1, void, Tail...>::call(std::forward<Tail>(tail)...);
  }
};
template<template <class> class Condition, size_t index>
struct extract_arg_by_filtered_index_<Condition, index, void> {
  static decltype(auto) call() {
    static_assert(index != index, "extract_arg_by_filtered_index out of range.");
  }
};
template<template <class> class Condition, size_t index, class Head, class... Tail>
struct extract_arg_by_filtered_index_<Condition, index, std::enable_if_t<Condition<Head>::value && index == 0>, Head, Tail...> {
  static decltype(auto) call(Head&& head, Tail&&... /*tail*/) {
    return std::forward<Head>(head);
  }
};
}
template<template <class> class Condition, size_t index, class... Args>
decltype(auto) extract_arg_by_filtered_index(Args&&... args) {
  return details::extract_arg_by_filtered_index_<Condition, index, void, Args...>::call(std::forward<Args>(args)...);
}



/**
 * Use filter_map to map a subset of the arguments to values.
 * The subset is defined by type traits, and will be evaluated at compile time.
 * At runtime, it will just loop over the pre-filtered arguments to create an std::array.
 *
 * Example:
 *  std::array<double, 2> result = filter_map<double, std::is_integral>([] (auto a) {return (double)a;}, 3, "bla", 4);
 *  // result == {3.0, 4.0}
 */
namespace details {

template<class ResultType, size_t num_results> struct filter_map_ {
   template<template <class> class Condition, class Mapper, class... Args, size_t... I>
   static std::array<ResultType, num_results> call(const Mapper& mapper, std::index_sequence<I...>, Args&&... args) {
     return std::array<ResultType, num_results> { mapper(extract_arg_by_filtered_index<Condition, I>(std::forward<Args>(args)...))... };
   }
};
template<class ResultType> struct filter_map_<ResultType, 0> {
  template<template <class> class Condition, class Mapper, class... Args, size_t... I>
  static std::array<ResultType, 0> call(const Mapper& /*mapper*/, std::index_sequence<I...>, Args&&... /*args*/) {
    return std::array<ResultType, 0> { };
  }
};
}

template<class ResultType, template <class> class Condition, class Mapper, class... Args> auto filter_map(const Mapper& mapper, Args&&... args) {
  static constexpr size_t num_results = typelist::count_if<Condition, typelist::typelist<Args...>>::value;
  return details::filter_map_<ResultType, num_results>::template call<Condition, Mapper, Args...>(mapper, std::make_index_sequence<num_results>(), std::forward<Args>(args)...);
}

}}
