#pragma once

#include <type_traits>
#include <array>
#include "TypeList.h"

namespace c10 { namespace guts {


/*
 * Compile-time operations on std::array.
 * Only call these at compile time, they're slow if called at runtime.
 * Examples:
 *  equals({2, 3, 4}, {2, 3, 4}) == true  // needed because operator==(std::array, std::array) isn't constexpr
 *  tail({2, 3, 4}) == {3, 4}
 *  prepend(2, {3, 4}) == {2, 3, 4}
 */
namespace details {
template<class T, size_t N, size_t... I> struct eq__ final {};
template<class T, size_t N, size_t IHead, size_t... ITail> struct eq__<T, N, IHead, ITail...> final {
 static constexpr bool call(std::array<T, N> lhs, std::array<T, N> rhs) {
   return std::get<IHead>(lhs) == std::get<IHead>(rhs) && eq__<T, N, ITail...>::call(lhs, rhs);
 }
};
template<class T, size_t N> struct eq__<T, N> final {
 static constexpr bool call(std::array<T, N> /*lhs*/, std::array<T, N> /*rhs*/) {
   return true;
 }
};
template<class T, size_t N, size_t... I>
constexpr inline bool eq_(std::array<T, N> lhs, std::array<T, N> rhs, std::index_sequence<I...>) {
 return eq__<T, N, I...>::call(lhs, rhs);
}
}
template<class T, size_t N>
constexpr inline bool eq(std::array<T, N> lhs, std::array<T, N> rhs) {
 return details::eq_(lhs, rhs, std::make_index_sequence<N>());
}

namespace details {
template<class T, size_t N, size_t... I>
constexpr inline std::array<T, N-1> tail_(std::array<T, N> arg, std::index_sequence<I...>) {
  static_assert(sizeof...(I) == N-1, "invariant");
  return {{std::get<I+1>(arg)...}};
}
}
template<class T, size_t N>
constexpr inline std::array<T, N-1> tail(std::array<T, N> arg) {
  static_assert(N > 0, "Can only call tail() on an std::array with at least one element");
  return details::tail_(arg, std::make_index_sequence<N-1>());
}

namespace details {
template<class T, size_t N, size_t... I>
constexpr inline std::array<T, N+1> prepend_(T head, std::array<T, N> tail, std::index_sequence<I...>) {
  return {{head, std::get<I>(tail)...}};
}
}
template<class T, size_t N>
constexpr inline std::array<T, N+1> prepend(T head, std::array<T, N> tail) {
  return details::prepend_(head, tail, std::make_index_sequence<N>());
}

// TODO Move to test cases
namespace test_eq {
static_assert(eq(std::array<int, 3>{{2, 3, 4}}, std::array<int, 3>{{2, 3, 4}}), "test");
static_assert(!eq(std::array<int, 3>{{2, 3, 4}}, std::array<int, 3>{{2, 5, 4}}), "test");
}
namespace test_tail {
static_assert(eq(std::array<int, 2>{{3, 4}}, tail(std::array<int, 3>{{2, 3, 4}})), "test");
static_assert(eq(std::array<int, 0>{{}}, tail(std::array<int, 1>{{3}})), "test");
}
namespace test_prepend {
static_assert(eq(std::array<int, 3>{{2, 3, 4}}, prepend(2, std::array<int, 2>{{3, 4}})), "test");
static_assert(eq(std::array<int, 1>{{3}}, prepend(3, std::array<int, 0>{{}})), "test");
}

/*
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::argument_typle_type // A == tuple<float, double>
 */
template<class Func> struct function_traits {
  static_assert(!std::is_same<Func, Func>::value, "Can only use function_traits on function types");
};
template<class Result, class... Args>
struct function_traits<Result (Args...)> final {
  using func_type = Result (Args...);
  using return_type = Result;
  using argument_types = typelist::typelist<Args...>;
};

namespace test_function_traits {
static_assert(std::is_same<void, typename function_traits<void (int, float)>::return_type>::value, "test");
static_assert(std::is_same<int, typename function_traits<int (int, float)>::return_type>::value, "test");
static_assert(std::is_same<typelist::typelist<int, float>, typename function_traits<void (int, float)>::argument_types>::value, "test");
static_assert(std::is_same<typelist::typelist<int, float>, typename function_traits<int (int, float)>::argument_types>::value, "test");
}

namespace details {
template<class T, size_t N, size_t... I>
constexpr void assign_(std::array<T, N>& result, const T (&arr)[N], std::index_sequence<I...>) {
  // This is a trick to do "for each i: result[i] = arr[i]" at compile time.
  (void)std::initializer_list<int>{(std::get<I>(result) = arr[I], 0)...};
}
}

/*
 * Convert a C array into a std::array.
 */
template<class T, size_t N>
constexpr std::array<T, N> to_std_array(const T (&arr)[N]) {
  std::array<T, N> result{};

  details::assign_(result, arr, std::make_index_sequence<N>());

  return result;
}

namespace test_to_std_array {
constexpr int obj2[3] = {3, 5, 6};
static_assert(eq(std::array<int, 3>{{3, 5, 6}}, to_std_array(obj2)), "test");
static_assert(eq(std::array<int, 3>{{3, 5, 6}}, to_std_array({3, 5, 6})), "test");
}

}}
