#pragma once

#include <type_traits>
#include <utility>
#include <memory>

/*
 * This header adds some utils with C++14 and C++17 functionality
 */

namespace c10 { namespace guts {

#if __cplusplus >= 201402L || defined(__cpp_lib_make_unique) && __cpp_lib_make_unique >= 201304L || \
  (defined(__ANDROID__) && __ANDROID__ && __cplusplus >= 201300L) || defined(_MSC_VER) && _MSC_VER >= 1900

/* using override */ using std::make_unique;

#else

// Implementation taken from folly
template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template <typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}
// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template <typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;

#endif


#ifdef __cpp_lib_integer_sequence

template<class T, T... Ints> using integer_sequence = std::integer_sequence<T, Ints...>;
template<std::size_t... Ints> using index_sequence = std::index_sequence<Ints...>;
template<class T, T N> using make_integer_sequence = std::make_integer_sequence<T, N>;
template<std::size_t N> using make_index_sequence = std::make_index_sequence<N>;
template<class... T> using index_sequence_for = std::index_sequence_for<T...>;

#else

template<class T, T... Ints> struct integer_sequence {
  using value_type = T;
  static constexpr std::size_t size() noexcept {return sizeof...(Ints);}
};
template<std::size_t... Ints> using index_sequence = integer_sequence<std::size_t, Ints...>;
namespace detail {
  template<class T, std::size_t I, std::size_t N, T... Ints>
  struct make_integer_sequence_ {
    using type = typename make_integer_sequence_<T, I+1, N, Ints..., I>::type;
  };
  template<class T, std::size_t N, T... Ints>
  struct make_integer_sequence_<T, N, N, Ints...> {
    using type = integer_sequence<T, Ints...>;
  };
}
template<class T, T N> using make_integer_sequence = typename detail::make_integer_sequence_<T, 0, N>::type;
template<std::size_t N> using make_index_sequence = make_integer_sequence<std::size_t, N>;
static_assert(std::is_same<index_sequence<>, make_index_sequence<0>>::value, "");
static_assert(std::is_same<index_sequence<0, 1, 2>, make_index_sequence<3>>::value, "");
template<class... T> using index_sequence_for = make_index_sequence<sizeof...(T)>;

#endif


#ifdef __cpp_lib_transformation_trait_aliases
template<bool B, class T, class F> using conditional_t = std::conditional_t<B, T, F>;
template<bool B, class T = void> using enable_if_t = std::enable_if_t<B, T>;
template<class T> using add_lvalue_reference_t = std::add_lvalue_reference_t<T>;
template<class T> using remove_reference_t = std::remove_reference_t<T>;
template<class T> using remove_cv_t = std::remove_cv_t<T>;
template<class T> using result_of_t = std::result_of_t<T>;
template<class T> using decay_t = std::decay_t<T>;
#else
template<bool B, class T, class F> using conditional_t = typename std::conditional<B, T, F>::type;
template<bool B, class T = void> using enable_if_t = typename std::enable_if<B, T>::type;
template<class T> using add_lvalue_reference_t = typename std::add_lvalue_reference<T>::type;
template<class T> using remove_reference_t = typename std::remove_reference<T>::type;
template<class T> using remove_cv_t = typename std::remove_cv<T>::type;
template<class T> using result_of_t = typename std::result_of<T>::type;
template<class T> using decay_t = typename std::decay<T>::type;
#endif



#ifdef __cpp_lib_logical_traits

using conjunction = std::conjunction;
using disjunction = std::disjunction;
using bool_constant = std::bool_constant;
using negation = std::negation;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/conjunction
template<class...> struct conjunction : std::true_type { };
template<class B1> struct conjunction<B1> : B1 { };
template<class B1, class... Bn>
struct conjunction<B1, Bn...>
    : conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

// Implementation taken from http://en.cppreference.com/w/cpp/types/disjunction
template<class...> struct disjunction : std::false_type { };
template<class B1> struct disjunction<B1> : B1 { };
template<class B1, class... Bn>
struct disjunction<B1, Bn...>
    : conditional_t<bool(B1::value), B1, disjunction<Bn...>>  { };

// Implementation taken from http://en.cppreference.com/w/cpp/types/integral_constant
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

// Implementation taken from http://en.cppreference.com/w/cpp/types/negation
template<class B>
struct negation : bool_constant<!bool(B::value)> { };

#endif



#ifdef __cpp_lib_void_t

template<class T> using void_t = std::void_t<T>;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/void_t
// (it takes CWG1558 into account and also works for older compilers)
template<typename... Ts> struct make_void { typedef void type;};
template<typename... Ts> using void_t = typename make_void<Ts...>::type;

#endif



#ifdef __cpp_lib_apply

using apply = std::apply;

#else

// Implementation from http://en.cppreference.com/w/cpp/utility/apply (but modified)
// TODO This is an incomplete implementation of std::apply, not working for member functions.
namespace detail {
template <class F, class Tuple, std::size_t... I>
constexpr auto apply_impl(F&& f, Tuple&& t, guts::index_sequence<I...>) -> decltype(std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...))
{
    return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
constexpr auto apply(F&& f, Tuple&& t) -> decltype(detail::apply_impl(
    std::forward<F>(f), std::forward<Tuple>(t),
    guts::make_index_sequence<std::tuple_size<guts::remove_reference_t<Tuple>>::value>{}))
{
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        guts::make_index_sequence<std::tuple_size<guts::remove_reference_t<Tuple>>::value>{});
}

#endif

}}
