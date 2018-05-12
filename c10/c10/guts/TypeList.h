#pragma once

#include "C++17.h"

namespace c10 { namespace guts { namespace typelist {

template<class... Items> struct typelist final {
  using tuple_type = std::tuple<Items...>;

  static constexpr size_t size = sizeof...(Items);
};

template<class Tuple> struct from_tuple;
template<class... Types> struct from_tuple<std::tuple<Types...>> final {
  using type = typelist<Types...>;
};
template<class Tuple> using from_tuple_t = typename from_tuple<Tuple>::type;

namespace test_from_tuple {
class MyClass {};
static_assert(std::is_same<typelist<int, float&, const MyClass&&>, from_tuple_t<std::tuple<int, float&, const MyClass&&>>>::value, "test");
}

template<class... TypeLists> struct concat;
template<class... Head1Types, class... Head2Types, class... TailLists>
struct concat<typelist<Head1Types...>, typelist<Head2Types...>, TailLists...> final {
  using type = typename concat<typelist<Head1Types..., Head2Types...>, TailLists...>::type;
};
template<class... HeadTypes>
struct concat<typelist<HeadTypes...>> final {
  using type = typelist<HeadTypes...>;
};
template<>
struct concat<> final {
  using type = typelist<>;
};
template<class... TypeLists> using concat_t = typename concat<TypeLists...>::type;

namespace test_concat {
class MyClass {};
static_assert(std::is_same<typelist<>, concat_t<>>::value, "test");
static_assert(std::is_same<typelist<>, concat_t<typelist<>>>::value, "test");
static_assert(std::is_same<typelist<>, concat_t<typelist<>, typelist<>>>::value, "test");
static_assert(std::is_same<typelist<int>, concat_t<typelist<int>>>::value, "test");
static_assert(std::is_same<typelist<int>, concat_t<typelist<int>, typelist<>>>::value, "test");
static_assert(std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>>>::value, "test");
static_assert(std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>, typelist<>>>::value, "test");
static_assert(std::is_same<typelist<int, float&>, concat_t<typelist<int>, typelist<float&>>>::value, "test");
static_assert(std::is_same<typelist<int, float&>, concat_t<typelist<>, typelist<int, float&>, typelist<>>>::value, "test");
static_assert(std::is_same<typelist<int, float&, const MyClass&&>, concat_t<typelist<>, typelist<int, float&>, typelist<const MyClass&&>>>::value, "test");
}

template<template <class> class Condition, class TypeList> struct filter;
template<template <class> class Condition, class Head, class... Tail>
struct filter<Condition, typelist<Head, Tail...>> final {
  using type = std::conditional_t<
    Condition<Head>::value,
    concat_t<typelist<Head>, typename filter<Condition, typelist<Tail...>>::type>,
    typename filter<Condition, typelist<Tail...>>::type
  >;
};
template<template <class> class Condition>
struct filter<Condition, typelist<>> final {
  using type = typelist<>;
};
template<template <class> class Condition, class TypeList>
using filter_t = typename filter<Condition, TypeList>::type;

namespace test_filter {
class MyClass {};
static_assert(std::is_same<typelist<>, filter_t<std::is_reference, typelist<>>>::value, "test");
static_assert(std::is_same<typelist<>, filter_t<std::is_reference, typelist<int, float, double, MyClass>>>::value, "test");
static_assert(std::is_same<typelist<float&, const MyClass&&>, filter_t<std::is_reference, typelist<int, float&, double, const MyClass&&>>>::value, "test");
}

template<template <class> class Condition, class TypeList>
struct count_if final {
  // TODO Direct implementation might be faster
  static constexpr size_t value = filter_t<Condition, TypeList>::size;
};

namespace test_count_if {
class MyClass final {};
static_assert(count_if<std::is_reference, typelist<int, bool&, const MyClass&&, float, double>>::value == 2, "Test count_if");
}

template<template <class> class Condition, class TypeList> struct true_for_each_type;
template<template <class> class Condition, class... Types>
struct true_for_each_type<Condition, typelist<Types...>> final
: guts::conjunction<Condition<Types>...> {};

namespace test_true_for_each_type {
class MyClass {};
static_assert(true_for_each_type<std::is_reference, typelist<int&, const float&&, const MyClass&>>::value, "test");
static_assert(!true_for_each_type<std::is_reference, typelist<int&, const float, const MyClass&>>::value, "test");
}


template<template <class> class Mapper, class TypeList> struct map;
template<template <class> class Mapper, class... Types>
struct map<Mapper, typelist<Types...>> final {
  using type = typelist<Mapper<Types>...>;
};
template<template <class> class Mapper, class TypeList>
using map_t = typename map<Mapper, TypeList>::type;

// TODO Test map_t

template<class TypeList> struct head;
template<class Head, class... Tail> struct head<typelist<Head, Tail...>> final {
  using type = Head;
};
template<class TypeList> using head_t = typename head<TypeList>::type;

// TODO Test head_t

namespace details {

}

template<class TypeList> struct reverse;
template<class Head, class... Tail> struct reverse<typelist<Head, Tail...>> final {
  using type = concat_t<typename reverse<typelist<Tail...>>::type, typelist<Head>>;
};
template<> struct reverse<typelist<>> final {
  using type = typelist<>;
};
template<class TypeList> using reverse_t = typename reverse<TypeList>::type;

namespace test_reverse {
class MyClass {};
static_assert(std::is_same<
  typelist<int, double, MyClass*, const MyClass&&>,
  reverse_t<typelist<const MyClass&&, MyClass*, double, int>>
>::value, "");
static_assert(std::is_same<
  typelist<>,
  reverse_t<typelist<>>
>::value, "");
}

namespace details {
template<class TypeList> struct map_types_to_values;
template<class... Types> struct map_types_to_values<typelist<Types...>> final {
  template<class Func>
  static std::tuple<Types...> call(Func&& func) {
    return { std::forward<Func>(func)(static_cast<Types*>(nullptr))... };
  }
};
}

template<class TypeList, class Func> typename TypeList::tuple_type map_types_to_values(Func&& func) {
  return details::map_types_to_values<TypeList>::call(std::forward<Func>(func));
}

// TODO Test map_types_to_values

}}}
