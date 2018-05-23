#pragma once

#include "C++17.h"

namespace c10 {
namespace guts {

/**
 * Extended type traits, these can for example be used in std::enable_if.
 *  is_equality_comparable_t<T>  // true iff equality operator is defined for T
 *  is_hashable_t<T>  // true iff std::hash has a specialisation for T
 *  is_function_type_t<T> // true iff T is a C function type (i.e. "Result(Args....)")
 *  is_instantiation_of_t<T, I> // true iff I is a template instantiation of T (e.g. vector<int> is an instantiation of vector)
 *  Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template<class T, class Enable = void> struct is_equality_comparable : std::false_type {};
template<class T> struct is_equality_comparable<T, void_t<decltype(std::declval<T&>() == std::declval<T&>())>> : std::true_type {};
template<class T> using is_equality_comparable_t = typename is_equality_comparable<T>::type;

template<class T, class Enable = void> struct is_hashable : std::false_type {};
template<class T> struct is_hashable<T, void_t<decltype(std::hash<T>()(std::declval<T&>()))>> : std::true_type {};
template<class T> using is_hashable_t = typename is_hashable<T>::type;

template<class T>
struct is_function_type : std::false_type {};
template<class Result, class... Args>
struct is_function_type<Result (Args...)> : std::true_type {};
template<class T> using is_function_type_t = typename is_function_type<T>::type;

template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template<template<class...> class Template, class T> using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;

}
}
