#pragma once

#include "DispatchKey.h"
#include <c10/guts/Metaprogramming.h>
#include <c10/Tensor.h>

namespace c10 {

namespace details {
template<class Arg> using is_valid_tensor_arg = std::is_same<Tensor, Arg>;
template<class Arg> using is_valid_nontensor_arg = guts::conjunction<
  guts::negation<is_valid_tensor_arg<Arg>>,
  std::is_same<std::remove_cv_t<std::remove_reference_t<Arg>>, Arg>
>;
template<class Arg> using is_valid_arg = guts::disjunction<
  is_valid_tensor_arg<Arg>,
  is_valid_nontensor_arg<Arg>
>;

namespace test_is_valid_arg {
static_assert(is_valid_tensor_arg<Tensor>::value, "");
static_assert(!is_valid_tensor_arg<int>::value, "");
static_assert(!is_valid_nontensor_arg<Tensor>::value, "");
static_assert(is_valid_nontensor_arg<int>::value, "");
static_assert(is_valid_arg<int>::value, "");
static_assert(is_valid_arg<Tensor>::value, "");
}
}

namespace details {
template<size_t index, class Enable, class... Args> struct getTensorTypeId__;
template<size_t index, class Head, class... Tail>
struct getTensorTypeId__<index, std::enable_if_t<!is_valid_tensor_arg<Head>::value>, Head, Tail...> final {
  static TensorTypeId call(const Head& /*head*/, const Tail&... tail) {
    return getTensorTypeId__<index, void, Tail...>::call(tail...);
  }
};
template<size_t index, class Head, class... Tail>
struct getTensorTypeId__<index, std::enable_if_t<is_valid_tensor_arg<Head>::value && index != 0>, Head, Tail...> final {
  static TensorTypeId call(const Head& /*head*/, const Tail&... tail) {
    return getTensorTypeId__<index - 1, void, Tail...>::call(tail...);
  }
};
template<size_t index, class Head, class... Tail>
struct getTensorTypeId__<index, std::enable_if_t<is_valid_tensor_arg<Head>::value && index == 0>, Head, Tail...> final {
  static TensorTypeId call(const Head& head, const Tail&... /*tail*/) {
    return head._to_impl()->type_id();
  }
};

template<class... Args, size_t... I> std::vector<TensorTypeId> getTensorTypeIds__(std::index_sequence<I...>, const Args&... args) {
  return { getTensorTypeId__<I, void, Args...>::call(args...)... };
}

template<class... Args> std::vector<TensorTypeId> getTensorTypeIds_(const Args&... args) {
  static constexpr size_t num_tensor_args = guts::typelist::count_if<is_valid_tensor_arg, guts::typelist::typelist<Args...>>::value;
  return getTensorTypeIds__(std::make_index_sequence<num_tensor_args>(), args...);
}

// TODO Test getTensorTypeIds_

template<class T>
struct is_operator_function_type final : std::false_type {};
template<class Result, class... Args>
struct is_operator_function_type<Result (Args...)> final : std::true_type {};

// TODO Test is_operator_function_type

template<class T, typename = void>
struct has_signature_defined final : std::false_type {};
template<class T>
struct has_signature_defined<T, guts::void_t<
  typename T::Signature
>> final : std::true_type {};

// TODO Test has_signature_defined

}


template<class OpSchemaDef> class OpSchema final {
private:
  static_assert(details::has_signature_defined<OpSchemaDef>::value, "Given operator schema doesn't define a valid Signature member type.");
  static_assert(details::is_operator_function_type<typename OpSchemaDef::Signature>::value, "Signature member of operator schema must be a function type.");

  using signature_traits = guts::function_traits<typename OpSchemaDef::Signature>;
public:
  using func_type = typename signature_traits::func_type;
  using return_type = typename signature_traits::return_type;
  using argument_types = typename signature_traits::argument_types;

  static constexpr size_t num_args = argument_types::size;
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_valid_tensor_arg, argument_types>::value;

  static_assert(guts::typelist::true_for_each_type<details::is_valid_arg, argument_types>::value, "Invalid argument type in operator signature. Arguments must be passed by value.");
  static_assert(details::is_valid_arg<return_type>::value, "Invalid result type in operator signature. Must return by value.");

  template<class... Args>
  static inline DispatchKey dispatchKey(const Args&... args) {
    // TODO pass to OpSchemaDef::dispatchKey if defined

    static_assert(std::is_same<guts::typelist::typelist<Args...>, argument_types>::value, "Invalid argument types passed to OpSchema::dispatchKey()");
    std::vector<TensorTypeId> dispatchTypeIds = details::getTensorTypeIds_(args...);
    return DispatchKey {
      OpSchemaDef::op_id(),
      std::move(dispatchTypeIds)
    };
  }

  static inline DispatchKey dispatchKey() {
    static_assert(OpSchema<OpSchemaDef>::num_tensor_args == 0, "DispatchKey can only be generated without tensor arguments if the OpSchemaDef doesn't have tensor arguments.");
    return DispatchKey {
      OpSchemaDef::op_id(),
      {}
    };
  }

  static inline DispatchKey dispatchKey(const std::array<TensorTypeId, num_tensor_args>& tensorTypeIds) {
    // TODO pass to OpSchemaDef::dispatchKey if defined
    return DispatchKey {
      OpSchemaDef::op_id(),
      std::vector<TensorTypeId>(tensorTypeIds.begin(), tensorTypeIds.end())
    };
  }
};

// TODO Move to test cases
namespace test_opschema {
struct SchemaDef final {
  using Signature = bool (int, Tensor, float, Tensor, Tensor, unsigned int);
};
static_assert(6 == OpSchema<SchemaDef>::num_args, "test num_tensor_args");
static_assert(3 == OpSchema<SchemaDef>::num_tensor_args, "test num_tensor_args");
static_assert(std::is_same<bool, typename OpSchema<SchemaDef>::return_type>::value, "test num_tensor_args");
static_assert(std::is_same<guts::typelist::typelist<int, Tensor, float, Tensor, Tensor, unsigned int>, typename OpSchema<SchemaDef>::argument_types>::value, "test num_tensor_args");
}

}
