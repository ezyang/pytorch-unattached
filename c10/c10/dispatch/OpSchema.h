#pragma once

#include "impl/DispatchKey.h"
#include <c10/guts/Metaprogramming.h>
#include <c10/Tensor.h>

namespace c10 {

namespace details {
template<class Arg> using is_tensor_arg = std::is_same<Tensor, std::remove_cv_t<std::remove_reference_t<Arg>>>;

namespace test_is_tensor_arg {
static_assert(is_tensor_arg<Tensor>::value, "");
static_assert(is_tensor_arg<const Tensor&>::value, "");
static_assert(is_tensor_arg<Tensor&&>::value, "");
static_assert(!is_tensor_arg<int>::value, "");
}
}

namespace details {

template<class... Args> auto getTensorTypeIds_(const Args&... args) {
  return guts::filter_map<TensorTypeId, is_tensor_arg>([] (const Tensor& t) { return t._to_impl()->type_id(); }, args...);
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
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, argument_types>::value;

  // TODO using dispatch_key_type = typename OpSchema::DispatchKey;
  using dispatch_key_type = DispatchKey<num_tensor_args>;

  template<class... Args>
  static inline DispatchKey<num_tensor_args> dispatchKey(const Args&... args) {
    // TODO pass to OpSchemaDef::dispatchKey if defined
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, argument_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatchKey()");
    return DispatchKey<num_tensor_args> {
      details::getTensorTypeIds_(args...)
    };
  }

  static inline DispatchKey<num_tensor_args> dispatchKey() {
    static_assert(OpSchema<OpSchemaDef>::num_tensor_args == 0, "DispatchKey can only be generated without tensor arguments if the OpSchemaDef doesn't have tensor arguments.");
    return DispatchKey<num_tensor_args> {
      std::array<TensorTypeId, 0>{}
    };
  }

  static inline constexpr DispatchKey<num_tensor_args> dispatchKey(const std::array<TensorTypeId, num_tensor_args>& tensorTypeIds) {
    // TODO pass to OpSchemaDef::dispatchKey if defined
    return DispatchKey<num_tensor_args> {
      std::move(tensorTypeIds)
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
