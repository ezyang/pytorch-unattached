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

template<class... Args> auto getTensorTypeIds_(const Args&... args) {
  return guts::filter_map<TensorTypeId, is_tensor_arg>([] (const Tensor& t) { return t._to_impl()->type_id(); }, args...);
}

// TODO Test getTensorTypeIds_

template<class T, typename = void>
struct has_signature_defined : std::false_type {};
template<class T>
struct has_signature_defined<T, guts::void_t<
  typename T::Signature
>> : std::true_type {};

// TODO Test has_signature_defined

template<class OpSchemaDef> class OpSignatureSchema final {
  static_assert(details::has_signature_defined<OpSchemaDef>::value, "Operator schema doesn't define a valid Signature member type.");
  static_assert(guts::is_function_type<typename OpSchemaDef::Signature>::value, "Signature member of operator schema must be a function type.");

  using signature_traits = guts::function_traits<typename OpSchemaDef::Signature>;
public:
  using func_type = typename signature_traits::func_type;
  using return_type = typename signature_traits::return_type;
  using argument_types = typename signature_traits::argument_types;

  static constexpr size_t num_args = argument_types::size;
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, argument_types>::value;
};

template<class T, typename = void>
struct has_function_dispatchKeyForOpCalling_defined : std::false_type {};
template<class T>
struct has_function_dispatchKeyForOpCalling_defined<T, guts::void_t<
  decltype(&T::dispatchKeyForOpCalling)
>> : std::true_type {};

template<class T, typename = void>
struct has_function_dispatchKeyForOpRegistration_defined : std::false_type {};
template<class T>
struct has_function_dispatchKeyForOpRegistration_defined<T, guts::void_t<
  decltype(&T::dispatchKeyForOpRegistration)
>> : std::true_type {};

template<class OpSchemaDef, class Enable = void> class OpDispatchKeySchema final {
  // General case. Operator doesn't overwrite DispatchKey generation. Use default.
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(!has_function_dispatchKeyForOpCalling_defined<OpSchemaDef>::value, "Operator schema specifies a custom dispatchKeyForOpCalling function, but doesn't specify a custom DispatchKey type. Please specify it.");
  static_assert(!has_function_dispatchKeyForOpRegistration_defined<OpSchemaDef>::value, "Operator schema specifies a custom dispatchKeyForOpRegistration function, but doesn't specify a custom DispatchKey type. Please specify it.");

public:
  using dispatch_key_type = DispatchKey<signature::num_tensor_args>;

  template<class... Args>
  static inline DispatchKey<signature::num_tensor_args> dispatchKeyForOpCalling(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::argument_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatchKeyForOpCalling()");
    return DispatchKey<signature::num_tensor_args> {
      details::getTensorTypeIds_(args...)
    };
  }

  static inline constexpr DispatchKey<signature::num_tensor_args> dispatchKeyForOpRegistration(const std::array<TensorTypeId, signature::num_tensor_args>& tensorTypeIds) {
    return DispatchKey<signature::num_tensor_args> {
      std::move(tensorTypeIds)
    };
  }
};

template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, guts::void_t<typename OpSchemaDef::DispatchKey>> final {
  // Special case. Operator overwrites DispatchKey generation. Use that.
  static_assert(guts::is_equality_comparable<typename OpSchemaDef::DispatchKey>::value, "Operator schema specified custom dispatch key type, but that type doesn't have the equality operator defined. Please define it.");
  static_assert(guts::is_hashable<typename OpSchemaDef::DispatchKey>::value, "Operator schema specified custom dispatch key type, but that type doesn't have an overload for std::hash. Please define it.");

  static_assert(has_function_dispatchKeyForOpCalling_defined<OpSchemaDef>::value, "Operator schema specifies a custom DispatchKey type but is missing the dispatchKeyForOpCalling function to specify how to generate dispatch keys.");
  static_assert(has_function_dispatchKeyForOpRegistration_defined<OpSchemaDef>::value, "Operator schema specifies a custom DispatchKey type but is missing the dispatchKeyForOpRegistration function to specify how to generate dispatch keys.");

  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatchKeyForOpCalling)>::value, "Operator schema defines dispatchKeyForOpCalling, but it isn't a function.");
  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatchKeyForOpRegistration)>::value, "Operator schema defines dispatchKeyForOpRegistration, but it isn't a function.");

public:

  using dispatch_key_type = typename OpSchemaDef::DispatchKey;
};

}

template<class OpSchemaDef> class OpSchema final {
public:
  using signature = details::OpSignatureSchema<OpSchemaDef>;
  using dispatch = details::OpDispatchKeySchema<OpSchemaDef>;
};

// TODO Move to test cases
namespace test_opschema {
struct SchemaDef final {
  using Signature = bool (int, Tensor, float, Tensor, Tensor, unsigned int);
};
static_assert(6 == OpSchema<SchemaDef>::signature::num_args, "test num_tensor_args");
static_assert(3 == OpSchema<SchemaDef>::signature::num_tensor_args, "test num_tensor_args");
static_assert(std::is_same<bool, typename OpSchema<SchemaDef>::signature::return_type>::value, "test num_tensor_args");
static_assert(std::is_same<guts::typelist::typelist<int, Tensor, float, Tensor, Tensor, unsigned int>, typename OpSchema<SchemaDef>::signature::argument_types>::value, "test num_tensor_args");
}

}
