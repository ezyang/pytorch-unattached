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

// General case. Operator doesn't overwrite DispatchKey generation. Use default.
template<class OpSchemaDef, class Enable = void> class OpDispatchKeySchema final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(!has_function_dispatchKeyForOpCalling_defined<OpSchemaDef>::value, "Operator schema specifies a custom dispatchKeyForOpCalling function, but doesn't specify a custom DispatchKey type. Please specify it.");
  static_assert(!has_function_dispatchKeyForOpRegistration_defined<OpSchemaDef>::value, "Operator schema specifies a custom dispatchKeyForOpRegistration function, but doesn't specify a custom DispatchKey type. Please specify it.");

public:
  using dispatch_key_type = DispatchKey<signature::num_tensor_args>;
  using registration_data_type = std::array<TensorTypeId, signature::num_tensor_args>;

  template<class... Args>
  static inline dispatch_key_type dispatchKeyForOpCalling(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::argument_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatchKeyForOpCalling()");
    return dispatch_key_type {
      details::getTensorTypeIds_(args...)
    };
  }

  static inline constexpr dispatch_key_type dispatchKeyForOpRegistration(const registration_data_type& tensorTypeIds) {
    // TODO static_assert(Schema::signature::num_tensor_args == num_tensor_args, "Operator registration failed. Number of tensor type ids must match the number of tensor arguments in the operator signature.");
    // TODO Also do checking of this kind for custom dispatch keys
    return dispatch_key_type {
      tensorTypeIds
    };
  }

  // overload for ops with zero tensor arguments (C arrays with size zero are invalid in C++, so they can't use the method above)
  /*void registerOp(typename Schema::signature::func_type* func) {
    static_assert(Schema::signature::num_tensor_args == 0, "Operator registration failed. Number of tensor type ids must match the number of tensor arguments in the operator signature.");
    return dispatch_key_type {};
  }*/
};

// Special case. Operator overwrites DispatchKey generation. Use that.
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, guts::void_t<typename OpSchemaDef::DispatchKey>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(guts::is_equality_comparable<typename OpSchemaDef::DispatchKey>::value, "Operator schema specified custom dispatch key type, but that type doesn't have the equality operator defined. Please define it.");
  static_assert(guts::is_hashable<typename OpSchemaDef::DispatchKey>::value, "Operator schema specified custom dispatch key type, but that type doesn't have an overload for std::hash. Please define it.");

  static_assert(has_function_dispatchKeyForOpCalling_defined<OpSchemaDef>::value, "Operator schema specifies a custom DispatchKey type but is missing the dispatchKeyForOpCalling function to specify how to generate dispatch keys.");
  static_assert(has_function_dispatchKeyForOpRegistration_defined<OpSchemaDef>::value, "Operator schema specifies a custom DispatchKey type but is missing the dispatchKeyForOpRegistration function to specify how to generate dispatch keys.");

  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatchKeyForOpCalling)>::value, "Operator schema defines dispatchKeyForOpCalling, but it isn't a function.");
  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatchKeyForOpRegistration)>::value, "Operator schema defines dispatchKeyForOpRegistration, but it isn't a function.");

  using dispatchKeyForOpCalling_traits = guts::function_traits<decltype(OpSchemaDef::dispatchKeyForOpCalling)>;
  using dispatchKeyForOpRegistration_traits = guts::function_traits<decltype(OpSchemaDef::dispatchKeyForOpRegistration)>;

  static_assert(std::is_same<
    guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename dispatchKeyForOpCalling_traits::argument_types>>,
    guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename signature::argument_types>>
    >::value, "Operator schema defines dispatchKeyForOpCalling, but the arguments don't match the operator signature.");
  static_assert(std::is_same<
    typename dispatchKeyForOpCalling_traits::return_type,
    typename OpSchemaDef::DispatchKey
    >::value, "Operator schema defines dispatchKeyForOpCalling, but the return value doesn't match the defined DispatchKey type.");

  static_assert(std::is_same<
    typename dispatchKeyForOpRegistration_traits::return_type,
    typename OpSchemaDef::DispatchKey
    >::value, "Operator schema defines dispatchKeyForOpRegistration, but the return value doesn't match the defined DispatchKey type.");

public:

  using dispatch_key_type = typename OpSchemaDef::DispatchKey;
  using registration_data_type = typename dispatchKeyForOpRegistration_traits::argument_types::tuple_type;

  template<class... Args>
  static inline dispatch_key_type dispatchKeyForOpCalling(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::argument_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatchKeyForOpCalling()");
    return OpSchemaDef::dispatchKeyForOpCalling(args...);
  }

  static inline constexpr dispatch_key_type dispatchKeyForOpRegistration(const registration_data_type& registration_data) {
    return guts::apply(&OpSchemaDef::dispatchKeyForOpRegistration, registration_data);
  }
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

// TODO test OpSchema::dispatch stuff
}

}
