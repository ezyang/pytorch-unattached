#pragma once

#include "impl/DispatchKey.h"
#include <c10/guts/Metaprogramming.h>
#include <c10/Tensor.h>

namespace caffe2 {
template<class Context> class Tensor;
class CPUContext;
class CUDAContext;
}

namespace c10 {

// TODO Get rid of CAFFE2_CPU_TENSOR and CAFFE2_CUDA_TENSOR once the caffe2 tensor type is gone
C10_DECLARE_TENSOR_TYPE(CAFFE2_CPU_TENSOR)
C10_DECLARE_TENSOR_TYPE(CAFFE2_CUDA_TENSOR)

namespace details {

/**
 * If Arg is a Tensor or reference to a Tensor, provide the member constant value equal to true.  Otherwise
 * return false.
 */
template<class Arg> using is_tensor_arg = guts::disjunction<
  std::is_same<Tensor, std::remove_cv_t<std::remove_reference_t<Arg>>>,
  guts::is_instantiation_of<caffe2::Tensor, std::remove_cv_t<std::remove_reference_t<Arg>>>
>;

// TODO get rid of tensor_to_dispatch_key once c2::Tensor is not used anymore, this then fits into a template lambda instead of a functor.
template<class TensorType, class Enable = void> struct tensor_to_dispatch_key_ final {};
template<>
struct tensor_to_dispatch_key_<c10::Tensor, void> final {
    static TensorParameterDispatchKey call(const c10::Tensor& tensor) {
      auto *impl = tensor._to_impl();
      return TensorParameterDispatchKey{impl->type_id(), impl->dtype().id()};
    }
};
template<class TensorType>
struct tensor_to_dispatch_key_<TensorType, std::enable_if_t<std::is_same<TensorType, caffe2::Tensor<caffe2::CPUContext>>::value>> final {
    static TensorParameterDispatchKey call(const TensorType& tensor) {
      return TensorParameterDispatchKey{CAFFE2_CPU_TENSOR(), tensor.meta().id()};
    }
};

template<class TensorType>
struct tensor_to_dispatch_key_<TensorType, std::enable_if_t<std::is_same<TensorType, caffe2::Tensor<caffe2::CUDAContext>>::value>> final {
    static TensorParameterDispatchKey call(const TensorType& tensor) {
      return TensorParameterDispatchKey{CAFFE2_CUDA_TENSOR(), tensor.meta().id()};
    }
};
struct tensor_to_dispatch_key final {
    template<class TensorType>
    TensorParameterDispatchKey operator()(const TensorType& tensor) const {
      return tensor_to_dispatch_key_<TensorType, void>::call(tensor);
    }
};

/**
 * Extract the type ids of all tensors in a variadic list of arguments
 *
 * @tparam Args Inferred variadic list of argument types
 * @param args List of arguments to get type ids from
 * @return std::array<TypeId, n>, where n is the number of tensor arguments (is_tensor_arg) in the class
 */
template<class... Args> auto getTensorTypeIds_(const Args&... args) {
  return guts::filter_map<TensorParameterDispatchKey, is_tensor_arg>(tensor_to_dispatch_key(), args...);
}

// TODO Test getTensorTypeIds_

/**
 * If T is a struct with a type field Signature, provides the member constant
 * @tparam T
 */
template<class T, typename = void>
struct has_signature_defined : std::false_type {};
template<class T>
struct has_signature_defined<T, guts::void_t<
  typename T::Signature
>> : std::true_type {};

// TODO Test has_signature_defined

template<class T, typename = void>
struct has_parameter_names_defined : std::false_type {};
template<class T>
struct has_parameter_names_defined<T, guts::void_t<
  decltype(T::parameter_names)
>> : std::true_type {};

// TODO Test has_parameter_names_defined

/**
 * Wrapper class around a user-provided schema definition some useful information about the schema.
 *
 * @tparam OpSchemaDef Operator schema definition.  See OpSchema for more details.
 */
template<class OpSchemaDef> class OpSignatureSchema final {
  static_assert(details::has_signature_defined<OpSchemaDef>::value, "Operator schema doesn't define a valid Signature member type.");
  static_assert(guts::is_function_type<typename OpSchemaDef::Signature>::value, "Signature member of operator schema must be a function type.");

  using signature_traits = guts::function_traits<typename OpSchemaDef::Signature>;
public:
  /**
   * The function type OpSchemaDef::Signature
   */
  using func_type = typename signature_traits::func_type;
  /**
   * The return type of the function OpSchemaDef::Signature
   */
  using return_type = typename signature_traits::return_type;
  /**
   * A type list of the parameter types of OpSchemaDef::Signature
   */
  using parameter_types = typename signature_traits::parameter_types;

  /**
   * The number of arguments of OpSchemaDef::Signature
   */
  static constexpr size_t num_args = guts::typelist::size<parameter_types>::value;
  /**
   * The number of tensor arguments (as per is_tensor_arg) in OpSchemaDef::Signature
   */
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, parameter_types>::value;

private:
  static_assert(details::has_parameter_names_defined<OpSchemaDef>::value, "Operator schema doesn't define parameter_names member.");
  // TODO Allow simpler definition of parameter_names without having to spell out the std::array type in the schema def.
  static_assert(std::is_same<const std::array<const char*, num_args>, decltype(OpSchemaDef::parameter_names)>::value, "Operator schema defines parameter_names member, but it isn't the correct type. Must be a static constexpr std::array of const char* with one entry for each parameter.");

public:
  /**
   * The names of the parameters (as per OpSchemaDef::parameter_names)
   * @return Array
   */
  static constexpr const std::array<const char*, num_args>& parameter_names() {
    return OpSchemaDef::parameter_names;
  }
};

/**
 * If T has a method dispatch_key, provide a member constant value equal to true.  Otherwise return false.
 * @tparam T
 */
template<class T, typename = void>
struct has_function_dispatch_key_defined : std::false_type {};
template<class T>
struct has_function_dispatch_key_defined<T, guts::void_t<
  decltype(&T::dispatch_key)
>> : std::true_type {};

/**
 * Wrapper class around a user-defined schema definition providing a way of computing a dispatch key
 * from arguments matching the signature of that schema.
 *
 * @tparam OpSchemaDef Operator schema definition.  See OpSchema for more details.
 * @tparam Enable Inferred, used to control specialization
 */
template<class OpSchemaDef, class Enable = void> class OpDispatchKeySchema final {};

// General case. Operator doesn't overwrite DispatchKey generation. Use default.
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, std::enable_if_t<!has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

public:
  using dispatch_key_type = DispatchKey<signature::num_tensor_args>;

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return dispatch_key_type {
      details::getTensorTypeIds_(args...)
    };
  }
};

// Special case. Operator overwrites DispatchKey generation. Use that.
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, std::enable_if_t<has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatch_key)>::value, "Operator schema defines dispatch_key member, but it isn't a function.");

  using dispatch_key_traits = guts::function_traits<decltype(OpSchemaDef::dispatch_key)>;

public:
  using dispatch_key_type = typename dispatch_key_traits::return_type;

private:

  static_assert(guts::is_equality_comparable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have the equality operator defined. Please define it.");
  static_assert(guts::is_hashable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have an overload for std::hash. Please define it.");

  static_assert(std::is_same<
    guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename dispatch_key_traits::parameter_types>>,
    guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename signature::parameter_types>>
    >::value, "Operator schema defines custom dispatch_key() derivation function, but the arguments don't match the operator signature.");

public:

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return OpSchemaDef::dispatch_key(args...);
  }
};

}

/**
 * Wrapper class for user-defined OpSchemaDef, providing functionality for determining
 * information about the signature and dispatching on that signature.  This is the
 * "public" facing class.
 *
 * @tparam OpSchemaDef User-defined OpSchemaDef.
 *   This struct is expected to define:
 *      - a function type Signature
 *      - a constexpr std::array<const char*, n_args> parameter_names field (where n_args is
 *        the number of arguments in Signature)
 */
template<class OpSchemaDef> class OpSchema final {
  // TODO static_assert OpSchemaDef isn't an instanciation of OpSchema. If yes, the caller probably passed an OpSchema somewhere where an OpSchemaDef was expected.
public:
  /**
   * Information about the signature
   */
  using signature = details::OpSignatureSchema<OpSchemaDef>;
  /**
   * Functionality for dispatching on that signature
   */
  using dispatch = details::OpDispatchKeySchema<OpSchemaDef>;
};

// TODO test OpSchema::dispatch stuff
}
