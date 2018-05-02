#pragma once

#include <type_traits>
#include <array>
#include <unordered_map>
#include <iostream>
#include <c10/guts/Metaprogramming.h>
#include "DispatchKey.h"
#include "OpSchema.h"
#include "TensorTypeId.h"

namespace c10 {

class Dispatcher final {
public:
  static Dispatcher& singleton();

  template<class OpSchemaDef, size_t num_tensor_args>
  void registerOp(typename OpSchema<OpSchemaDef>::func_type* func, const TensorTypeId (&tensorTypeIds)[num_tensor_args]) {
    static_assert(OpSchema<OpSchemaDef>::num_tensor_args == num_tensor_args, "Operator registration failed. Number of tensor type ids must match the number of tensor arguments in the operator signature.");
    registerOp_<OpSchemaDef>(func, OpSchema<OpSchemaDef>::dispatchKey(guts::to_std_array(tensorTypeIds)));
  }

  // overload for ops with zero tensor arguments (C arrays with size zero are invalid in C++, so they can't use the method above)
  template<class OpSchemaDef>
  void registerOp(typename OpSchema<OpSchemaDef>::func_type* func) {
    static_assert(OpSchema<OpSchemaDef>::num_tensor_args == 0, "Operator registration failed. Number of tensor type ids must match the number of tensor arguments in the operator signature.");
    registerOp_<OpSchemaDef>(func, OpSchema<OpSchemaDef>::dispatchKey());
  }

  template<class OpSchemaDef, class... Args>
  typename OpSchema<OpSchemaDef>::return_type call(Args... args) const {
    // TODO Perfect forwarding of arguments
    using Schema = OpSchema<OpSchemaDef>;
    static_assert(std::is_same<typename Schema::return_type (Args...), typename Schema::func_type>::value, "Argument types don't match operator signature");
    DispatchKey dispatchKey = Schema::dispatchKey(args...);
    auto found = ops_.find(dispatchKey);
    if (found == ops_.end()) {
      throw std::logic_error("Didn't find operator to dispatch to");
    }
    typename Schema::func_type* func = reinterpret_cast<typename Schema::func_type*>(found->second);
    return func(std::forward<Args>(args)...);
  }

private:
  Dispatcher();
  
  template<class OpSchemaDef>
  void registerOp_(typename OpSchema<OpSchemaDef>::func_type* func, const DispatchKey& dispatchKey) {
    auto emplaced = ops_.emplace(dispatchKey, reinterpret_cast<void*>(func));
    if (!emplaced.second) {
      throw std::logic_error("Tried to register conflicting operators to the dispatcher.");
    }
  }

  // TODO Use better hash map
  std::unordered_map<DispatchKey, void*> ops_;
};

}
