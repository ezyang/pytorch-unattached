#pragma once

#include <type_traits>
#include <array>
#include <unordered_map>
#include <iostream>
#include <c10/guts/Metaprogramming.h>
#include "../OpSchema.h"
#include "../TensorTypeId.h"

namespace c10 {

template<class OpSchemaDef>
class DispatchTable final {
private:
  using Schema = OpSchema<OpSchemaDef>;

public:
  DispatchTable(): ops_() {}

  template<size_t num_tensor_args>
  void registerOp(typename Schema::func_type* func, const TensorTypeId (&tensorTypeIds)[num_tensor_args]) {
    static_assert(Schema::num_tensor_args == num_tensor_args, "Operator registration failed. Number of tensor type ids must match the number of tensor arguments in the operator signature.");
    registerOp_(func, Schema::dispatchKey(guts::to_std_array(tensorTypeIds)));
  }

  // overload for ops with zero tensor arguments (C arrays with size zero are invalid in C++, so they can't use the method above)
  void registerOp(typename Schema::func_type* func) {
    static_assert(Schema::num_tensor_args == 0, "Operator registration failed. Number of tensor type ids must match the number of tensor arguments in the operator signature.");
    registerOp_(func, Schema::dispatchKey());
  }

  template<class... Args>
  typename Schema::return_type call(Args&&... args) const {
    // TODO Better error message, but need to take care that reference arguments match non-reference arguments and so on.
    //      static_assert(std::is_same<typename Schema::return_type (Args...), typename Schema::func_type>::value, "Argument types don't match operator signature");
    auto operator_func = lookupOp_(args...);
    return operator_func(std::forward<Args>(args)...);
  }

private:
  template<class... Args>
  typename Schema::func_type* lookupOp_(const Args&... args) const {
    auto dispatchKey = Schema::dispatchKey(args...);
    auto found = ops_.find(dispatchKey);
    if (found == ops_.end()) {
      throw std::logic_error("Didn't find operator to dispatch to");
    }
    return reinterpret_cast<typename Schema::func_type*>(found->second);
  }

  void registerOp_(typename Schema::func_type* func, const typename Schema::dispatch_key_type& dispatchKey) {
    auto emplaced = ops_.emplace(dispatchKey, reinterpret_cast<void*>(func));
    if (!emplaced.second) {
      throw std::logic_error("Tried to register conflicting operators to the dispatcher.");
    }
  }

  // TODO Use better hash map
  std::unordered_map<typename Schema::dispatch_key_type, void*> ops_;
};

}

/*
 * Use this to access the dispatch table singleton for a given op schema.
 * It has an implementation for each op schema def in a cpp file, because
 * we can't rely on the one-definition-rule.
 */
template<class OpSchemaDef> c10::DispatchTable<OpSchemaDef>& c10_dispatch_table();
