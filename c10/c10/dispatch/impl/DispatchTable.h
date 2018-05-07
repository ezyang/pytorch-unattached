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

  void registerOp(typename Schema::signature::func_type* func, const typename Schema::dispatch::registration_data_type& registration_data) {
    registerOp_(func, Schema::dispatch::dispatchKeyForOpRegistration(registration_data));
  }

  template<class... Args>
  typename Schema::signature::return_type call(Args&&... args) const {
    // TODO Better error message, but need to take care that reference arguments match non-reference arguments and so on.
    //      static_assert(std::is_same<typename Schema::return_type (Args...), typename Schema::func_type>::value, "Argument types don't match operator signature");
    auto operator_func = lookupOp_(args...);
    return operator_func(std::forward<Args>(args)...);
  }

private:
  template<class... Args>
  typename Schema::signature::func_type* lookupOp_(const Args&... args) const {
    auto dispatchKey = Schema::dispatch::dispatchKeyForOpCalling(args...);
    auto found = ops_.find(dispatchKey);
    if (found == ops_.end()) {
      throw std::logic_error("Didn't find operator to dispatch to");
    }
    return reinterpret_cast<typename Schema::signature::func_type*>(found->second);
  }

  void registerOp_(typename Schema::signature::func_type* func, const typename Schema::dispatch::dispatch_key_type& dispatchKey) {
    auto emplaced = ops_.emplace(dispatchKey, reinterpret_cast<void*>(func));
    if (!emplaced.second) {
      throw std::logic_error("Tried to register conflicting operators to the dispatcher.");
    }
  }

  // TODO Use better hash map
  std::unordered_map<typename Schema::dispatch::dispatch_key_type, void*> ops_;
};

}

/*
 * Use this to access the dispatch table singleton for a given op schema.
 * It has an implementation for each op schema def in a cpp file, because
 * we can't rely on the one-definition-rule.
 */
template<class OpSchemaDef> c10::DispatchTable<OpSchemaDef>& c10_dispatch_table();
