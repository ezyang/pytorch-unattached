#pragma once

#include <c10/guts/Metaprogramming.h>
#include <c10/dispatch/OpSchema.h>
#include <c10/dispatch/TensorTypeId.h>

#include <type_traits>
#include <array>
#include <unordered_map>
#include <iostream>
#include <shared_mutex>

namespace c10 {

/**
 * Per-operator dispatch table.
 *
 * Given an operator specified by 'OpSchemaDef', this class records a dispatch table for
 * various backends provided for this operator.  For example, if we consider the operator
 * add(Tensor, Tensor), the dispatch table for this operator may contain implementations
 * for various dynamic tensor types, such as (CPUFloatTensor, CPUFloatTensor),
 * (CUDAFloatTensor, CUDAFloatTensor), etc.
 *
 * @tparam OpSchemaDef The operator signature this dispatch table encodes.
 */
// TODO: Support dispatch for meta-operators (which apply to all dynamic types)
template<class OpSchemaDef>
class DispatchTable final {
private:
  using Schema = OpSchema<OpSchemaDef>;

public:
  DispatchTable(): ops_() {}

  /**
   * Register an operator implementation in the table at some dispatch key.
   * @param func Concrete function implementation to register
   * @param dispatch_key Dispatch key to implement this function with
   */
  void registerOp(typename Schema::signature::func_type* func, typename Schema::dispatch::dispatch_key_type dispatch_key) {
    std::unique_lock<std::shared_timed_mutex> lock(ops_mutex_);

    auto emplaced = ops_.emplace(std::move(dispatch_key), reinterpret_cast<void*>(func));
    if (!emplaced.second) {
      throw std::logic_error("Tried to register conflicting operators to the dispatcher.");
    }
  }

  /**
   * Unregister the operator implementation at some dispatch key.
   *
   * @param dispatch_key Dispatch key to unregister.
   */
  // TODO: This isn't going to work so well when we get more complicated override patterns!
  // In this case, an operator will show up in multiple slots, and erasing them one-by-one
  // is probably not such a good idea.
  void deregisterOp(const typename Schema::dispatch::dispatch_key_type& dispatch_key) {
    std::unique_lock<std::shared_timed_mutex> lock(ops_mutex_);

    auto found = ops_.find(dispatch_key);
    if (found == ops_.end()) {
      throw std::logic_error("Tried to deregister an operator that isn't registered.");
    }
    ops_.erase(found);
  }

  /**
   * Perform a dynamic dispatch on this table.
   *
   * @tparam Args Perfect forwarding template arguments to the dispatch
   * @param args Arguments to invoke the function with
   * @return Returned value of the operator
   */
  // ezyang to smessmer: It's a pity this has to be templated, because we technically already know
  // the argument type of this function (since this class is templated on OpSchemaDef).  Is there
  // really nothing we can do here?  Well, since it's perfect forwarding it should work OK for
  // most cases.
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
    // ezyang to smessmer: We will probably need to remove the read-side lock.  This will probably
    // necessitate replacing unordered_map with our own map implementation
    std::shared_lock<std::shared_timed_mutex> lock(ops_mutex_);

    auto dispatch_key = Schema::dispatch::dispatch_key(args...);
    auto found = ops_.find(dispatch_key);
    if (found == ops_.end()) {
      throw std::logic_error("Didn't find operator to dispatch to");
    }
    return reinterpret_cast<typename Schema::signature::func_type*>(found->second);
  }

  // TODO Use better hash map
  std::unordered_map<typename Schema::dispatch::dispatch_key_type, void*> ops_;
  mutable std::shared_timed_mutex ops_mutex_;
};

}

/*
 * Use this to access the dispatch table singleton for a given op schema.
 * It has an implementation for each op schema def in a cpp file, because
 * we can't rely on the one-definition-rule.
 */
template<class OpSchemaDef> c10::DispatchTable<OpSchemaDef>& c10_dispatch_table();
