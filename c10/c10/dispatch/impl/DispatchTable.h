#pragma once

#include <flat_hash_map/flat_hash_map.h>
#include <c10/guts/Metaprogramming.h>
#include <c10/dispatch/OpSchema.h>
#include <c10/dispatch/TensorTypeId.h>

#include <type_traits>
#include <array>
#include <unordered_map>
#include <iostream>
#include <mutex>

namespace c10 {

namespace details {

// Internal implementation of the operator as a thread-safe hash table.
//
// The current implementation below does not have the correct correctness characteristics
// which we need.  It's worth spelling out exactly what we need:
//
//  - We need LOCK FREE read access to the table (as per the performance benchmark
//    at https://fb.quip.com/hvz3AGnx8MQ8
//
//  - We need to support writes which are possibly concurrent with reads, occurring when
//    a dynamic library is loaded or unloaded.
//
//  - We probably can require that dynamic library loads/unloads be synchronized (so
//    there are never two concurrent loads.)
template<class Key>
class ThreadsafeOperatorTable_ final {
public:
    template<class Key_>
    void emplace(Key_&& key, void* value) {
      // TODO Locking
      //std::unique_lock<std::shared_timed_mutex> lock(mutex_);

      auto result = map_.emplace(std::forward<Key>(key), value);
      if (!result.second) {
        throw std::logic_error("Tried to register conflicting operators to the dispatcher.");
      }
    }

    void erase(const Key& key) {
      // TODO Locking
      //std::unique_lock<std::shared_timed_mutex> lock(mutex_);

      size_t num_removed = map_.erase(key);
      C10_ASSERT(num_removed <= 1, "This is not a multi-map");
      if (num_removed == 0) {
        throw std::logic_error("Tried to deregister an operator that isn't registered.");
      }
    }

    void* lookup(const Key& key) const {
      // TODO (needed but slow perf. Find better way) std::shared_lock<std::shared_timed_mutex> lock(mutex_);
      auto found = map_.find(key);
      if (found == map_.end()) {
        return nullptr;
      } else {
        return found->second;
      }
    }

private:
    ska::flat_hash_map<Key, void*> map_;
    // TODO Figure out how to get fast locking in C++11 (use boost::shared_timed_mutex? folly::SharedMutex?)
    //mutable std::shared_timed_mutex mutex_;
};
}

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
    ops_.emplace(std::move(dispatch_key), reinterpret_cast<void*>(func));
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
    ops_.erase(dispatch_key);
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
    // smessmer to ezyang: It's a readers/writers lock, so reading should be fast.
    // I wouldn't start hand-rolling our own threadsafe map unless we really know we need it.
    auto dispatch_key = Schema::dispatch::dispatch_key(args...);
    void* found = ops_.lookup(dispatch_key);
    if (found == nullptr) {
      throw std::logic_error("Didn't find operator to dispatch to");
    }
    return reinterpret_cast<typename Schema::signature::func_type*>(found);
  }

  details::ThreadsafeOperatorTable_<typename Schema::dispatch::dispatch_key_type> ops_;
};

}

/*
 * Use this to access the dispatch table singleton for a given op schema.
 * It has an implementation for each op schema def in a cpp file, because
 * we can't rely on the one-definition-rule.
 */
template<class OpSchemaDef> c10::DispatchTable<OpSchemaDef>& c10_dispatch_table();
