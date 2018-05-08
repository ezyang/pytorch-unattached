#pragma once

#include <c10/dispatch/impl/DispatchTable.h>

namespace c10 {

// ezyang to smessmer: Given that these are all static functions, no reason to have
// a class right?

// TODO: auto return type is C++14

// TODO: Because these signatures are all perfect-forwarded, you have to look under the
// covers at impl::DispatchTable to know what the types are.

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 */
// Implementation note: this class abstracts over the fact that we have per-operator
// dispatch tables.  This could be easily adjusted to have a single global hash
// table.
class Dispatcher final {
public:

  /**
   * Register an operator to the dispatch table for some operator schema.
   *
   * @tparam OpSchemaDef Operator schema to register this operator to (mandatory)
   * @tparam Args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::registerOp (inferred)
   * @param args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::registerOp
   * @return void
   */
  template<class OpSchemaDef, class... Args>
  static auto registerOp(Args&&... args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.registerOp(std::forward<Args>(args)...);
  }

  /**
   * Remove an operator from the dispatch table for some operator schema.
   *
   * @tparam OpSchemaDef Operator schema to deregister from (mandatory)
   * @tparam Args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::deregisterOp (inferred)
   * @param args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::deregisterOp
   * @return void
   */
  template<class OpSchemaDef, class... Args>
  static auto deregisterOp(Args&&... args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.deregisterOp(std::forward<Args>(args)...);
  }

  /**
   * Perform a dynamic dispatch to some operator
   *
   * @tparam OpSchemaDef Operator schema to dispatch with (mandatory)
   * @tparam Args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::call (inferred)
   * @param args Perfect-forwarding args to c10::dispatch::impl::DispatchTable::call
   * @return Return type of this operator
   */
  template<class OpSchemaDef, class... Args>
  static auto call(Args&&... args) {
    auto& dispatch_table_for_this_op = c10_dispatch_table<OpSchemaDef>();
    return dispatch_table_for_this_op.call(std::forward<Args>(args)...);
  }
};

}
