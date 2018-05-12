#pragma once

#include "Dispatcher.h"
#include "../stack_bindings/StackBasedOperatorRegistry.h"

// TODO Better error message when this definition is missing

/**
 * Macro for defining an operator schema.  Every user-defined OpSchemaDef struct must
 * invoke this macro on it.  Internally, this arranges for the dispatch table for
 * the operator to be created.
 */
#define C10_DEFINE_OP_SCHEMA(OpSchemaDef)                                         \
  template<>                                                                      \
  c10::DispatchTable<OpSchemaDef>& c10_dispatch_table<OpSchemaDef>() {            \
    static c10::DispatchTable<OpSchemaDef> singleton;                             \
    return singleton;                                                             \
  }                                                                               \
  /* TODO Force definition of C10_DEFINE_OP_SCHEMA (and other macros) inside c10 namespace instead */ \
  namespace c10 {                                                                 \
    C10_REGISTER_STACK_INTERFACE_FOR_OPERATOR(OpSchemaDef)                        \
  }
