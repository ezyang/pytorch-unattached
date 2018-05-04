#pragma once

#include "Dispatcher.h"

#define C10_DEFINE_OP_SCHEMA(OpSchemaDef)                                         \
  template<>                                                                      \
  c10::DispatchTable<OpSchemaDef>& c10_dispatch_table<OpSchemaDef>() {            \
    static c10::DispatchTable<OpSchemaDef> singleton;                             \
    return singleton;                                                             \
  }