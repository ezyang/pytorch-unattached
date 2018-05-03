#pragma once

#include "impl/DispatchTable.h"

namespace c10 {

class Dispatcher final {
public:
  template<class OpSchemaDef, class... Args>
  static auto registerOp(Args&&... args) {
    auto& dispatch_table_for_this_op = DispatchTable<OpSchemaDef>::singleton();
    return dispatch_table_for_this_op.registerOp(std::forward<Args>(args)...);
  }

  template<class OpSchemaDef, class... Args>
  static auto call(Args&&... args) {
    auto& dispatch_table_for_this_op = DispatchTable<OpSchemaDef>::singleton();
    return dispatch_table_for_this_op.call(std::forward<Args>(args)...);
  }
};

}
