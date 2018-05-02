#pragma once

#include "OpSchema.h"
#include "Dispatcher.h"

/**
 * To register your own operator, do in one (!) cpp file:
 *   C10_DEFINE_OPERATOR(OpSchemaDef, func, {tensor_type1, tensor_type2, ...})
 * Both must be in the same namespace.
 */

namespace c10 {

template<class OpSchemaDef>
class OpRegistrar final {
public:
  OpRegistrar() {}

  template<class... Args>
  static OpRegistrar create(Args&&... args) {
    Dispatcher::singleton().registerOp<OpSchemaDef>(std::forward<Args>(args)...);
    return OpRegistrar();
  }

  OpRegistrar(OpRegistrar&& rhs) = default;

  // TODO deregister in ~OpRegistrar();

private:
  DISALLOW_COPY_AND_ASSIGN(OpRegistrar);
};

class OpRegistrarEntryPoint final {
public:
  template<class OpSchemaDef, size_t num_tensor_args>
  static OpRegistrar<OpSchemaDef> define(typename OpSchema<OpSchemaDef>::func_type* func, const TensorTypeId (&tensorTypeIds)[num_tensor_args]) {
    return OpRegistrar<OpSchemaDef>::create(func, tensorTypeIds);
  }

  // overload for ops with zero tensor arguments (C arrays with size zero are invalid in C++, so they can't use the method above)
  template<class OpSchemaDef>
  static OpRegistrar<OpSchemaDef> define(typename OpSchema<OpSchemaDef>::func_type* func) {
    return OpRegistrar<OpSchemaDef>::create(func);
  }
};

}

// TODO Improve macro API
#define CONCAT_IMPL( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT_IMPL( x, y )
#define C10_REGISTER_OP()                                                           \
  static auto MACRO_CONCAT(__opRegistrar_, __COUNTER__) = OpRegistrarEntryPoint()   \
