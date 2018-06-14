#pragma once

#include <c10/Registry.h>
#include "StackBasedOperator.h"

namespace c10 {

C10_DECLARE_REGISTRY(
    StackBasedOperatorRegistry,
    StackBasedOperator)

#define C10_REGISTER_STACK_INTERFACE_FOR_OPERATOR(OpSchemaDef)           \
  C10_REGISTER_CLASS(StackBasedOperatorRegistry, OpSchemaDef, ConcreteStackBasedOperator<OpSchemaDef>)

}
