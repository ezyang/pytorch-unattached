#pragma once

#include "Tensor.h"

// TODO: Strictly temporary: these polymorphic functions should still go through the dispatcher
#include "c10/op/All.h"

// TODO: Strictly temporary, hardcoded CPU
#include "c10/cpu/op/CPUAll.h"

namespace c10 {

inline Tensor tensor(DataType dtype) {
  // TODO: This should go through dispatcher, instead of hardcoding CPU
  return cpu::op::tensor(dtype);
}

}
