#include "TensorTypeId.h"

namespace c10 {

std::ostream& operator<<(std::ostream& str, TensorTypeId rhs) {
  return str << rhs.underlyingId();
}

}