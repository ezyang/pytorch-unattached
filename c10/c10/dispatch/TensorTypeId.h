#pragma once

#include <c10/guts/IdWrapper.h>
#include <string>
#include <iostream>

namespace c10 {

/**
 * To register your own tensor types, do in a header file:
 *   C10_DECLARE_TENSOR_TYPE(MY_TENSOR)
 * and in one (!) cpp file:
 *   C10_DEFINE_TENSOR_TYPE(MY_TENSOR)
 * Both must be in the same namespace.
 */

namespace details {
  using _tensorTypeId_underlyingType = uint8_t;
}

class TensorTypeId final : public guts::IdWrapper<TensorTypeId, details::_tensorTypeId_underlyingType> {
private:
  constexpr explicit TensorTypeId(details::_tensorTypeId_underlyingType id): IdWrapper(id) {}

  friend class TensorTypeIds;
  friend std::ostream& operator<<(std::ostream&, TensorTypeId);
};

inline std::ostream& operator<<(std::ostream& str, TensorTypeId rhs) {
  return str << rhs.underlyingId();
}

class TensorTypeIds final {
public:
  TensorTypeId create() {
    auto id = TensorTypeId(++next_id_);

    if (id == max_id_) {
      // If this happens in prod, we have to change details::_tensorTypeId_underlyingType to uint16_t.
      throw std::logic_error("Tried to define more than " + std::to_string(std::numeric_limits<details::_tensorTypeId_underlyingType>::max()-1) + " tensor types, which is unsupported");
    }

    return id;
  }

  static constexpr TensorTypeId undefined() {
    return TensorTypeId(0);
  }
private:
  std::atomic<details::_tensorTypeId_underlyingType> next_id_;

  static constexpr TensorTypeId max_id_ = TensorTypeId(std::numeric_limits<details::_tensorTypeId_underlyingType>::max());
};

}

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::TensorTypeId);

#define C10_DECLARE_TENSOR_TYPE(TensorName)                                \
  TensorTypeId CPU_TENSOR();                                               \

#define C10_DEFINE_TENSOR_TYPE(TensorName)                                 \
  TensorTypeId CPU_TENSOR() {                                              \
    static TensorTypeId singleton = dispatch().createTensorTypeId();       \
    return singleton;                                                      \
  }                                                                        \
