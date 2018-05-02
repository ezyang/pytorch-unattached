#pragma once

#include <c10/guts/IdWrapper.h>
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_set>

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

  friend class TensorTypeIdCreator;
  friend std::ostream& operator<<(std::ostream&, TensorTypeId);
};

}
C10_DEFINE_HASH_FOR_IDWRAPPER(c10::TensorTypeId);
namespace c10 {

class TensorTypeIdCreator final {
public:
  TensorTypeId create();

  static constexpr TensorTypeId undefined() {
    return TensorTypeId(0);
  }
private:
  std::atomic<details::_tensorTypeId_underlyingType> next_id_;

  static constexpr TensorTypeId max_id_ = TensorTypeId(std::numeric_limits<details::_tensorTypeId_underlyingType>::max());
};

class TensorTypeIdRegistry final {
public:
  void registerId(TensorTypeId id);
  void deregisterId(TensorTypeId id);

private:
  // TODO Something faster than unordered_set?
  std::unordered_set<TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;
};

class TensorTypeIds final {
public:
  static TensorTypeIds& singleton();

  TensorTypeId createAndRegister();
  void deregister(TensorTypeId id);

  static constexpr TensorTypeId undefined();

private:
  TensorTypeIds();

  TensorTypeIdCreator creator_;
  TensorTypeIdRegistry registry_;
};

inline constexpr TensorTypeId TensorTypeIds::undefined() {
  return TensorTypeIdCreator::undefined();
}

class TensorTypeIdRegistrar final {
public:
  TensorTypeIdRegistrar();
  ~TensorTypeIdRegistrar();

  TensorTypeIdRegistrar(const TensorTypeIdRegistrar&) = delete;
  TensorTypeIdRegistrar& operator=(const TensorTypeIdRegistrar&) = delete;
  TensorTypeIdRegistrar(TensorTypeIdRegistrar&& rhs) = delete;
  TensorTypeIdRegistrar& operator=(TensorTypeIdRegistrar&&) = delete;

  TensorTypeId id() const;

private:
  TensorTypeId id_;
};

inline TensorTypeId TensorTypeIdRegistrar::id() const {
  return id_;
}

}

#define C10_DECLARE_TENSOR_TYPE(TensorName)                                      \
  TensorTypeId TensorName();                                                     \

#define C10_DEFINE_TENSOR_TYPE(TensorName)                                       \
  TensorTypeId TensorName() {                                                    \
    static TensorTypeIdRegistrar registration_raii;                              \
    return registration_raii.id();                                               \
  }                                                                              \
