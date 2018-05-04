#pragma once

/**
 * To register your own tensor types, do in a header file:
 *   C10_DECLARE_TENSOR_TYPE(MY_TENSOR)
 * and in one (!) cpp file:
 *   C10_DEFINE_TENSOR_TYPE(MY_TENSOR)
 * Both must be in the same namespace.
 */

 #include "TensorTypeId.h"
 #include <atomic>

namespace c10 {

class TensorTypeIdCreator final {
public:
  TensorTypeIdCreator();

  TensorTypeId create();

  static constexpr TensorTypeId undefined() {
    return TensorTypeId(0);
  }

private:
  std::atomic<details::_tensorTypeId_underlyingType> last_id_;

  static constexpr TensorTypeId max_id_ = TensorTypeId(std::numeric_limits<details::_tensorTypeId_underlyingType>::max());

  DISALLOW_COPY_AND_ASSIGN(TensorTypeIdCreator);
};

class TensorTypeIdRegistry final {
public:
  TensorTypeIdRegistry();

  void registerId(TensorTypeId id);
  void deregisterId(TensorTypeId id);

private:
  // TODO Something faster than unordered_set?
  std::unordered_set<TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;

  DISALLOW_COPY_AND_ASSIGN(TensorTypeIdRegistry);
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

  DISALLOW_COPY_AND_ASSIGN(TensorTypeIds);
};

inline constexpr TensorTypeId TensorTypeIds::undefined() {
  return TensorTypeIdCreator::undefined();
}

class TensorTypeIdRegistrar final {
public:
  TensorTypeIdRegistrar();
  ~TensorTypeIdRegistrar();

  TensorTypeId id() const;

private:
  TensorTypeId id_;

  DISALLOW_COPY_AND_ASSIGN(TensorTypeIdRegistrar);
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
  }
