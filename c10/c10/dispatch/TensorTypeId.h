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

inline std::ostream& operator<<(std::ostream& str, TensorTypeId rhs) {
  return str << rhs.underlyingId();
}

}
C10_DEFINE_HASH_FOR_IDWRAPPER(c10::TensorTypeId);
namespace c10 {

class TensorTypeIdCreator final {
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

class TensorTypeIdRegistry final {
public:
  void registerId(TensorTypeId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    registeredTypeIds_.emplace(id);
  }

  void deregisterId(TensorTypeId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    registeredTypeIds_.erase(id);
  }

private:
  // TODO Something faster than unordered_set?
  std::unordered_set<TensorTypeId> registeredTypeIds_;
  std::mutex mutex_;
};

class TensorTypeIdRegistrar final {
public:
  TensorTypeIdRegistrar(TensorTypeId id, TensorTypeIdRegistry* registry)
  : id_(id), registry_(registry) {
    registry_->registerId(id);
  }

  ~TensorTypeIdRegistrar() {
    if (registry_ != nullptr) {
      registry_->deregisterId(id_);
    }
  }

  // copy assignment/constructor would break RAII promise
  TensorTypeIdRegistrar(const TensorTypeIdRegistrar&) = delete;
  TensorTypeIdRegistrar& operator=(const TensorTypeIdRegistrar&) = delete;

  TensorTypeIdRegistrar(TensorTypeIdRegistrar&& rhs)
  : id_(rhs.id_), registry_(rhs.registry_) {
    rhs.registry_ = nullptr;
  }
  // move assignment not needed currently, can be added if needed.
  TensorTypeIdRegistrar& operator=(TensorTypeIdRegistrar&&) = delete;

  TensorTypeId id() const {
    return id_;
  }

private:
  TensorTypeId id_;
  TensorTypeIdRegistry* registry_;
};

class TensorTypeIds final {
public:
  TensorTypeIdRegistrar createAndRegister() {
    TensorTypeId id = creator_.create();
    return TensorTypeIdRegistrar(id, &registry_);
  }

  static constexpr TensorTypeId undefined() {
    return TensorTypeIdCreator::undefined();
  }

private:
  TensorTypeIdCreator creator_;
  TensorTypeIdRegistry registry_;
};

}

#define C10_DECLARE_TENSOR_TYPE(TensorName)                                      \
  TensorTypeId TensorName();                                                     \

#define C10_DEFINE_TENSOR_TYPE(TensorName)                                       \
  TensorTypeId TensorName() {                                                    \
    static auto registration_raii = dispatch().createAndRegisterTensorTypeId();  \
    return registration_raii.id();                                               \
  }                                                                              \
