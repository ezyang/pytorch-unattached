#pragma once

#include <c10/guts/IdWrapper.h>
#include <string>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <c10/guts/Macros.h>

namespace c10 {

namespace details {
  using _tensorTypeId_underlyingType = uint8_t;
}

class TensorTypeId final : public guts::IdWrapper<TensorTypeId, details::_tensorTypeId_underlyingType> {
public:
  // Don't use this!
  // Unfortunately, a default constructor needs to be defined because of https://reviews.llvm.org/D41223
  TensorTypeId(): IdWrapper(0) {}
private:
  constexpr explicit TensorTypeId(details::_tensorTypeId_underlyingType id): IdWrapper(id) {}

  friend class TensorTypeIdCreator;
  friend std::ostream& operator<<(std::ostream&, TensorTypeId);
};

}

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::TensorTypeId);
