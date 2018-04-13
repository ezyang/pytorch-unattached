#pragma once

#include <cstdint>

namespace c10 {

// Modeled off of TypeMeta

#define C10_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(int16_t,Short,i) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(float,Float,d) \
_(double,Double,d)

// _(Half,Half,d) \

class ScalarType {
  int64_t itemsize_;
public:
  constexpr ScalarType(int64_t itemsize)
      : itemsize_(itemsize) {}
};

enum class ScalarTypeId : int64_t {
#define DEFINE_ENUM(_1,n,_2) \
  n,

  C10_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

namespace ScalarTypes {
#define DEFINE_STATIC(_1,name,_2) \
  constexpr ScalarType name(static_cast<int64_t>(ScalarTypeId::name));

C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC
}

template <typename T> const ScalarType& mkScalarType();
#define DEFINE_TEMPLATE(ctype,name,_1) \
template <> \
const ScalarType& mkScalarType<ctype>() { \
  return ScalarTypes::name; \
}

C10_FORALL_SCALAR_TYPES(DEFINE_TEMPLATE)
#undef DEFINE_TEMPLATE

}
