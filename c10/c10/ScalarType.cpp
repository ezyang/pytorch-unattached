#include "ScalarType.h"

namespace c10 {

// Appease the linker

#define DEFINE_STATIC(_1,name,_3) \
  constexpr guts::ScalarTypeImpl guts::ScalarTypeImpls::name; \
  constexpr guts::_ScalarType ScalarType::name;

C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

constexpr guts::ScalarTypeImpl guts::ScalarTypeImpls::String;
constexpr guts::_ScalarType ScalarType::String;
constexpr guts::ScalarTypeImpl guts::ScalarTypeImpls::Undefined;
constexpr guts::_ScalarType ScalarType::Undefined;

}
