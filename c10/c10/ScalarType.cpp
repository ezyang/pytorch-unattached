#include "ScalarType.h"

namespace c10 {

#define DEFINE_STATIC(_1,name,_3) \
  constexpr guts::ScalarTypeImpl guts::ScalarTypeImpls::name;

C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

constexpr guts::ScalarTypeImpl guts::ScalarTypeImpls::String;

}
