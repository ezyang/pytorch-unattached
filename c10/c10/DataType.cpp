#include "DataType.h"

namespace c10 {

// These constexpr stubs are not necessary in C++17
//    see https://stackoverflow.com/a/43194989/23845

#define DEFINE_STATIC(_1,name,_3) \
  constexpr guts::DataTypeImpl guts::DataTypeImpls::name; \

C10_FORALL_BUILTIN_DATA_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

constexpr guts::DataTypeImpl guts::DataTypeImpls::string_dtype;
constexpr guts::DataTypeImpl guts::DataTypeImpls::undefined_dtype;

}
