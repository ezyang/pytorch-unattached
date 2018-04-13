#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace c10 {

// Poor man's TypeMeta

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
  // ezyang to @smessmer: Sorry, old school typedef ^^"
  typedef void (*PlacementNew)(void*, size_t);
  typedef void (*TypedCopy)(const void*, void*, size_t);
  typedef void (*TypedDestructor)(void*, size_t);

  int64_t itemsize_;
  const char* name_;
  PlacementNew ctor_;
  TypedCopy copy_;
  TypedDestructor dtor_;
public:

  constexpr ScalarType(int64_t itemsize, const char* name, PlacementNew ctor, TypedCopy copy, TypedDestructor dtor)
  : itemsize_(itemsize)
  , name_(name)
  , ctor_(ctor)
  , copy_(copy)
  , dtor_(dtor)
  {}

  constexpr PlacementNew ctor() const { return ctor_; }
  constexpr TypedCopy copy() const { return copy_; }
  constexpr TypedDestructor dtor() const { return dtor_; }
};

/**
 * Placement new function for the type.
 */
template <typename T>
void _Ctor(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (int i = 0; i < n; ++i) {
    new (typed_ptr + i) T;
  }
}

/**
 * Destructor for non-fundamental types.
 */
template <typename T>
void _Dtor(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (int i = 0; i < n; ++i) {
    typed_ptr[i].~T();
  }
}

/**
 * Typed copy function for classes.
 */
template <typename T>
void _Copy(const void* src, void* dst, size_t n) {
  const T* typed_src = static_cast<const T*>(src);
  T* typed_dst = static_cast<T*>(dst);
  for (int i = 0; i < n; ++i) {
    typed_dst[i] = typed_src[i];
  }
}

namespace ScalarTypes {
#define DEFINE_STATIC(ctype,name,_2) \
  constexpr ScalarType name(sizeof(ctype), #name, nullptr, nullptr, nullptr);

C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

  constexpr ScalarType String(sizeof(std::string), "String", _Ctor<std::string>, _Copy<std::string>, _Dtor<std::string>);
}

template <typename T> const ScalarType* mkScalarType();
#define DEFINE_TEMPLATE(ctype,name,_1) \
template <> \
const ScalarType* mkScalarType<ctype>() { \
  return &ScalarTypes::name; \
}

C10_FORALL_SCALAR_TYPES(DEFINE_TEMPLATE)
#undef DEFINE_TEMPLATE

}
