#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>
#include <algorithm>

#define C10_FORALL_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(int16_t,Short,i) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(float,Float,d) \
_(double,Double,d)
// Not yet implemented
// _(Half,Half,d) \

namespace c10 {

// Desired API:
// ScalarType x = scalar_type<int>;
// if (x == scalar_type<float>) ...
// ScalarType::Byte
// x.itemsize()

// Poor man's TypeMeta


// ezyang to @smessmer: Sorry, old school typedef ^^"
// WARNING WARNING WARNING: this is number of elements, NOT number of bytes
typedef void (*PlacementNew)(void *, int64_t numel);
typedef void (*TypedCopy)(const void *, void *, int64_t numel);
typedef void (*TypedDestructor)(void *, int64_t numel);

class ScalarType;

namespace guts {

/**
 * Placement new function for the type.
 */
template<typename T>
void _ctor(void *ptr, int64_t n) {
  T *typed_ptr = static_cast<T *>(ptr);
  for (int i = 0; i < n; ++i) {
    new(typed_ptr + i) T;
  }
}

/**
 * Typed copy function for classes.
 */
// TODO: This argument order is swapped versus memcpy! How confusing!!
template<typename T>
void _copy(const void *src, void *dst, int64_t n) {
  const T *typed_src = static_cast<const T *>(src);
  T *typed_dst = static_cast<T *>(dst);
  for (int i = 0; i < n; ++i) {
    typed_dst[i] = typed_src[i];
  }
}

/**
 * Destructor for non-fundamental types.
 */
template<typename T>
void _dtor(void *ptr, int64_t n) {
  T *typed_ptr = static_cast<T *>(ptr);
  for (int i = 0; i < n; ++i) {
    typed_ptr[i].~T();
  }
}


class ScalarTypeImpl {
  int64_t itemsize_;
  const char *name_;
  PlacementNew ctor_;
  TypedCopy copy_;
  TypedDestructor dtor_;

  constexpr ScalarTypeImpl(int64_t itemsize, const char *name, PlacementNew ctor, TypedCopy copy, TypedDestructor dtor)
      : itemsize_(itemsize), name_(name), ctor_(ctor), copy_(copy), dtor_(dtor) {}

  friend class ScalarTypeImpls;

public:
  constexpr int64_t itemsize() const noexcept { return itemsize_; }
  constexpr PlacementNew ctor() const noexcept { return ctor_; }
  constexpr TypedCopy copy() const noexcept { return copy_; }
  constexpr TypedDestructor dtor() const noexcept { return dtor_; }
  constexpr const char* name() const noexcept { return name_; }
};

// NB: This has to be in a separate class, because ScalarTypeImpl is not a complete type when
// defining static members of ScalarTypeImpl
// NB: struct rather than namespace so we can friend it
struct ScalarTypeImpls {
#define DEFINE_STATIC(ctype, name, _3) \
  static constexpr ScalarTypeImpl name = {sizeof(ctype), #name, nullptr, nullptr, nullptr};

  C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

  static constexpr ScalarTypeImpl String = {sizeof(std::string), "String", _ctor<std::string>, _copy<std::string>,
                                            _dtor<std::string>};
  // I'm not too sure about undefined scalar type, but I've put it in for now since ATen has it.
  static constexpr ScalarTypeImpl Undefined = {0, "Undefined", nullptr, nullptr, nullptr};
};

// A little "pre-implementation" class, so that we can later add the actual static members.
// If you see this type in an error, do note that it is implicitly convertible into a ScalarType,
// which is the thing you actually want.
// See Note [Cult of the dot]
class _ScalarType {
  const ScalarTypeImpl* impl_;
  friend class ScalarType;
public:
  // Sigh, don't really want this to be public, but don't want to define another struct
  // to place ScalarType
  constexpr _ScalarType(const ScalarTypeImpl* impl) : impl_(impl) {};
  _ScalarType() = default;
  _ScalarType(const _ScalarType &rhs) = default;
  _ScalarType(_ScalarType &&rhs) noexcept = default;
  _ScalarType& operator=(_ScalarType &&rhs) = default;
  _ScalarType& operator=(const _ScalarType &rhs) = default;

  constexpr int64_t itemsize() const { return impl_->itemsize(); }
  constexpr PlacementNew ctor() const { return impl_->ctor(); }
  constexpr TypedCopy copy() const { return impl_->copy(); }
  constexpr TypedDestructor dtor() const { return impl_->dtor(); }
  constexpr const char* name() const { return impl_->name(); }

  // NB: if you ever add methods that return/take ScalarType, make sure to define
  // them with ScalarType, not ScalarType_
};

} // namespace guts

class ScalarType : public guts::_ScalarType {
public:
  /*implicit*/ constexpr ScalarType(guts::_ScalarType self) : _ScalarType(self) {}

#define DEFINE_STATIC(_1,name,_3) \
  static constexpr _ScalarType name = {&guts::ScalarTypeImpls::name};

  C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

  static constexpr _ScalarType String = {&guts::ScalarTypeImpls::String};
  static constexpr _ScalarType Undefined = {&guts::ScalarTypeImpls::Undefined};
};

template <typename T> constexpr const ScalarType scalar_type();
#define DEFINE_TEMPLATE(ctype,name,_1) \
template <> \
constexpr const ScalarType scalar_type<ctype>() { \
  return ScalarType::name; \
}

C10_FORALL_SCALAR_TYPES(DEFINE_TEMPLATE)
#undef DEFINE_TEMPLATE

}
