#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

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
typedef void (*PlacementNew)(void *, size_t);
typedef void (*TypedCopy)(const void *, void *, size_t);
typedef void (*TypedDestructor)(void *, size_t);

namespace guts {

/**
 * Placement new function for the type.
 */
template<typename T>
void _ctor(void *ptr, size_t n) {
  T *typed_ptr = static_cast<T *>(ptr);
  for (int i = 0; i < n; ++i) {
    new(typed_ptr + i) T;
  }
}

/**
 * Typed copy function for classes.
 */
template<typename T>
void _copy(const void *src, void *dst, size_t n) {
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
void _dtor(void *ptr, size_t n) {
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
  constexpr int64_t itemsize() const { return itemsize_; }
  constexpr PlacementNew ctor() const { return ctor_; }
  constexpr TypedCopy copy() const { return copy_; }
  constexpr TypedDestructor dtor() const { return dtor_; }
  constexpr const char* name() const { return name_; }
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
};

} // namespace guts

class ScalarType {
  const guts::ScalarTypeImpl* impl_;
public:
  // Sigh, don't really want this to be public, but don't want to define another struct
  // to place ScalarType
  constexpr ScalarType(const guts::ScalarTypeImpl* impl) : impl_(impl) {};
  ScalarType() = default;
  ScalarType(const ScalarType &rhs) = default;
  ScalarType(ScalarType &&rhs) noexcept = default;

  constexpr int64_t itemsize() const { return impl_->itemsize(); }
  constexpr PlacementNew ctor() const { return impl_->ctor(); }
  constexpr TypedCopy copy() const { return impl_->copy(); }
  constexpr TypedDestructor dtor() const { return impl_->dtor(); }
  constexpr const char* name() const { return impl_->name(); }
};

// Top level namespace grab!!!  Using the ATen k-prefix convention

#define DEFINE_STATIC(_1,name,_3) \
constexpr ScalarType k ## name(&guts::ScalarTypeImpls::name);

C10_FORALL_SCALAR_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

constexpr ScalarType kString(&guts::ScalarTypeImpls::String);

template <typename T> constexpr const ScalarType mkScalarType();
#define DEFINE_TEMPLATE(ctype,name,_1) \
template <> \
constexpr const ScalarType mkScalarType<ctype>() { \
  return k##name; \
}

C10_FORALL_SCALAR_TYPES(DEFINE_TEMPLATE)
#undef DEFINE_TEMPLATE

}
