#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <functional>
#include <algorithm>

#define C10_FORALL_BUILTIN_DATA_TYPES(_) \
_(uint8_t,uint8,i) \
_(int8_t,int8,i) \
_(int16_t,int16,i) \
_(int,int32,i) \
_(int64_t,int64,i) \
_(float,float32,d) \
_(double,float64,d)
// Not yet implemented
// _(Half,Half,d)

namespace c10 {

// ezyang to @smessmer: Sorry, old school typedef ^^"
// WARNING WARNING WARNING: these functions take number of elements, NOT number of bytes

/**
 * Function type for vectorized placement new.  A function of this type will apply placement
 * new for numel elements in the memory pointed at by p.
 */
typedef void (*PlacementNew)(void * p, int64_t numel);

/**
 * Function type for vectorized copy-assignment.  A function of this type will apply copy assignment
 * for numel elements from the memory pointed at by `src` to the memory pointed at `dst`.
 *
 * @todo This is argument signature is swapped versus `std::memcpy`, which is confusing.  We inherited
 * this ordering from Caffe2.
 */
typedef void (*TypedCopy)(const void * src, void * dst, int64_t numel);

/**
 * Function type for vectorized placement delete.  A function of this type will apply placement
 * delete for `numel` elements from the memory pointed at by `p`.
 */
typedef void (*TypedDestructor)(void * p, int64_t numel);

namespace guts {

/**
 * Default placement new function for the type T.
 *
 * @tparam T   Type to perform placement-new with
 * @param ptr  Pointer to region of memory to placement-new
 * @param n    How many elements at ptr to placement new.
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


class DataTypeImpl {
  int64_t itemsize_;
  const char *name_;
  PlacementNew ctor_;
  TypedCopy copy_;
  TypedDestructor dtor_;

  constexpr DataTypeImpl(int64_t itemsize, const char *name, PlacementNew ctor, TypedCopy copy, TypedDestructor dtor)
      : itemsize_(itemsize), name_(name), ctor_(ctor), copy_(copy), dtor_(dtor) {}

  friend struct DataTypeImpls;

public:
  constexpr int64_t itemsize() const noexcept { return itemsize_; }
  constexpr PlacementNew ctor() const noexcept { return ctor_; }
  constexpr TypedCopy copy() const noexcept { return copy_; }
  constexpr TypedDestructor dtor() const noexcept { return dtor_; }
  constexpr const char* name() const noexcept { return name_; }
};

// NB: This has to be in a separate class, because DataTypeImpl is not a complete type when
// defining static members of DataTypeImpl
// NB: struct rather than namespace so we can friend it
struct DataTypeImpls {
#define DEFINE_STATIC(ctype, name, _3) \
  static constexpr DataTypeImpl name = {sizeof(ctype), #name, nullptr, nullptr, nullptr};

  C10_FORALL_BUILTIN_DATA_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

  static constexpr DataTypeImpl string_dtype = {sizeof(std::string), "string", _ctor<std::string>, _copy<std::string>,
                                            _dtor<std::string>};
  // I'm not too sure about undefined scalar type, but I've put it in for now since ATen has it.
  static constexpr DataTypeImpl undefined_dtype = {0, "undefined", nullptr, nullptr, nullptr};
};

} // namespace guts

class DataType {
  const guts::DataTypeImpl* impl_;
public:
  // Sigh, don't really want this to be public, but don't want to define another struct
  // to place the DataType constants
  constexpr DataType(const guts::DataTypeImpl* impl) : impl_(impl) {};
  DataType() : impl_(&guts::DataTypeImpls::undefined_dtype) {};
  DataType(const DataType &rhs) = default;
  DataType(DataType &&rhs) noexcept = default;
  DataType& operator=(DataType &&rhs) = default;
  DataType& operator=(const DataType &rhs) = default;

  inline bool operator==(const DataType& other) const {
    return impl_ == other.impl_;
  }

  constexpr int64_t itemsize() const { return impl_->itemsize(); }
  constexpr PlacementNew ctor() const { return impl_->ctor(); }
  constexpr TypedCopy copy() const { return impl_->copy(); }
  constexpr TypedDestructor dtor() const { return impl_->dtor(); }
  constexpr const char* name() const { return impl_->name(); }
};

inline std::ostream& operator<<(std::ostream& out, DataType d) {
  out << std::string(d.name());
  return out;
}

#define DEFINE_STATIC(_1,name,_3) \
constexpr DataType name = {&guts::DataTypeImpls::name};

C10_FORALL_BUILTIN_DATA_TYPES(DEFINE_STATIC)
#undef DEFINE_STATIC

// A little more wordy for these
constexpr DataType string_dtype = {&guts::DataTypeImpls::string_dtype};
constexpr DataType undefined_dtype = {&guts::DataTypeImpls::undefined_dtype};

// TODO: this templated function interacts poorly with dtype() methods we have on
// Tensor, and also the fact that conventionally DataType arguments are named
// dtype.  Strongly consider renaming this, maybe to data_type.  Conventional way
// to call this internally is to say c10::dtype<T>(); this will always be unambiguous.
template <typename T> constexpr const DataType dtype();
#define DEFINE_TEMPLATE(ctype,name,_1) \
template <> \
constexpr const DataType dtype<ctype>() { \
  return name; \
}

C10_FORALL_BUILTIN_DATA_TYPES(DEFINE_TEMPLATE)
#undef DEFINE_TEMPLATE

template <> constexpr const DataType dtype<std::string>() { return string_dtype; }

}
