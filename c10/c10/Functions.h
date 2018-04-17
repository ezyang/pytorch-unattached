#pragma once

#include "Tensor.h"
#include "Utils.h"

// TODO: Strictly temporary: these polymorphic functions should still go through the dispatcher
#include "c10/op/All.h"

// TODO: Strictly temporary, hardcoded CPU
#include "c10/cpu/op/CPUAll.h"

namespace c10 {

inline Tensor tensor(DataType dtype) {
  // TODO: This should go through dispatcher, instead of hardcoding CPU
  return cpu::op::tensor(dtype);
}

inline Tensor tensor(DataType dtype, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  // TODO: This should go through dispatcher, instead of hardcoding the polymorphic operator
  return op::tensor(dtype, size, stride);
}

// Channeling Caffe2 Tensor::Tensor(const vector<TIndex>& dims, const vector<T>& values, Context* context)
// NB: this is generic
// Because this is templated, it's implementation must live in a header
template<typename T>
inline Tensor tensor(ArrayRef<int64_t> size, std::vector<T> data) {
  auto r = tensor(c10::dtype<T>, size, contiguous_strides(size));
  C10_CHECK(r.numel() == data.size());
  r.template copy_<T>(data);
  return r;
}

// Channeling Caffe2 Tensor::Tensor(const T& value, Context* context)
// Create a scalar tensor from a single value
// NB: this is generic
// NB: the test that T is_scalar prevents this template from clobbering other
// overloads (though, this may not be an issue in C10, since Context is no longer
// a templated argument, so C++'s rule of preferring a non-template function over
// a templated one might actually work.)
template <typename T,
          typename = typename std::enable_if<std::is_scalar<T>::value>::type>
Tensor tensor(const T& value) {
  auto r = tensor(c10::dtype<T>, {}, {});
  r.template copy_<T>({&value, 1});
  return r;
}


}
