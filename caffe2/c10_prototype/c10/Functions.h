#pragma once

#include <c10/Tensor.h>
#include <c10/Utils.h>
#include <c10/Context.h>
#include <c10/dispatch/Dispatcher.h>

// TODO: Strictly temporary, because the dispatch is hardcoded to go to CPU at the moment
#include <c10/cpu/op/CPUAll.h>

#include <c10/op/OpSchemaDefs.h>

namespace c10 {

// TODO: Work out how the other overloads are going to work.  We only do dtype for now because
// that's the only backend being fleshed out right now.

/**
 * Returns a tensor filled with uninitialized data, with the shape defined by the argument `sizes`.
 *
 * @param sizes A sequence of integers defining the shape of the output tensor
 * @param dtype The desired data type of returned tensor
 */
inline Tensor empty(ArrayRef<int64_t> sizes, caffe2::TypeMeta dtype) {
  return cpu::op::empty(sizes, dtype);
}

/**
 * Returns a tensor filled with the scalar value 0, with the shape defined by the argument `sizes`.
 *
 * @param sizes A sequence of integers defining the shape of the output tensor
 * @param dtype The desired data type of returned tensor
 */
inline Tensor zeros(ArrayRef<int64_t> size, caffe2::TypeMeta dtype) {
  return cpu::op::zeros(size, dtype);
}

inline Tensor tensor(const void* data, ArrayRef<int64_t> size, caffe2::TypeMeta dtype) {
  return cpu::op::tensor(data, size, dtype);
}

// Channeling Caffe2 Tensor::Tensor(const vector<TIndex>& dims, const vector<T>& values, Context* context)
// NB: this is generic
// Because this is templated, it's implementation must live in a header
//
// This is similar to the tensor constructor in PyTorch, but because multidimensional arrays are a pain
// in vanilla C++, we instead let the user specify what size they want.
//
// TODO: Because this is templated, we don't get conversions to ArrayRef.  Add some more overloads
// to help the matcher out.
template<typename T>
inline Tensor tensor(ArrayRef<T> data, ArrayRef<int64_t> size) {
  C10_CHECK(static_cast<size_t>(product(size)) == data.size(),
            "tensor: tensor to be constructed is declared to have size ", size,
            " (and thus would contain ", product(size), " elements), but size of source data is ", data.size()
  );
  return tensor(data.data(), size, caffe2::TypeMeta::Make<T>());
}

template<typename T>
inline Tensor tensor(ArrayRef<T> data) {
  return tensor<T>(data, {data.size()});
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
inline Tensor tensor(const T& value) {
  return tensor<T>(ArrayRef<T>(&value, 1), {});
}

}
