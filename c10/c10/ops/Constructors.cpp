#include <c10/c10.h>

#include "Constructors.h"

namespace c10 { namespace ops {

#if 0

// NB: this is generic (assuming you pass in the backend dispatcher)
Tensor tensor(ScalarType scalar_type, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  auto r = tensor(scalar_type);
  r.resize_(size, stride);
  return r;
}

// Channeling Caffe2 Tensor::Tensor(const vector<TIndex>& dims, const vector<T>& values, Context* context)
// NB: this is generic
template<typename T>
Tensor tensor(ArrayRef<int64_t> size, std::vector<T> data) {
  auto r = tensor(c10::scalar_type<T>, size, contiguous_strides(size));
  C10_CHECK(r.numel() == data.size());
  r.template copy_<T>(data);
  return r;
}
#endif

}} // namespace c10::ops
