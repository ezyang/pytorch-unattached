#include <c10/c10.h>

#include "All.h"

namespace c10 { namespace ops {

// NB: this is generic (assuming you pass in the backend dispatcher)
Tensor tensor(DataType dtype, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  auto r = c10::tensor(dtype);
  r.resize_(size, stride);
  return r;
}

#if 0


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

}} // namespace c10::op
