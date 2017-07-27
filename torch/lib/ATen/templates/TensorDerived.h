#pragma once

$th_headers

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"

namespace at {

struct ${Tensor} : public TensorImpl {
public:
  explicit ${Tensor}(Context* context);
  ${Tensor}(Context* context, ${THTensor} * tensor);
  virtual ~${Tensor}();
  virtual const char * toString() const override;
  virtual IntList sizes() override;
  virtual IntList strides() override;
  virtual int64_t dim() override;
  virtual Scalar localScalar() override;
  virtual void assign_(Scalar s) override;
  virtual void * unsafeGetTH(bool retain) override;
  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  ${THTensor} * tensor;
  Context* context;
  friend struct ${Type};
};

} // namespace at
