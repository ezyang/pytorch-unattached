#pragma once

#include "c10/TypeId.h"
#include "c10/ArrayRef.h"

#include "Retainable.h"

#include <vector>

namespace c10 {
  class Tensor;
}

// NB: It's called guts because it's short and gets the point across :)
namespace c10 { namespace guts {

class UndefinedTensorImpl;

// TODO: Fill in an actual SmallVector implementation here.  Both Folly and LLVM's
// implementation are a bit annoying to make standalone.  Maybe this can be made
// simpler by assuming T is POD.
// TODO: For the common case of sizes and strides, the lengths of the two arrays
// are equal, so there is no need to store the ndim twice.  Worth thinking about.
template<typename T>
using SmallVector = std::vector<T>;

// For now: try using empty tensors for type (I think we'll probably add a Type
// object)

// NB: Use of virtual functions means that this is NOT a plain old data class.
// This means that we don't get inlineable C API functions which access the representation
// directly
class TensorImpl : public RetainableImpl {
  // Used for dispatch on the object
  const TypeId type_id_;

  SmallVector<int64_t> size_;

  friend class c10::Tensor;

public:
  explicit TensorImpl(TypeId type_id) : type_id_(type_id), RetainableImpl() {};

  inline ArrayRef<int64_t> size() const {
    return size_;
  }

  // NB: In Caffe2, this quantity is CACHED.  For simplicity, we don't cache it for now, but consider
  // doing so.
  // NB: This was ported in from the TH backend (which we normally defer to.)
  inline int64_t numel() const {
    int64_t r = 1;
    for (auto s : size()) r *= s;
    return r;
  }

  virtual ArrayRef<int64_t> stride() const {
    throw std::runtime_error("TensorImpl::stride()");
  }

  virtual int64_t dim() const {
    return static_cast<int64_t>(size().size());
  }

  virtual void *data_ptr() const {
    throw std::runtime_error("TensorImpl::data_ptr()");
  }

  virtual ~TensorImpl() = default;

  // The following virtual functions TEMPORARILY live here.  When the
  // dispatcher comes online, they will become dispatched by that mechanism.

  // Create a new tensor of the same type as this tensor
  virtual Tensor tensor(ArrayRef<int64_t> size) const {
  }
};

// See design notes on Tensor.h, where this is hardcoded a few times.
// smessmer to @ezyang: What are your concerns about using an empty CPU tensor instead of an undefined one?
// ezyang to @smessmer: For example, there will be a method tensor, the semantics are x.tensor({2, 3}) will
//      create a 2x3 tensor of the same "type" as x.  If x is an empty CPU tensor, you'll get a CPU tensor,
//      instead of an error, which should have happened.  It just seems morally wrong to privilege empty CPU
//      tensors in this way.  Also, you don't get reliable pointer equality tests anymore.
class UndefinedTensorImpl : public TensorImpl {
  UndefinedTensorImpl() : TensorImpl(TypeIds::Undefined) {};
public:
  int64_t dim() const override {
    throw std::runtime_error("UndefinedTensorImpl::dim()");
  }

  static UndefinedTensorImpl *singleton() {
    // smessmer to @ezyang: Not sure this singleton is a good idea. If wrapped in Tensor, it is subject to ref counting and might get destructed.
    //          If we need this singleton, we should wrap it into a Tensor instance here to make sure the ref count is always positive.
    // ezyang to @smessmer: I added checks for UndefinedTensorImpl in retain/release, but it is a bit awkward.
    //          But if it's just one singleton it might be OK.  The primary motivation for a singleton is so we can
    //          avoid dereferencing the pointer to do a null check.
    static UndefinedTensorImpl singleton_;
    return &singleton_;
  }
};

}} // namespace c10::guts
