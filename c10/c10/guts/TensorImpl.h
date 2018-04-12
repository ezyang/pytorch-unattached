#pragma once

#include "c10/TypeId.h"
#include "c10/ArrayRef.h"
#include "c10/Tensor.h"
#include "c10/SmallVector.h"
#include "c10/Optional.h"

#include "Retainable.h"

#include <vector>
#include <c10/DimVector.h>

namespace c10 {
  class Tensor;
}

// NB: It's called guts because it's short and gets the point across :)
namespace c10 { namespace guts {

// For now: try using empty tensors for type (I think we'll probably add a Type
// object)

// NB: Use of virtual functions means that this is NOT a plain old data class.
// This means that we don't get inlineable C API functions which access the representation
// directly
// ezyang to @smessmer: You need some sort of way to cast from TensorImpl to RetainableImpl, otherwise
// the wrapper doesn't seem to work???
class TensorImpl : public RetainableImpl {
  // TODO: Is this OK to be protected
protected:
  // Used for dispatch on the object
  const TypeId type_id_;

  DimVector size_;

public:
  explicit TensorImpl(TypeId type_id) : RetainableImpl(), type_id_(type_id), size_() {};

  ArrayRef<int64_t> size() const {
    return size_;
  }

  // NB: In Caffe2, this quantity is CACHED.  For simplicity, we don't cache it for now, but consider
  // doing so.
  // NB: This was ported in from the TH backend (which we normally defer to.)
  // smessmer to @ezyang: Do we need this in here? Seems like something that can live as a non-member.
  int64_t numel() const {
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
  // They're labeled with HACK in their name

  virtual void HACK_resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
    throw std::runtime_error("resize");
  }
};

// See design notes on Tensor.h, where this is hardcoded a few times.
// smessmer to @ezyang: What are your concerns about using an empty CPU tensor instead of an undefined one?
// ezyang to @smessmer: For example, there will be a method tensor, the semantics are x.tensor({2, 3}) will
//      create a 2x3 tensor of the same "type" as x.  If x is an empty CPU tensor, you'll get a CPU tensor,
//      instead of an error, which should have happened.  It just seems morally wrong to privilege empty CPU
//      tensors in this way.  Also, you don't get reliable pointer equality tests anymore.
class UndefinedTensorImpl final : public TensorImpl {
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
