#pragma once

#include "c10/TypeId.h"
#include "c10/ArrayRef.h"

#include <vector>

// NB: It's called guts because it's short and gets the point across :)
namespace c10 { namespace guts {

// TODO: Fill in an actual SmallVector implementation here.  Both Folly and LLVM's
// implementation are a bit annoying to make standalone.  Maybe this can be made
// simpler by assuming T is POD.
template<typename T>
using SmallVector = std::vector<T>;

// For now: try using empty tensors for type (I think we'll probably add a Type
// object)

// NB: Use of virtual functions means that this is NOT a plain old data class.
// This means that we don't get inlineable C API functions which access the representation
// directly
class TensorImpl {
  // Used for dispatch on the object
  const TypeId type_id_;

  SmallVector<int64_t> size_;
  void* data_ptr_;

  friend class c10::Tensor;

public:
  explicit TensorImpl(TypeId type_id) : type_id_(type_id), refcount_(1) {};

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

  inline void *data_ptr() const {
    return data_ptr_;
  }

  virtual ~TensorImpl() = default;
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

class CPUTensorImpl final : public TensorImpl {
  void *data_ptr_;
public:
  CPUTensorImpl() : TensorImpl(TypeIds::CPUTensor), data_ptr_(nullptr) {};

};

/*
// smessmer to @ezyang: Inheriting StridedCPUTensorImpl from CPUTensorImpl has some issues with
//          the TypeId. I'd recommend having these classes independent of each other
//          and pull out common functionality into a member object used by both.
// ezyang to @smessmer: Have at it. The point of having private Impls is we can move stuff
//          around with impunity.  My personal preference is for CPUTensorImpl to have data_ptr_
//          and stride_, and have no StridedCPUTensorImpl.
// ezyang: Temporarily suspending this while I experiment with an Impl hierarchy which Caffe2 legacy needs.
class StridedCPUTensorImpl final : public TensorImpl {
  SmallVector<int64_t> stride_;

  StridedCPUTensorImpl() : TensorImpl(TypeIds::StridedCPUTensor) {};

  ArrayRef<int64_t> stride() const override {
    return stride_;
  }
  // Missing retain/release
};
*/

/*
// Example:

class OpenGLTensorImpl final : public TensorImpl {
  opengl_handle handle_;

  OpenGLTensorImpl() : TensorImpl(TypeIds::OpenCLTensor) {};
public:
  opengl_handle handle() const {
    return handle_;
  }
};
*/

}} // namespace c10::guts
