#pragma once

// NB: NO dependency on Tensor.h!!!

#include "c10/ArrayRef.h"
#include "c10/SmallVector.h"
#include "c10/Optional.h"
#include "c10/TypeId.h"
#include "c10/DataType.h"
#include <c10/DimVector.h>

#include "Retainable.h"
#include "Storage.h"

#include <vector>

namespace c10 {
  class Tensor;
}

// NB: It's called guts because it's short and gets the point across :)
// dzhulgakov: it's cool, less creative and longer version is "detail"
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
  // NB: shares_data from Caffe2 was axed, because it is SOLELY used to determine
  // check what the overall tensor usage is.  We can rewrite that code to
  // keep a mapping of storage base pointers that it has seen (these all
  // "count" the same), and perhaps add a bit to storage which tells us if
  // it is "external" or "internal" (external storages don't count for accounting
  // purposes.)

  // Used for dispatch on the object
  const TypeId type_id_;

  // The scalar type of elements stored in this tensor.  This contains
  // important information like "what is the sizes of the scalar element."
  // TODO: Pointer to scalar type means there's a possibly unnecessary indirection here!
  // TODO: This is going to be redundant with type_id_, so if we want to squeeze down sizes
  // we can make this a computed property from type_id_.
  DataType dtype_;

  DimVector sizes_;

  // dzhulgakov: I'd strongly suggest to keep around actual type, not just sizes to do type checking. Please look at TypeMeta - it solves a lot of issues
  // dzhulgakov: Caffe2 now supports fancy stuff like Tensor of std::string (or other types), TF too. I think we should handle it which requires some TypeMeta-like care to call constructors at right places. We can reuse it verbatim
  int64_t element_size_bytes_;

  // This lives here because we really want data_ptr() calculation to inline.
  // Note: storage->sizes() may be greater than the recorded sizes of the tensor
  // ezyang to @smessmer: Maybe we should consider using a never-null pointer.
  // If you do that a number of "is null" tests can be deleted.
  Storage storage_;

  // Note: In Torch this can be nonzero, because we support views into the
  // inside of tensors.  In historic Caffe2 this was always zero.
  // NB: This is BYTES!!!  Different from TH historically, which was scalar sizes.
  int64_t storage_offset_bytes_;

  // TODO: consider whether or not to inline cuda_device here.  Then we can change CPUStorage from
  // an "is-a" to "has-a" relationship and inline the storage struct in Tensor.

public:
  explicit TensorImpl(TypeId type_id, DataType dtype, Storage storage)
      : RetainableImpl()
      , type_id_(type_id)
      , sizes_()
      , dtype_(dtype)
      , storage_(storage)
  {};

  // TODO: Not sure about this...
  TypeId type_id() const {
    return type_id_;
  }

  ArrayRef<int64_t> sizes() const {
    return sizes_;
  }

  // Previously was type().scalarType() but I haven't committed to adding a Type object
  // to the design yet.
  DataType dtype() const {
    return dtype_;
  }

  int64_t storage_offset() const {
    return storage_offset_bytes_ / dtype_.itemsize();
  }

  // NB: In Caffe2, this quantity is CACHED.  For simplicity, we don't cache it for now, but consider
  // doing so.
  // NB: This was ported in from the TH backend (which we normally defer to.)
  // smessmer to @ezyang: Do we need this in here? Seems like something that can live as a non-member.
  // dzhulgakov: it should be a member imho and might be nice to cache it indeed
  int64_t numel() const {
    int64_t r = 1;
    for (auto s : sizes()) r *= s;
    return r;
  }

  virtual ArrayRef<int64_t> strides() const {
    throw std::runtime_error("TensorImpl::strides()");
  }

  int64_t dim() const {
    // dzhulgakov: this line is exactly why `sizes` is a bad name :)
    return static_cast<int64_t>(sizes().size());
  }

  void *data_ptr() const {
    if (!storage_) return nullptr;
    return static_cast<void*>(static_cast<char*>(storage_->data_ptr()) + storage_offset_bytes_);
  }

  virtual ~TensorImpl() = default;
};

// See design notes on Tensor.h, where this is hardcoded a few times.
// smessmer to @ezyang: What are your concerns about using an empty CPU tensor instead of an undefined one?
// ezyang to @smessmer: For example, there will be a method tensor, the semantics are x.tensor({2, 3}) will
//      create a 2x3 tensor of the same "type" as x.  If x is an empty CPU tensor, you'll get a CPU tensor,
//      instead of an error, which should have happened.  It just seems morally wrong to privilege empty CPU
//      tensors in this way.  Also, you don't get reliable pointer equality tests anymore.
class UndefinedTensorImpl final : public TensorImpl {
  UndefinedTensorImpl() : TensorImpl(TypeIds::Undefined, c10::undefined_dtype, nullptr) {};
public:
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
