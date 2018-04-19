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
#include <cinttypes>

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

  // TODO: This is not always valid, and needs to be treated accordingly
  // NB: reserved from Caffe2 axed; as there are TWO sizes, we can easily
  // implement the reserved pattern by having the storage be larger than the
  // sizes recorded in a Tensor.  Hooray!
  // dzhulgakov: superlike! :)
  // TODO: Move this to the parent class
  // Reminder: The way strides works is:
  //    sizes[0]*strides[0] + sizes[1]*strides[1] + ...
  // This means you can end up in weird situations.  Make sure to think about:
  //    strides[i] == 0 (broadcasting)
  //    strides[i] < 0 (negative strides)
  //    sizes[i] == 0 (useful to maintain sizes information!)
  //    strides[i] % sizes[i-1] != 0 (rolling window strides / not "embeddable")
  //    len(sizes) == 0 (scalars)
  // dzhulgakov: how much "strides analysis" do implementations usually do in TH?
  // See also https://ezyang.github.io/stride-visualizer/index.html
  DimVector strides_;

  // This lives here because we really want data_ptr() calculation to inline.
  // Note: storage->sizes() may be greater than the recorded sizes of the tensor
  // ezyang to @smessmer: Maybe we should consider using a never-null pointer.
  // If you do that a number of "is null" tests can be deleted.
  Storage storage_;

  // Note: In Torch this can be nonzero, because we support views into the
  // inside of tensors.  In historic Caffe2 this was always zero.
  // NB: This is BYTES!!!  Different from TH historically, which was scalar sizes.
  int64_t storage_offset_bytes_;

  // TODO: need boolean flags to specify whether or not elements like strides and storage are valid.

  // TODO: consider whether or not to inline cuda_device here.  Then we can change CPUStorage from
  // an "is-a" to "has-a" relationship and inline the storage struct in Tensor.

public:
  explicit TensorImpl(TypeId type_id, DataType dtype, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, Storage storage, int64_t storage_offset_bytes)
      : RetainableImpl()
      , type_id_(type_id)
      , sizes_(sizes)
      , strides_(strides)
      , dtype_(dtype)
      , storage_(storage)
      , storage_offset_bytes_(storage_offset_bytes)
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

  // TODO: precompute this value
  bool is_contiguous() const {
    int64_t z = 1;
    for (int64_t d = dim()-1; d >= 0; d++) {
      // NB: Strides don't affect contiguity when size is zero or one,
      // because you never multiply against the stride with a nonzero index.
      // Historical Torch had a more stringent requirement, but @SsnL changed it.
      if (sizes()[d] <= 1) continue;
      if (strides()[d] != z) return false;
      z *= sizes()[d];
    }
    return true;
  }

  // Low level functions which should not be propagated to the Tensor class

  // TODO: Can we make it impossible to accidentally out-of-bound read when you set the
  // sizes and strides?  This is not necessarily the right API, but it is one that
  // sort of works
  void _set_sizes_and_strides(ArrayRef<int64_t> new_sizes_ref, ArrayRef<int64_t> new_strides_ref) {
    // Extra copy here to make sure bad things don't happen when new_sizes_ref aliases sizes.
    // Hypothetically we could test for aliasing by doing some pointer comparisons but
    // I am not sure if this is defined behavior in C.
    // Note that you WILL run into this problem in practice, because SmallVector's assign implementation clears the
    // SmallVector first
    DimVector new_sizes(new_sizes_ref);
    sizes_.swap(new_sizes);
    DimVector new_strides(new_strides_ref);
    strides_.swap(new_strides);
  }

  // Channeling Caffe2 Tensor::Shrink
  // TODO: Should this really be defined here?  Something more general?
  void _shrink(int64_t outer_dim_new_size) {
    C10_CHECK(sizes().size() >= 1, "shrink: tensor is 0D (scalar), but expected at least 1D tensor");
    C10_CHECK(outer_dim_new_size < sizes().at(0),
              "shrink: new outer dimension size %" PRId64 " must be smaller than current size %" PRId64,
              outer_dim_new_size, sizes().at(0));
    sizes_[0] = outer_dim_new_size;
  }

  // TODO: This is hella unsafe.  Part of PyTorch public API (but it accepted a storage rather than a Tensor)
  // A more flexible version of Caffe2 Tensor::ShareData
  void _set(TensorImpl* src, int64_t storage_offset, ArrayRef<int64_t> new_sizes, ArrayRef<int64_t> new_strides) {
    // In principle we could allow the non-contiguous case, but the original API accepted storages
    // (which were guaranteed to contiguous); thus shall we.
    C10_CHECK(src->is_contiguous(), "set: src tensor must be contiguous");
    // TODO: I guess if type_id() has refinements this check is no longer correct
    C10_CHECK(type_id() == src->type_id(),
              "src tensor has type ", src->type_id(), ", but expected ", type_id(), " (from destination tensor)");
    C10_CHECK(dtype() == src->dtype(),
              "src tensor has dtype ", src->dtype(), ", but expected dtype ", dtype(), " (from destination tensor)");
    if (src != this) storage_ = src->storage_;
    // NB: The storage_offset is relative!  This means you can't just translate
    //    x.set_(y.storage(), ...)
    // into
    //    x.set_(y, ...)
    // because the first call would have "lost" the offset in y, but second call preserves it.
    // Really, what you need is for y.storage() to return a contiguous, 1D tensor that spans
    // all of storage, then no translation necessary.  TODO: implement this
    storage_offset_bytes_ = src->storage_offset_bytes_ + storage_offset * dtype_.itemsize();
    _set_sizes_and_strides(new_sizes, new_strides);
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
  UndefinedTensorImpl() : TensorImpl(TypeIds::Undefined, c10::undefined_dtype, {}, {}, nullptr, 0) {};
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
