#pragma once

#include <c10/guts/Retainable.h>
#include <c10/Error.h>
#include <c10/DataType.h>

#include <cstddef>
#include <memory>
#include <functional>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <algorithm>

namespace c10 { namespace guts {

/**
 * Every Tensor is backed by a storage; multiple tensors may share the same storage
 * (which is why we need an indirection.)  We have a base class for storage so
 * that we can compute the data pointer without a virtual dispatch.
 *
 * Storage is NOT part of the public API.  Don't add it to methods in Tensor.
 */
// dzhulgakov: enabled_shared_from_this ?
// NB: Feel free to add enable_shared_from_this if you need it
//
// NB: Purposely not CRTP'ified, so we can rely on convertibility of pointers of
// subclasses.
class StorageImpl {
  // Mmm.... incestuous :>
protected:

  // smessmer to @ezyang: We might want to use folly::Function instead.
  //                      The folly::Function header is quite self contained, i.e. can be copied here without the rest of folly.
  // NB: The reason to use folly::Function is that it allows functions to be moved, rather than only copied
  // dzhulgakov: std::function is 32 bytes, we can be paranoid and probably optimize it to a single pointer as
  // a common case is no custom deleter imho (just default to the context one)
  // TODO: Replace the std::function here with something better; probably Folly function, or something else
  using data_t = std::unique_ptr<void, std::function<void(void *)>>;

  // NB: THAllocator is axed; instead, all you can pass now is a custom deleter,
  // which is enough to do the important things.
  data_t data_;

  // dzhulgakov: do we ever need a flag to see whether storage is shared among several views or not?
  // if we add enabled_shared_from_this then we can just use shared_ptr::use_count()
  // ezyang: Yeah, I think enable_shared_from_this is the logical choice for this, assuming that
  // you really really really need to get it from a raw StorageImpl, which should actually be
  // fairly rare, since it's not an operation that is all that useful for internal implementation
  // inside CPUStorageImpl

  // ezyang: In the original draft of this class, sizes was recorded in bytes because storage was content
  // agnostic.  Now that we may possibly need to placement-new/placement-delete when storage resize, storage
  // must be content-aware, and now it is less clear if the sizes should be counted in bytes or number of
  // elements.  (Bytes is more likely to be the number you need, but number of elements reduces the number
  // of unrepresentable states.)  For reference, TH used to count the number of elements, but it also
  // created a copy of the struct per data type, so the sizes of the element was statically known.
  int64_t size_bytes_; // in bytes

  // The scalar type of this storage.  We need this in case we need to do placement-new/placement-delete
  // after allocation
  DataType data_type_;

  // Is this storage resizable?  If it comes externally, or has been shared to some external system, it may not be.
  // Corresponds to TH_STORAGE_RESIZABLE.
  //
  // Why do we need this flag?  The trouble comes from operations which *resize their underlying
  // storages.*  For example, consider the following PyTorch transcript:
  //
  // >>> x = torch.tensor([0,1,2])
  // >>> y = x[0:1]    # Make another view on this tensor
  // >>> x.resize_(4)  # Resize the underlying storage
  //
  // 0
  // 1
  // 2
  // 1
  // [torch.LongTensor of sizes (4,)]
  //
  // >>> x[0] = 4      # Modify the resized tensor
  // >>> y
  //
  // 4
  // [torch.LongTensor of sizes (1,)]
  //
  // First, notice that the only way to actually implement this behavior is to modify the actual underlying
  // storage: in general, if we resize a block of memory, it means that the pointer to that memory updates.
  // We need this pointer to update for ALL tensors which view the same data; thus the modification happens
  // in storage.  Second, notice that if we swap out the storage at one tensor to something that is
  // unresizable, ALL views on that tensor need to reject further resizes.  So the correct place to store
  // this information is indeed in Storage.  This is not an operation you'd ever need to actually do,
  // but it makes clear where this information semantically belongs.
  //
  // dzhulgakov: omg, do we have to support this behavior? sounds kind of too fancy - do people rely on it often? In my mind if one resizes the storage all views are de-facto invalidated - I'd basically just allocate the new storage instead.
  // dzhulgakov: we should document this behavior if we keep it - it's pretty non-obvious
  //
  // TODO: pack this boolean flag?
  bool resizable_;

  // NB: I axed TH_STORAGE_FREEMEM; the expectation is you install a no-op
  // deallocator if this is what you want.
  // NB: I axed TH_STORAGE_VIEW; it makes things complicated for not a good enough
  // reason
  // NB: I axed TH_STORAGE_REFCOUNTED; it seems to always be turned on.  If we do want
  // this, it would be a good fit for the Retainable class.

  // NB: the constructors are protected because you shouldn't be allowed to use them
  // if you're a generic end-user; in general, operating on a StorageImpl won't work
  // because you will always actually have a CPUStorageImpl or something similar, and
  // so if you use the base class you will miss methods.

  StorageImpl(DataType data_type)
      : data_(nullptr), size_bytes_(0), data_type_(data_type), resizable_(true) {}

  // TODO: Make a more descriptive constructor for non-resizable things.  Note that since you're
  // using make_shared most of the time for storages, you probably just want to make another
  // top-level 'make' function.
  StorageImpl(DataType data_type, data_t &&data, int64_t size, bool resizable = true)
      : data_(std::move(data)), size_bytes_(size), data_type_(data_type), resizable_(resizable) {}

  // NB: Move constructor is legitimately used to destructively overwrite a storage, as in the case of a resize_()
  // TODO: explicitly declare permitted constructors.  (Consult my "rule of X" stuff...)

  // Rule of Five
  StorageImpl(StorageImpl&&) = default;
  ~StorageImpl() = default;
  StorageImpl& operator=(StorageImpl&&) = default;

public:

  StorageImpl(const StorageImpl&) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;

  const void *data_ptr() const {
    return data_.get();
  }

  void *data_ptr() {
    return data_.get();
  }

  int64_t size_bytes() const {
    return size_bytes_;
  }
};

using Storage = std::shared_ptr<StorageImpl>;

// TODO: perfect forwarding helper constructor for make_shared

}} // namespace c10::guts
