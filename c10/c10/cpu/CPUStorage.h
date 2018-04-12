#pragma once

#include "CPUContext.h"
#include "CPUAllocator.h"
#include "c10/guts/Retainable.h"

#include <cstddef>
#include <memory>
#include <functional>
#include <cstdlib>
#include <utility>
#include <c10/Assert.h>
#include <algorithm>

namespace c10 { namespace cpu {

// CPUStorage is NOT part of the public API
//
// Every Tensor is backed by a storage; multiple tensors may share the same storage
// (which is why we need an indirection.)
//
// This is implemented as a regular shared_ptr to reduce implementation
// cost.  Since it's internal we don't mind if it's a little clunky to use.
//
// In Caffe2, this was previously implemented directly as a std::shared_ptr.  Doing
// it this means that realloc is not possible because a std::shared_ptr only
// records the free() pointer.
//
// TODO: Consider making it possible to allocate CPUStorageImpl in the same block as a CPUTensor, so that
// allocating a tensor is only one dynamic allocation rather than two
// dzhulgakov: while I appreciate this approach - it's tricky as we'd need to override free/realloc functions and probably have higher cost.
//
// Roughly corresponds to THStorage from old ATen
// dzhulgakov: enabled_shared_from_this ?
class CPUStorageImpl {
  // smessmer to @ezyang: We might want to use folly::Function instead.
  //                      The folly::Function header is quite self contained, i.e. can be copied here without the rest of folly.
  // dzhulgakov: std::function is 32 bytes, we can be paranoid and probably optimize it to a single pointer as a common case is no custom deleter imho (just default to the context one)
  using data_t = std::unique_ptr<void, std::function<void(void*)>>;

  // NB: THAllocator is axed; instead, all you can pass now is a custom deleter,
  // which is enough tod o the important things.
  data_t data_;

  // dzhulgakov: do we ever need a flag to see whether storage is shared among several views or not? if we add enabled_shared_from_this then we can just use shared_ptr::use_count()
  ssize_t size_; // in bytes

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
  // [torch.LongTensor of size (4,)]
  //
  // >>> x[0] = 4      # Modify the resized tensor
  // >>> y
  //
  // 4
  // [torch.LongTensor of size (1,)]
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

public:
  // TODO: Permit allocator to be passed in through this function
  // TODO: Maybe ball up the allocator into a context

  CPUStorageImpl()
  : data_(nullptr)
  , size_(0)
  , resizable_(true)
  {}

  CPUStorageImpl(ssize_t size)
  : data_(globalCPUContext().getCPUAllocator()->malloc(size))
  , size_(size)
  , resizable_(true)
  {}

  // TODO: Make a more descriptive constructor for non-resizable things.  Note that since you're
  // using make_shared most of the time for storages, a static method won't cut it.
  CPUStorageImpl(data_t&& data, ssize_t size, bool resizable=true)
  : data_(std::move(data))
  , size_(size)
  , resizable_(resizable)
  {}

  // NB: Move constructor is legitimately used to destructively overwrite a storage, as in the case of a resize_()
  // TODO: explicitly declare permitted constructors.  (Consult my "rule of X" stuff...)

  // Straight up reimplementation of the ATen CPUStorage API

  const void* data_ptr() const {
    return data_.get();
  }

  void* data_ptr() {
    return data_.get();
  }

  ssize_t sizeBytes() const {
    return size_;
  }

  // THStorage_(swap)
  // This is used to implement resize, which needs to "replace" a Storage.
  // NB: This can be used to cause memory unsafety, as size bounds stored in Tensors may become invalid.
  // NB: if you have a CPUStorage x, this is NOT the same thing as x.swap(y).  All that does is twiddle
  // the shared pointers.  This actually swaps all the CONTENTS of the storage.  This is why I didn't call
  // it swap().
  void swap_contents(CPUStorageImpl& other) {
    // TODO: my IDE (clion) hates all uses of swap, for some reason
    std::swap(*this, other);
  }

  // NB: deleted newWithSizeN variants
  // NB: deleted setFlag/clearFlag
  // NB: deleted retain/free
  // NB: all of the new variants are sucked through the new constructor
  // NB: deleted fill/set/get

  // Meditation of THStorage_(resize)
  // Caffe2 behavior is when keep_data == false
  // dzhulgakov: Caffe2 has Reserve()/Extend() which is basically keep_data = true. I'd suggest to limit this behavior as much as possible, for example: allow only incremental growth and call it something more uncommon than 'resize'
  void resize_(ssize_t new_size, bool keep_data = true) {
    if (!resizable_) throw std::runtime_error("trying to resize storage that is not resizable");
    // TODO: Consider bringing back the old realloc path from TH?
    data_t old_data = std::move(data_);
    ssize_t old_size = size_;
    if (new_size == 0) {
      data_ = nullptr;
    } else {
      data_ = globalCPUContext().getCPUAllocator()->malloc(new_size);
    }
    size_ = new_size;
    if (old_data != nullptr && keep_data) {
      ssize_t copy_size = std::min(new_size, size_);
      if (copy_size > 0) {
        std::memcpy(data_.get(), old_data.get(), copy_size);
      }
      old_data.reset();
    }
  }

};

using CPUStorage = std::shared_ptr<CPUStorageImpl>;

// TODO: perfect forwarding helper constructor for make_shared

}} // namespace c10::cpu
