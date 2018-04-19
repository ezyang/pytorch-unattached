#pragma once

#include <c10/cpu/CPUContext.h>
#include <c10/cpu/CPUAllocator.h>
#include <c10/guts/Retainable.h>
#include <c10/guts/Storage.h>
#include <c10/Error.h>
#include <c10/DataType.h>

#include <cstddef>
#include <memory>
#include <functional>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <cinttypes>

namespace c10 { namespace cpu {

// TODO: Consider making it possible to allocate CPUStorageImpl in the same block as a CPUTensor, so that
// allocating a tensor is only one dynamic allocation rather than two
// dzhulgakov: while I appreciate this approach - it's tricky as we'd need to override free/realloc functions and probably have higher cost.

/**
 * Storage of a CPU tensor.  Multiple CPU tensors can have the same storage (meaning that they
 * share data.)
 */
class CPUStorageImpl final : public guts::StorageImpl {
public:
  // TODO: Permit allocator to be passed in through this function

  CPUStorageImpl(DataType data_type)
      : StorageImpl(data_type)
  {}

  // Rule of Five (make these constructors public)
  CPUStorageImpl(const CPUStorageImpl&) = default;
  CPUStorageImpl(CPUStorageImpl&&) = default;
  ~CPUStorageImpl() = default;
  CPUStorageImpl& operator=(const CPUStorageImpl&) = default;
  CPUStorageImpl& operator=(CPUStorageImpl&&) = default;

  // NB: Move constructor is legitimately used to destructively overwrite a storage, as in the case of a resize_()
  // TODO: explicitly declare permitted constructors.  (Consult my "rule of X" stuff...)

  // THStorage_(swap)
  // This is used to implement resize, which needs to "replace" a Storage.
  // NB: This can be used to cause memory unsafety, as sizes bounds stored in Tensors may become invalid.
  // NB: if you have a CPUStorage x, this is NOT the same thing as x.swap(y).  All that does is twiddle
  // the shared pointers.  This actually swaps all the CONTENTS of the storage.  This is why I didn't call
  // it swap().
  void swap_contents(CPUStorageImpl& other) {
    // TODO: my IDE (clion) hates all uses of swap, for some reason
    std::swap(*this, other);
  }

  void copy_(const void* src, int64_t copy_size_bytes) {
    if (copy_size_bytes <= 0) return;
    if (auto copy = data_type_.copy()) {
      // Swapped argument order?! How confusing!
      copy(src, data_.get(), copy_size_bytes / data_type_.itemsize());
    } else {
      std::memcpy(data_.get(), src, static_cast<size_t>(copy_size_bytes));
    }
  }

  // NB: deleted newWithSizeN variants
  // NB: deleted setFlag/clearFlag
  // NB: deleted retain/free
  // NB: all of the new variants are sucked through the new constructor
  // NB: deleted fill/set/get

  // Meditation of THStorage_(resize)
  // Caffe2 behavior is when keep_data == false
  // dzhulgakov: Caffe2 has Reserve()/Extend() which is basically keep_data = true. I'd suggest to limit this
  // behavior as much as possible, for example: allow only incremental growth and call it something more uncommon
  // than 'resize'
  //
  // NB: This has the logic for Caffe2-style placement-new/placement-delete
  void resize_(int64_t new_size_bytes, bool keep_data = true) {
    C10_ASSERT(new_size_bytes % data_type_.itemsize() == 0,
      "requested new size of ", new_size_bytes, " bytes is not a multiple of ",
      data_type_.itemsize(), "(aka, the item size of dtype ", data_type_, ")");
    auto new_size_elems = new_size_bytes / data_type_.itemsize();
    if (!resizable_) throw std::runtime_error("trying to resize storage that is not resizable");
    // TODO: Consider bringing back the old realloc path from TH?
    data_t old_data = std::move(data_);
    int64_t old_size = size_bytes_;
    if (new_size_bytes == 0) {
      data_ = nullptr;
    } else {
      void* raw_data;
      std::function<void(void*)> deleter;
      std::tie(raw_data, deleter) = globalCPUContext().getCPUAllocator()->malloc(new_size_bytes);
      // TODO: Exception safety?!  If an exception happens before we allocate the unique_ptr
      // we will lose this data.
      if (auto dtor = data_type_.dtor()) {
        // TODO: It is too bad we can't move capture 'deleter'; an unnecessary
        // copy happens here. (It also happened in the old Caffe2 version of this code)
        auto deleter_with_dtor = [dtor, deleter, new_size_elems](void* p) {
          dtor(p, new_size_elems);
          deleter(p);
        };
        // TODO: It's probably an error if ctor is set but not dtor
        data_ = {raw_data, deleter_with_dtor};
      } else {
        data_ = {raw_data, deleter};
      };
      // TODO: Still exception safety?!?!  If an exception happens before we placement-new,
      // we will attempt to deallocate the data using the placement-deleter, which is obviously
      // not going to work
      if (auto ctor = data_type_.ctor()) {
        ctor(data_.get(), new_size_bytes / data_type_.itemsize());
      }
    }
    size_bytes_ = new_size_bytes;
    if (old_data != nullptr && keep_data) {
      int64_t copy_size_bytes = std::min(new_size_bytes, size_bytes_);
      copy_(old_data.get(), copy_size_bytes);
      old_data.reset();
    }
  }

};

using CPUStorage = std::shared_ptr<CPUStorageImpl>;

// TODO: perfect forwarding helper constructor for make_shared

}} // namespace c10::cpu
