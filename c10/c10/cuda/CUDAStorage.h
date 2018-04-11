#pragma once

#include <memory>
#include <functional>

namespace c10 { namespace guts {

// See CPUStorage for some commentary
class CUDAStorageImpl {
  using data_t = std::unique_ptr<void, std::function<void(void*)>>;

  data_t data_;
  std::size_t size_;
  std::size_t element_size_;
  bool resizable_;
  // Which CUDA device this storage lives on
  int device_;

public:
  CUDAStorageImpl(data_t&& data, std::size_t element_size, std::size_t size, bool resizable=true)
  : data_(std::move(data))
  , element_size_(element_size)
  , size_(size)
  , resizable_(resizable)
  {
    CUDA_CHECK(cudaGetDevice(&device_));
  }

  inline const void* data_ptr() const {
    return data_.get();
  }

  inline void* data_ptr() {
    return data_.get();
  }

  // THStorage_(size)
  inline std::size_t size() const {
    return size_;
  }

  // THStorage_(elementSize)
  // I'm... not really sure why we need to store this in here.
  inline std::size_t elementSize() const {
    return element_size_;
  }

  // THStorage_(swap)
  // This is used to implement resize, which needs to "replace" a Storage.
  // NB: This can be used to cause memory unsafety, as size bounds stored in Tensors may become invalid.
  // NB: if you have a CPUStorage x, this is NOT the same thing as x.swap(y).  All that does is twiddle
  // the shared pointers.  This actually swaps all the CONTENTS of the storage.  This is why I didn't call
  // it swap().
  void swap_contents(CUDAStorageImpl& other) {
    // TODO: my IDE (clion) hates all uses of swap, for some reason
    std::swap(*this, other);
  }

  // NB: deleted set/get
};

using CUDAStorage = std::shared_ptr<CUDAStorageImpl>;

}} // namespace c10::guts
