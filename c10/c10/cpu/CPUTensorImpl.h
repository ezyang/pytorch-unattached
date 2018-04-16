#include <c10/DimVector.h>
#include "c10/guts/TensorImpl.h"
#include "c10/Optional.h"

#include "CPUStorage.h"

#include <numeric>
#include <cmath>

namespace c10 { namespace cpu {

// Return the strides corresponding to a contiguous layout of size.
DimVector contiguous_strides(ArrayRef<int64_t> size) {
  DimVector v(size.size());
  int64_t total_size = 1;
  for (int64_t d = size.size() - 1; d >= 0; d--) {
    v[d] = total_size;
    total_size *= size[d];
  }
  return v;  // RVO
}

// Given size and strides, return the lowest and highest indices (inclusive-exclusive) which may
// be accessed with them.
// TODO: Refactor this into a utility header file
std::pair<int64_t, int64_t> compute_extent(ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
  // Watermarks are inclusive.  NB: watermarks can be negative! Careful!
  // NB: when size is empty, we get {0, 1}; that is to say,
  // there is ONE valid location.  This is correct!
  int64_t low_watermark = 0; // inclusive
  int64_t high_watermark = 1; // exclusive
  for (int64_t d = size.size() - 1; d >= 0; d--) {
    // TODO: This special case is so irritating.  But if we don't apply it,
    // this function returns {0, 1} when you pass it size {0} stride {0}.
    if (size[d] == 0) return {0, 0};
    C10_ASSERT(size[d] > 0);
    if (stride[d] >= 0) {
      high_watermark += (size[d] - 1) * stride[d];
    } else {
      low_watermark += (size[d] - 1) * stride[d];
    }
  }
  return {low_watermark, high_watermark};
};

int64_t required_new_storage_size_bytes(
    ScalarType scalar_type,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    int64_t storage_offset_bytes) {
  int64_t low_watermark, high_watermark;
  std::tie(low_watermark, high_watermark) = compute_extent(size, stride);
  if (low_watermark * scalar_type.itemsize() + storage_offset_bytes < 0) {
    throw std::runtime_error("Cannot resize past beginning of tensor");
  }
  return high_watermark * scalar_type.itemsize() + storage_offset_bytes;
}

int64_t product(ArrayRef<int64_t> xs) {
  return std::accumulate(xs.begin(), xs.end(), 1, std::multiplies<int64_t>());
}

// Everything is int64_t to prevent us from accidentally doing a signed-unsigned operation
// which is basically never what you want.  But using int64_t instead of int64_t shuts
// up the compiler about size_t conversions from standard library.

class CPUTensorImpl final : public guts::TensorImpl {
  // Note: storage->size() may be greater than the recorded size of the tensor
  // ezyang to @smessmer: Maybe we should consider using a never-null pointer.
  // If you do that a number of "is null" tests can be deleted.
  CPUStorage storage_;

  // Note: In Torch this can be nonzero, because we support views into the
  // inside of tensors.  In historic Caffe2 this was always zero.
  // NB: This is BYTES!!!  Different from TH historically, which was scalar size.
  int64_t storage_offset_bytes_;

  // NB: shares_data from Caffe2 was axed, because it is SOLELY used to determine
  // check what the overall tensor usage is.  We can rewrite that code to
  // keep a mapping of storage base pointers that it has seen (these all
  // "count" the same), and perhaps add a bit to storage which tells us if
  // it is "external" or "internal" (external storages don't count for accounting
  // purposes.)

  // NB: reserved from Caffe2 axed; as there are TWO sizes, we can easily
  // implement the reserved pattern by having the storage be larger than the
  // size recorded in a Tensor.  Hooray!
  // dzhulgakov: superlike! :)
  // TODO: Move this to the parent class
  // Reminder: The way stride works is:
  //    size[0]*stride[0] + size[1]*stride[1] + ...
  // This means you can end up in weird situations.  Make sure to think about:
  //    stride[i] == 0 (broadcasting)
  //    stride[i] < 0 (negative strides)
  //    size[i] == 0 (useful to maintain size information!)
  //    stride[i] % size[i-1] != 0 (rolling window strides / not "embeddable")
  //    len(size) == 0 (scalars)
  // dzhulgakov: how much "stride analysis" do implementations usually do in TH?
  // See also https://ezyang.github.io/stride-visualizer/index.html
  DimVector stride_;

  // TODO: consider whether or not to inline cuda_device here.  Then we can change CPUStorage from
  // an "is-a" to "has-a" relationship and inline the storage struct in Tensor.
public:
  CPUTensorImpl(ScalarType scalar_type, const CPUStorage& storage)
  : TensorImpl(TypeIds::CPUTensor, scalar_type)
      , storage_(storage)
  {
    C10_ASSERT(storage);
  };

  void *data_ptr() const override {
    if (!storage_) return nullptr;
    return static_cast<void*>(static_cast<char*>(storage_->data_ptr()) + storage_offset_bytes_);
  }

  // Hacked up operators

  static Tensor HACK_tensor(ScalarType scalar_type) {
    auto storage = std::make_shared<CPUStorageImpl>(scalar_type);
    return Tensor::_fromImpl(new CPUTensorImpl(scalar_type, storage));
  }

  // NB: this is generic (assuming you pass in the backend dispatcher)
  static Tensor HACK_tensor(ScalarType scalar_type, ArrayRef<int64_t> size, ArrayRef<int64_t> stride) {
    auto r = HACK_tensor(scalar_type);
    r.resize_(size, stride);
    return r;
  }

  // Channeling Caffe2 Tensor::Tensor(const vector<TIndex>& dims, const vector<T>& values, Context* context)
  // NB: this is generic
  template <typename T>
  static Tensor HACK_tensor(ArrayRef<int64_t> size, std::vector<T> data) {
    auto r = HACK_tensor(c10::scalar_type<T>, size, contiguous_strides(size));
    C10_CHECK(r.numel() == data.size());
    r.template copy_<T>(data);
    return r;
  }

  // Channeling Caffe2 Tensor::Tensor(const T& value, Context* context)
  // Create a scalar tensor from a single value
  // NB: this is generic
  // NB: the test that T is_scalar prevents this template from clobbering other
  // overloads (though, this may not be an issue in C10, since Context is no longer
  // a templated argument, so C++'s rule of preferring a non-template function over
  // a templated one might actually work.)
  template <typename T,
            typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  static Tensor HACK_tensor(const T& value) {
    auto r = HACK_tensor(c10::scalar_type<T>, {}, {});
    r.template copy_<T>({&value, 1});
    return r;
  }

  // Channeling Caffe2 Tensor::Tensor(const T& value, Context* context)
  void HACK_copy_(ScalarType s, const void* p, int64_t size_bytes) override {
    C10_CHECK(s == scalar_type_);
    storage_->copy_(p, size_bytes);
  }

  // Channeling THTensor_(resizeNd)
  // If aggressive = true, we will always try to free up old memory (this means
  // we always have to do a reallocation).  Torch default behavior was to
  // keep the old data around; Caffe2's behavior is to do a full reallocate.
  // NB: This code is GENERIC for all strided tensors.
  // When stride is not set, it is assumed you wanted to preserve the original stride
  // NB: resizeNd used to accept NULL stride, in which case contiguous strides are
  // assumed.  To keep this function simple, we FORCE the callee to pass new_stride;
  // it's a simple matter to compute what the appropriate contiguous strides for a
  // tensor are.
  // WARNING: BC-breaking change; previously a negative number was assumed to mean
  // "compute whatever the appropriate contiguous stride is."  But this didn't even
  // work in all cases; when determining if sizes/strides had changed, resizeNd would
  // incorrectly assume the original tensor was
  // contiguously strided for every negative index, even when it was not.
  // See also https://github.com/pytorch/pytorch/issues/229
  void HACK_resize_(ArrayRef<int64_t> new_size, ArrayRef<int64_t> new_stride, bool keep_data) override {
    C10_ASSERT(new_size.size() == new_stride.size());
    bool unchanged = new_size.equals(size()) && new_stride.equals(stride());
    if (unchanged) return;
    auto new_size_bytes = required_new_storage_size_bytes(scalar_type_, new_size, new_stride, storage_offset_bytes_);
    size_.assign(new_size.begin(), new_size.end());
    stride_.assign(new_stride.begin(), new_stride.end());
    // NB: In the old TH code, it was permissible for Storage to be a nullptr at this point.
    // We have tightened the internal invariants.  I put the ASSERT back in where the old
    // test for storage_ being nullptr would have been.
    C10_ASSERT(storage_);
    bool needs_resize =
        // not enough space, OR
        new_size_bytes > storage_->sizeBytes() ||
        // we're not allowed to keep the old storage on a shrink, OR
        !globalCPUContext().keepOnShrink() ||
        // we shrunk greater than the maximum "keep on shrink" bytes.
        storage_->sizeBytes() - new_size_bytes > globalCPUContext().maxKeepOnShrinkBytes();
    if (needs_resize) {
      storage_->resize_(new_size_bytes, keep_data);
    }
  }

  // Channeling Caffe2 Tensor::Reserve(const std::vector<T>& newCapacity, ContextForCopy* context)
  // TODO: Consider also having a direct "numels" variant.  Note that this version accounts
  // correctly for strides
  void HACK_reserve_(ArrayRef<int64_t> new_size) override {
    auto new_size_bytes = required_new_storage_size_bytes(scalar_type(), new_size, stride(), storage_offset_bytes_);
    if (new_size_bytes > storage_->sizeBytes()) {
      // NB: Size of this tensor is unchanged!
      storage_->resize_(new_size_bytes, true);
    }
  }

  // Channeling Caffe2 Tensor::Extend(TIndex num, float growthPct, ContextForCopy* context)
  void HACK_extend_(int64_t num, double growthPct) override {
    C10_CHECK(dim() >= 1);
    DimVector new_size{size()};
    new_size[0] += num;
    // NB: Do not need to test for storage_ == nullptr as it is assumed to
    // have been initialized
    auto tentative_new_size_bytes = required_new_storage_size_bytes(scalar_type_, new_size, stride(), storage_offset_bytes_);
    if (tentative_new_size_bytes <= storage_->sizeBytes()) {
      size_ = new_size;
      return;
    }
    // Compute the true size increase, to ensure extend() amortizes correctly
    new_size[0] = std::max(new_size[0], static_cast<int64_t>(std::ceil(size()[0] * (growthPct + 100) / 100)));
    HACK_resize_(new_size, stride(), true);
  }

  /*
  // Channeling Caffe2 Tensor::CopyFrom(const Tensor<SrcContext>& src, ContextForCopy* context)
  // and Tensor::CopyFrom(const Tensor<SrcContext>& src)
  // This function is deferred until multiple dispatch is online, as it can only be conveniently
  // implemented inside the multiple dispatch framework
  void HACK_copy_(Tensor src) {
  }
   */
};

}} // namespace c10::cpu
