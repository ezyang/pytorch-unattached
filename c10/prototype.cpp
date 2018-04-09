#include <cstdint>

// Placeholder for asserts; ignore them for now
#define AT_ASSERT(cond, ...)

// ArrayRef (comes from LLVM, ATen uses it, we think it's pretty good)

//===--- ArrayRef.h - Array Reference Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// ATen: modified from llvm::ArrayRef.
// removed llvm-specific functionality
// removed some implicit const -> non-const conversions that rely on
// complicated std::enable_if meta-programming
// removed a bunch of slice variants for simplicity...

#include <array>
#include <iterator>
#include <vector>

/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
template<typename T>
class ArrayRef {
             public:
             typedef const T *iterator;
             typedef const T *const_iterator;
             typedef size_t size_type;

             typedef std::reverse_iterator<iterator> reverse_iterator;

             private:
             /// The start of the array, in an external buffer.
             const T *Data;

             /// The number of elements.
             size_type Length;

             public:
             /// @name Constructors
             /// @{

             /// Construct an empty ArrayRef.
             /*implicit*/ ArrayRef() : Data(nullptr), Length(0) {}

             /// Construct an ArrayRef from a single element.
             /*implicit*/ ArrayRef(const T &OneElt)
             : Data(&OneElt), Length(1) {}

             /// Construct an ArrayRef from a pointer and length.
             /*implicit*/ ArrayRef(const T *data, size_t length)
             : Data(data), Length(length) {}

             /// Construct an ArrayRef from a range.
             ArrayRef(const T *begin, const T *end)
             : Data(begin), Length(end - begin) {}

             /// Construct an ArrayRef from a std::vector.
             template<typename A>
             /*implicit*/ ArrayRef(const std::vector<T, A> &Vec)
             : Data(Vec.data()), Length(Vec.size()) {}

             /// Construct an ArrayRef from a std::array
             template <size_t N>
             /*implicit*/ constexpr ArrayRef(const std::array<T, N> &Arr)
             : Data(Arr.data()), Length(N) {}

             /// Construct an ArrayRef from a C array.
             template <size_t N>
             /*implicit*/ constexpr ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}

             /// Construct an ArrayRef from a std::initializer_list.
             /*implicit*/ ArrayRef(const std::initializer_list<T> &Vec)
             : Data(Vec.begin() == Vec.end() ? (T*)nullptr : Vec.begin()),
             Length(Vec.size()) {}

             /// @}
             /// @name Simple Operations
             /// @{

             iterator begin() const { return Data; }
             iterator end() const { return Data + Length; }

             reverse_iterator rbegin() const { return reverse_iterator(end()); }
             reverse_iterator rend() const { return reverse_iterator(begin()); }

             /// empty - Check if the array is empty.
             bool empty() const { return Length == 0; }

             const T *data() const { return Data; }

             /// size - Get the array size.
             size_t size() const { return Length; }

             /// front - Get the first element.
             const T &front() const {
             AT_ASSERT(!empty(), "Empty list!");
             return Data[0];
             }

             /// back - Get the last element.
             const T &back() const {
             AT_ASSERT(!empty(), "Empty list!");
             return Data[Length-1];
             }

             /// equals - Check for element-wise equality.
             bool equals(ArrayRef RHS) const {
             if (Length != RHS.Length)
             return false;
             return std::equal(begin(), end(), RHS.begin());
             }

             /// slice(n, m) - Chop off the first N elements of the array, and keep M
             /// elements in the array.
             ArrayRef<T> slice(size_t N, size_t M) const {
             AT_ASSERT(N+M <= size(), "Invalid specifier");
             return ArrayRef<T>(data()+N, M);
             }

             /// slice(n) - Chop off the first N elements of the array.
             ArrayRef<T> slice(size_t N) const { return slice(N, size() - N); }

             /// @}
             /// @name Operator Overloads
             /// @{
             const T &operator[](size_t Index) const {
             return Data[Index];
             }

             /// Vector compatibility
             const T &at(size_t Index) const {
             AT_ASSERT(Index < Length, "Invalid index!");
             return Data[Index];
             }

             /// Disallow accidental assignment from a temporary.
             ///
             /// The declaration here is extra complicated so that "arrayRef =TypeId( )"
             /// continues to select the move assignment operator.
             template <typename U>
             typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
             operator=(U &&Temporary) = delete;

             /// Disallow accidental assignment from a temporary.
             ///
             /// The declaration here is extra complicated so that "arrayRef = {}"
             /// continues to select the move assignment operator.
             template <typename U>
             typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
             operator=(std::initializer_list<U>) = delete;

             /// @}
             /// @name Expensive Operations
             /// @{
             std::vector<T> vec() const {
             return std::vector<T>(Data, Data+Length);
             }

             /// @}
             /// @name Conversion operators
             /// @{
             operator std::vector<T>() const {
             return std::vector<T>(Data, Data+Length);
             }

             /// @}
             };


//===--- TypeId.h -----------------------------------------------*- C++ -*-===//

// A compact identifier which stores all of the information necessary to
// carry out a dispatch on a type.  This is NOT NECESSARILY in one-to-one
// correspondence with the type hierarchy of TensorImpl, because we may decide
// that we want to refine dispatch on a runtime property of a tensor which is
// NOT reflected by the class hierarchy.
//
// Other note: there is no NATIVE notion of a subtyping relationship between
// these TypeIds.  We are planning to design one but we haven't decided on
// its specifics yet.
//
//    ezyang: CC smessmer; I know you wanted to have this line up exactly
//    with the concrete TensorImpl subclasses, but I don't want to commit
//    to that at the moment
//
// TODO: Does this also contain per Tensor properties, like contiguity?
class TypeId final {
  int64_t id_;
  explicit constexpr TypeId(int64_t id) : id_(id) {}
  friend class TypeIds;
};
class TypeIds final {
public:
  static constexpr TypeId Undefined = TypeId(0);
  static constexpr TypeId CPUTensor = TypeId(1);
  static constexpr TypeId StridedCPUTensor = TypeId(2);
  static constexpr TypeId OpenCLTensor = TypeId(3);
};
// @ezyang: PODs need default constructor. Probably don't want this for TypeId.
//static_assert(std::is_pod<TypeId>());

//===--- TensorImpl.h -----------------------------------------------*- C++ -*-===//

// TODO: Fill in an actual SmallVector implementation here.  Both Folly and LLVM's
// implementation are a bit annoying to make standalone.  Maybe this can be made
// simpler by assuming T is POD.
template <typename T>
using SmallVector = std::vector<T>;

// For now: try using empty tensors for type (I think we'll probably add a Type
// object)

// NB: Use of virtual functions means that this is NOT a plain old data class.
// This means that we don't get inlineable C API functions which access the representation
// directly
class TensorImpl {
  // Used for dispatch on the object
  TypeId type_id_;
  // We have an interesting problem here, which regards our short term plan for
  // integrating PyTorch and Caffe2 without having to rewrite all of Torch/Caffe2's
  // operators.  Recall that both Torch and Caffe2 have their own, existing tensor
  // types, which record sizes by themselves.
  SmallVector<int64_t> size_;
  // Refcounting
  std::atomic<int> refcount_;
  friend class Tensor;
public:
  explicit TensorImpl(TypeId type_id) : type_id_(type_id), refcount_(1) {};
  // Inline?  Virtual?  See the admonition above.
  virtual ArrayRef<int64_t> size() const {
    return size_;
  }
  virtual ArrayRef<int64_t> stride() const {
    throw std::runtime_error("TensorImpl::stride()");
  }
  virtual int64_t dim() const {
    return static_cast<int64_t>(size().size());
  }
  virtual void* data_ptr() const {
    throw std::runtime_error("TensorImpl::data_ptr()");
  }
  virtual ~TensorImpl() {}
};

// See design notes on Tensor.h, where this is hardcoded a few times.
class UndefinedTensorImpl : public TensorImpl {
  UndefinedTensorImpl() : TensorImpl(TypeIds::Undefined) {};
  static UndefinedTensorImpl singleton_;
public:
  ArrayRef<int64_t> size() const override {
    throw std::runtime_error("UndefinedTensorImpl::size()");
  }
  int64_t dim() const override {
    throw std::runtime_error("UndefinedTensorImpl::dim()");
  }
  static inline UndefinedTensorImpl* singleton() {
    return &singleton_;
  }
};

class CPUTensorImpl : public TensorImpl {
  void* data_ptr_;
public:
  CPUTensorImpl() : TensorImpl(TypeIds::CPUTensor), data_ptr_(nullptr) {};
  void* data_ptr() const override {
    return data_ptr_;
  }
  // Missing retain/release
};

class StridedCPUTensorImpl : public TensorImpl {
  SmallVector<int64_t> stride_;
  StridedCPUTensorImpl() : TensorImpl(TypeIds::StridedCPUTensor) {};
  ArrayRef<int64_t> stride() const override {
    return stride_;
  }
  // Missing retain/release
};

class OpenCLTensorImpl : public TensorImpl {
  void* opencl_handle;
  OpenCLTensorImpl() : TensorImpl(TypeIds::OpenCLTensor) {};
};

using opengl_handle = uint16_t;

class OpenGLTensorImpl : public TensorImpl {
  opengl_handle handle_;
  OpenGLTensorImpl() : TensorImpl(TypeIds::OpenCLTensor) {};
public:
  opengl_handle handle() const {
    return handle_;
  }
};

// const OpenCLTensorImpl&

// NB: From ATen I dropped the following methods:
//  - toString()

//===--- Tensor.h -----------------------------------------------*- C++ -*-===//

// Design notes:
//  - Manual retain/release instead of shared_ptr. Reasons:
//      - PRIMARY: It's possible to work with the underlying retained object using
//        a C API, which is basically impossible to do with shared_ptr because
//        it doesn't expose a manual retain()/release() API
//      - SECONDARY: A true intrusive reference count has some nice properties
//        which you don't get from use of std::make_shared (to put the refcount
//        metadata next to the regular dynamic allocation) and
//        std::enabled_shared_from_this (which generally needs to store a weak pointer
//        to the control block).
// - UndefinedTensorImpl instead of null pointer. Reasons:
//      - We originally had a null pointer in ATen, but this meant that when we
//        incorrectly attempted to use such a null pointer, we would segfault and
//        crash, which is very unfriendly for our Python users.  Using an UndefinedTensorImpl
//        as our default constructor is much better for us.
// - Fixed the mismatch between PyTorch and C++ methods
//      - sizes() is now size()
//
// Tensor x = ...;
// Tensor y = x;  // NO COPY



// SUMMING UP
// 1. There will NOT be a retain/release on the Tensor class.  There might be
// some unsafe mechanism to retain/release (because C API bindings need it), but it won't
// be a method with an easy name to spell.
// 2. virtual issue, DELAYED (it's OK for now)
// 3. Undefined tensor... can we make illegal states unrepresentable?
// 4. Because of ArrayRef, we will need to define code style guide (ArrayRef disagrees)

class Tensor final {
  TensorImpl * pImpl;

  // This is a relatively unsafe constructor which you should avoid using if you
  // don't need it.  The retain parameter specifies whether or not this constructor
  // takes ownership of the passed Impl or not (when retain = true, the caller retains
  // their reference.)
  Tensor(TensorImpl* self)
      : pImpl(self) {
    if (pImpl == nullptr) {
      throw std::runtime_error("Tensor with nullptr not supported");
    }
  }
  TensorImpl * get() const {
    return pImpl;
  }
  TensorImpl * detach() {
    TensorImpl * ret = pImpl;
    pImpl = UndefinedTensorImpl::singleton();
    return ret;
  }

  // Refcounting kit
  void retain() {
    ++pImpl->refcount_;
  }
  void release() {
    if(--pImpl->refcount_ == 0) {
      delete pImpl;
    }
  }

public:
  // Normal constructors
  Tensor(): Tensor(UndefinedTensorImpl::singleton()) {}
  Tensor(const Tensor & rhs)
      : pImpl(rhs.pImpl) {
    if (pImpl != UndefinedTensorImpl::singleton())
      retain();
  }
  Tensor(Tensor && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = UndefinedTensorImpl::singleton();
  }

  // Destructor
  ~Tensor() {
    if (pImpl != UndefinedTensorImpl::singleton())
      release();
  }

  // Copy assignment
  Tensor & operator=(Tensor && rhs) & noexcept {
    rhs.swap(*this);
    return *this;
  }
  Tensor & operator=(Tensor const & rhs) & {
    //TensorBase ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally TensorBase dtor releases rhs.pImpl, which was originally this->pImpl
    Tensor(rhs).swap(*this);
    return *this;
  }

  void reset() {
    Tensor().swap(*this);
  }
  void swap(Tensor & rhs) noexcept {
    TensorImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  inline bool defined() const {
    return pImpl != UndefinedTensorImpl::singleton();
  }

  // These methods are SO important, they are currently implemented via virtual dispatch
  // via our implementation classes.  Most non-core methods should be implemented by
  // the generic dispatch mechanism.

  int64_t dim() const {
    return pImpl->dim();
  }
  inline int64_t ndimension() const {
    return dim();
  }
  ArrayRef<int64_t> size() const {
    return pImpl->size();
  }
  ArrayRef<int64_t> stride() const {
    return pImpl->stride();
  }
  /*
  TypeId type_id() const {
    return pImpl->type_id();
  }
   */
  template<typename T>
  T * data() const {
    return static_cast<T*>(pImpl->data_ptr());
  }


  // TODO: work out the type() situation

  // TODO: work out the storage() situation

  // The "well known" Tensor functions will call into the dispatch mechanism (yet to be
  // implemented)
};

/*
opengl_handle handle(Tensor x) const {
  if (x.type_id() != TypeIds::OpenGL) {
    throw "";
  }
  // code that knows about Impls
  static_cast<>
}
 */

/*
void f(Tensor x, Tensor y) {
  //Tensor z = x + y;
  // 1. Is this checked?
  // 2. When are people going to use it
  //auto& zimpl = z.unsafe_get_impl<StridedCPUTensorImpl>();
  //...
}


// OK fine.
void f(Tensor x, Tensor y) {
  Tensor z = x + y;
  // 1. Is this checked?  YES
  // 2. When are people going to use it
  auto& zimpl = z.unsafe_get_impl<OpenGLTensorImpl>();
  auto h = zimpl.handle();
}

// BAD IDEA
void f(const OpenGLTensorImpl& x, const OpenGLTensorImpl& y) {
  Tensor z = Tensor(x) + Tensor(y);
  // 1. Is this checked?
  // 2. When are people going to use it
  auto& zimpl = z.unsafe_get_impl<OpenGLTensorImpl>();
  auto h = zimpl.handle();
}


void f(Tensor x, Tensor y) {
  Tensor z = x + y;
  auto h = opengl_handle(z);
}*/
