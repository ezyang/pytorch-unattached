#pragma once

//#include "../../../torch/csrc/utils/variadic.h"

#include <c10/Error.h>

#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>
#include "C++17.h"

namespace c10 {

// Reference:
// https://github.com/llvm-mirror/libcxx/blob/master/include/memory#L3091

template <typename T>
struct unique_type_for {
  using value = std::unique_ptr<T>;
};

template <typename T>
struct unique_type_for<T[]> {
  using unbounded_array = std::unique_ptr<T[]>;
};

template <typename T, size_t N>
struct unique_type_for<T[N]> {
  using bounded_array = void;
};

template <typename T, typename... Args>
typename unique_type_for<T>::value make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
typename unique_type_for<T>::unbounded_array make_unique(size_t size) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[size]());
}

template <typename T, size_t N, typename... Args>
typename unique_type_for<T>::bounded_array make_unique(Args&&...) = delete;
} // namespace c10


namespace c10 {
class Any;
namespace detail {
template <typename T>
Any make_any(T&& value);
} // namespace detail
} // namespace c10

namespace c10 {
/// A simplified implementation of `std::any` or `boost::any` which stores a
/// type erased object, whose concrete value can be retrieved at runtime by
/// checking if the `typeid()` of a requested type matches the `typeid()` of the
/// object stored. It is simplified in that it does not handle copying, as we do
/// not require it for our use cases. Moves are sufficient.
class Any {
 public:
  /// Default construction is disallowed. Thus our invariant: an `Any` contains
  /// an object at all times, from its construction, to its destruction.
  Any() = delete;

  /// Move construction and assignment is allowed, and follows the default
  /// behavior of move for `std::unique_ptr`.
  Any(Any&&) = default;
  Any& operator=(Any&&) = default;

  /// Copy is disallowed, because we don't need it.
  Any(const Any& other) = delete;
  Any& operator=(const Any& other) = delete;

  /// Returns the value contained in the `Any` if the type passed as template
  /// parameter matches the type of the object stored, and returns a null
  /// pointer otherwise.
  template <typename T>
  T* try_get() {
    static_assert(
        !std::is_reference<T>::value,
        "Any stores decayed types, you cannot cast it to a reference type");
    static_assert(
        !std::is_array<T>::value,
        "Any stores decayed types, you must cast it to T* instead of T[]");
    if (typeid(T).hash_code() == type_info().hash_code()) {
      return &static_cast<Holder<T>&>(*content_).value;
    }
    return nullptr;
  }

  template <typename T>
  T get() & {
    if (auto* value = try_get<T>()) {
      return *value;
    }
    C10_ERROR(
        "Attempted to cast Any to",
        c10::detail::demangle(typeid(T).name()),
        ", but its contained type is ",
        c10::detail::demangle(type_info().name()));
  }

  template <typename T>
  T get() && {
    if (auto* value = try_get<T>()) {
      return std::move(*value);
    }
    C10_ERROR(
        "Attempted to cast Any to",
        c10::detail::demangle(typeid(T).name()),
        ", but its contained type is ",
        c10::detail::demangle(type_info().name()));
  }

  /// Returns the `type_info` object of the contained value.
  const std::type_info& type_info() const noexcept {
    return content_->type_info;
  }

 private:
  /// Constructs the `Any` from any type.
  template <typename T>
  explicit Any(T&& value)
      : content_(
            guts::make_unique<Holder<guts::decay_t<T>>>(std::forward<T>(value))) {}

  /// `Any` is a public type, but its construction is only allowed via this
  /// "private" function in the `detail` namespace.
  template <typename T>
  friend Any detail::make_any(T&& value);

  /// The static type of the object we store in the `Any`, which erases the
  /// actual object's type, allowing us only to check the `type_info` of the
  /// type stored in the dynamic type.
  struct Placeholder {
    explicit Placeholder(const std::type_info& type_info_) noexcept
        : type_info(type_info_) {}
    virtual ~Placeholder() = default;
    const std::type_info& type_info;
  };

  /// The dynamic type of the object we store in the `Any`, which hides the
  /// actual object we have erased in this `Any`.
  template <typename T>
  struct Holder : public Placeholder {
    /// A template because T&& would not be universal reference here.
    template <typename U>
    explicit Holder(U&& value_) noexcept
        : Placeholder(typeid(T)), value(std::forward<U>(value_)) {}

    T value;
  };

  /// The type erased object.
  std::unique_ptr<Placeholder> content_;
};

namespace detail {
/// Constructs a new `Any` object from any value.
template <typename T>
Any make_any(T&& value) {
  return Any(std::forward<T>(value));
}

/// A group of methods to construct an `std::vector<Any>` from a variadic list
/// of arguments.
inline void push_any(std::vector<Any>& /*vector*/) {}

template <typename Head, typename... Tail>
void push_any(std::vector<Any>& vector, Head&& head, Tail&&... tail) {
  vector.push_back(make_any(std::forward<Head>(head)));
  push_any(vector, std::forward<Tail>(tail)...);
}

template <typename... Args>
std::vector<Any> make_any_vector(Args&&... args) {
  std::vector<Any> vector;
  vector.reserve(sizeof...(Args));
  push_any(vector, std::forward<Args>(args)...);
  return vector;
}
} // namespace detail
} // namespace c10
