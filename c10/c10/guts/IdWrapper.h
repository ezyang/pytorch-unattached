#pragma once

#include <functional>

namespace c10 { namespace guts {

/**
 * This template simplifies generation of simple classes that wrap an id
 * in a typesafe way. Namely, you can use it to create a very lightweight
 * type that only offers equality comparators and hashing. Example:
 *
 *   struct MyIdType final : IdWrapper<MyIdType, uint32_t> {
 *     constexpr explicit MyIdType(uint32_t id): IdWrapper(id) {}
 *   };
 *
 * Then in the global top level namespace:
 *
 *   C10_DEFINE_IDWRAPPER(MyIdType);
 *
 * That's it - equality operators and hash functions are automatically defined
 * for you, given the underlying type supports it.
 */
template <class ConcreteType, class UnderlyingType>
class IdWrapper {
public:
  using underlying_type = UnderlyingType;
  using concrete_type = ConcreteType;

protected:
  constexpr explicit IdWrapper(underlying_type id) : id_(id) {}
  constexpr underlying_type underlyingId() const { return id_; }

private:
  friend struct std::hash<ConcreteType>;
  template <class C1, class U1>
  friend constexpr bool operator==(const IdWrapper<C1, U1>& lhs, const IdWrapper<C1, U1>& rhs);
  template <class C1, class U1>
  friend constexpr bool operator!=(const IdWrapper<C1, U1>& lhs, const IdWrapper<C1, U1>& rhs);

  underlying_type id_;
};

template <class ConcreteType, class UnderlyingType>
inline constexpr bool operator==(const IdWrapper<ConcreteType, UnderlyingType>& lhs,
                                 const IdWrapper<ConcreteType, UnderlyingType>& rhs) {
  return lhs.id_ == rhs.id_;
}

template <class ConcreteType, class UnderlyingType>
inline constexpr bool operator!=(const IdWrapper<ConcreteType, UnderlyingType>& lhs,
                                 const IdWrapper<ConcreteType, UnderlyingType>& rhs) {
  return !operator==(lhs, rhs);
}

}}

#define C10_DEFINE_HASH_FOR_IDWRAPPER(ClassName)             \
  namespace std {                                            \
  template <>                                                \
  struct hash<ClassName> {                                   \
    size_t operator()(ClassName x) const {                   \
      return std::hash<ClassName::underlying_type>()(x.id_); \
    }                                                        \
  };                                                         \
  }
