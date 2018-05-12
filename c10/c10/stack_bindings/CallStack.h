#pragma once

#include <c10/guts/Any.h>
#include <vector>

namespace c10 {

class CallStack final {
public:
  template<class T>
  void push(T&& arg) {
    // TODO Why is make_any in detail?
    callStack_.push_back(c10::detail::make_any(std::forward<T>(arg)));
  }

  template<class T>
  T pop() {
    static_assert(!std::is_reference<T>::value, "Cannot pop by reference");
    // TODO Make sure this moving out of the stack works correctly and doesn't copy
    T result = std::move(callStack_.back()).get<T>();
    callStack_.pop_back();
    return result;
  }

private:
  std::vector<Any> callStack_;
};

}
