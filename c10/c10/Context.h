#pragma once

#include <c10/dispatch/Dispatcher.h>

namespace c10 {

class Context {
  Dispatcher dispatcher_;

public:

  Dispatcher& getMutDispatcher() { return dispatcher_ ;}
  const Dispatcher& getDispatcher() const { return dispatcher_ ;}

};

Context &globalContext();

} // namespace c10
