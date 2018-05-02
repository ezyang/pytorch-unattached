#include "Dispatcher.h"

namespace c10 {

Dispatcher::Dispatcher()
: ops_() {}

Dispatcher& Dispatcher::singleton() {
  static Dispatcher singleton;
  return singleton;
}

}
