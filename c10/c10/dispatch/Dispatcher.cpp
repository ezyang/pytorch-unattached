#include "Dispatcher.h"

namespace c10 {

Dispatcher& dispatch() {
  static Dispatcher singleton;
  return singleton;
}

}
