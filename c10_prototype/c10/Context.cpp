#include <c10/Context.h>

namespace c10 {

Context & globalContext() {
  static Context ctx;
  return ctx;
}

} // namespace c10
