#include "Utils.h"
#include <c10/ScopeGuard.h>

namespace c10 {

#if defined(_MSC_VER)
// Windows does not have cxxabi.h, so we will simply return the original.
std::string demangle(const char* name) {
  return std::string(name);
}
#else
std::string demangle(const char* name) {
  int status = 0;
  auto demangled = ::abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (demangled) {
    auto guard = MakeGuard([demangled]() { free(demangled); });
    return std::string(demangled);
  }
  return name;
}
#endif

}