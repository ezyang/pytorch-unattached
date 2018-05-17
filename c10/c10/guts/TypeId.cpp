#include "TypeId.h"
#include "scope_guard.h"

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

using std::string;

namespace c10 {

std::unordered_map<TypeId, string>& gTypeNames() {
  static std::unordered_map<TypeId, string> g_type_names;
  return g_type_names;
}

std::unordered_set<string>& gRegisteredTypeNames() {
  static std::unordered_set<string> g_registered_type_names;
  return g_registered_type_names;
}

std::mutex& gTypeRegistrationMutex() {
  static std::mutex g_type_registration_mutex;
  return g_type_registration_mutex;
}

#if defined(_MSC_VER)
// Windows does not have cxxabi.h, so we will simply return the original.
string Demangle(const char* name) {
  return string(name);
}
#else
string Demangle(const char* name) {
  int status = 0;
  auto demangled = ::abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (demangled) {
    auto guard = MakeGuard([demangled]() { free(demangled); });
    return string(demangled);
  }
  return name;
}
#endif

string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return Demangle(typeid(e).name()) + ": " + e.what();
#else
  return string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

namespace {
// This single registerer exists solely for us to be able to name a TypeMeta
// for unintializied blob. You should not use this struct yourself - it is
// intended to be only instantiated once here.
struct UninitializedTypeNameRegisterer {
  UninitializedTypeNameRegisterer() {
    gTypeNames()[TypeId::uninitialized()] = "nullptr (uninitialized)";
  }
};
static UninitializedTypeNameRegisterer g_uninitialized_type_name_registerer;

} // namespace


// TODO Find better place for these, probably copy types.h/cc from caffe2 to c10
C10_KNOWN_TYPE(float);
C10_KNOWN_TYPE(int);
C10_KNOWN_TYPE(std::string);
C10_KNOWN_TYPE(bool);
C10_KNOWN_TYPE(uint8_t);
C10_KNOWN_TYPE(int8_t);
C10_KNOWN_TYPE(uint16_t);
C10_KNOWN_TYPE(int16_t);
C10_KNOWN_TYPE(int64_t);
C10_KNOWN_TYPE(double);
C10_KNOWN_TYPE(char);
C10_KNOWN_TYPE(std::unique_ptr<std::mutex>);
C10_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>);
C10_KNOWN_TYPE(std::vector<int64_t>);
C10_KNOWN_TYPE(std::vector<unsigned long>);
C10_KNOWN_TYPE(bool*);
C10_KNOWN_TYPE(char*);
C10_KNOWN_TYPE(int*);

}
