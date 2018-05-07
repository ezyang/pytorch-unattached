#pragma once

#include "OpSchema.h"
#include "Dispatcher.h"
#include <c10/Optional.h>

/**
 * To register your own operator, do in one (!) cpp file:
 *   C10_DEFINE_OPERATOR(OpSchemaDef, func, {tensor_type1, tensor_type2, ...})
 * Both must be in the same namespace.
 */

namespace c10 {

// TODO Test different order for builder
// TODO Test no dispatch key defined

template<class OpSchemaDef>
class KernelRegistrar final {
private:
    using Schema = OpSchema<OpSchemaDef>;
public:
  KernelRegistrar(typename Schema::signature::func_type* kernel, typename Schema::dispatch::dispatch_key_type dispatch_key)
  : dispatch_key_(std::move(dispatch_key)), owns_registration_(true) {
    Dispatcher::registerOp<OpSchemaDef>(kernel, dispatch_key_);
  }

  KernelRegistrar(KernelRegistrar&& rhs)
  : dispatch_key_(std::move(rhs.dispatch_key_)), owns_registration_(true) {
    rhs.owns_registration_ = false;
  }

  // not needed for now
  KernelRegistrar& operator=(KernelRegistrar&& rhs) = delete;

  ~KernelRegistrar() {
    if (owns_registration_) {
      Dispatcher::deregisterOp<OpSchemaDef>(dispatch_key_);
    }
  }

private:
  const typename Schema::dispatch::dispatch_key_type dispatch_key_;
  bool owns_registration_;

  DISALLOW_COPY_AND_ASSIGN(KernelRegistrar);
};

template<class OpSchemaDef, bool hasKernel, bool hasDispatchKey>
class KernelRegistrationBuilder final {
private:
  using Schema = OpSchema<OpSchemaDef>;

  optional<typename Schema::signature::func_type*> kernel_;
  optional<typename Schema::dispatch::dispatch_key_type> dispatch_key_;

public:
  constexpr KernelRegistrationBuilder(): KernelRegistrationBuilder(nullopt, nullopt) {}

  constexpr KernelRegistrationBuilder(optional<typename Schema::signature::func_type*> kernel, optional<typename Schema::dispatch::dispatch_key_type> dispatch_key)
  : kernel_(std::move(kernel)), dispatch_key_(std::move(dispatch_key)) {}

  constexpr operator KernelRegistrar<OpSchemaDef>() && {
    static_assert(hasKernel, "Forgot to call .kernel() in kernel registration");
    static_assert(hasDispatchKey, "Forgot to call .dispatchKey() in kernel registration");
    return KernelRegistrar<OpSchemaDef>(std::move(*kernel_), std::move(*dispatch_key_));
  }

  constexpr auto kernel(typename Schema::signature::func_type* kernel) && {
    static_assert(!hasKernel, "Tried to define kernel twice in same op registration");
    return KernelRegistrationBuilder<OpSchemaDef, true, hasDispatchKey>(*kernel, std::move(dispatch_key_));
  }

  constexpr auto dispatchKey(typename Schema::dispatch::dispatch_key_type dispatch_key) && {
    static_assert(!hasDispatchKey, "Tried to define kernel twice in same op registration");
    return KernelRegistrationBuilder<OpSchemaDef, hasKernel, true>(std::move(kernel_), std::move(dispatch_key));
  }
};

}

// TODO Can the builder logic be moved to compile time?
#define CONCAT_IMPL( x, y ) x##y
#define MACRO_CONCAT( x, y ) CONCAT_IMPL( x, y )
#define C10_REGISTER_OP(OpSchemaDef)                                                           \
  static KernelRegistrar<OpSchemaDef> MACRO_CONCAT(__kernelRegistrationBuilder_, __COUNTER__) = KernelRegistrationBuilder<OpSchemaDef, false, false>()
