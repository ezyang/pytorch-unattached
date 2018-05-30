#pragma once

#include <c10/dispatch/OpSchema.h>
#include "ParameterStack.h"
#include <c10/dispatch/Dispatcher.h>
#include <c10/guts/C++17.h>

namespace c10 {

namespace details {
// A small gadget to invoke the dispatcher with a tuple of arguments (native API is
// variadic arguments).
template<class OpSchemaDef, class ReturnType, class ArgsTuple> struct call_dispatcher_with_args_tuple;
template<class OpSchemaDef, class ReturnType, class... Args> struct call_dispatcher_with_args_tuple<OpSchemaDef, ReturnType, std::tuple<Args...>> final {
  static ReturnType call(std::tuple<Args...>&& args) {
    return call_(std::move(args), std::index_sequence_for<Args...>());
  }
private:
  template<size_t... I>
  static ReturnType call_(std::tuple<Args...>&& args, std::index_sequence<I...>) {
    return Dispatcher<OpSchemaDef>::call(std::move(std::get<I>(args))...);
  }
};
}



/**
 * The stack based operator interface takes it's arguments via a ParameterStack,
 * and pushes its outputs onto the ParameterStack.  This makes it much easier to work with from
 * client code which needs to operate polymorphically over operators which have different signatures.
 *
 * Example: Suppose that you have the operator 'minus' which subtracts two tensors and returns the result.
 * Then supposing callStack is [c, b, a] upon entry, then on exit it will be [c, a - b] (c, in general,
 * is some prefix of parameters on the stack which are untouched by the stack.)
 */
class StackBasedOperator {
public:
  /**
   * Invoke the operator, popping arguments from the top of the stack (first argument is very top)
   * and pushing outputs onto the stack (only ever one argument; AT THE MOMENT).
   * @param callStack
   */
  virtual void operator()(ParameterStack* callStack) = 0;

  virtual ~StackBasedOperator() = default;
};

// TODO: This probably lives at the wrong level of abstraction now, given a change of plans
// due to https://fb.quip.com/o5pXA1LHgAq0
/**
 * The ConcreteStackBasedOperator implements the wrapper for a specific operator schema, popping
 * arguments off the stack into the correct unboxed form to subsequently call the dispatcher.
 *
 * @tparam OpSchemaDef The OpSchemaDef to create a ConcreteStackBasedOperator from.
 */
template<class OpSchemaDef>
class ConcreteStackBasedOperator final : public StackBasedOperator {
private:
  using Schema = OpSchema<OpSchemaDef>;
  using ReturnType = typename Schema::signature::return_type;

  using ParameterBaseTypes = guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename Schema::signature::parameter_types>>;
  using ArgumentsTuple = guts::typelist::to_tuple_t<ParameterBaseTypes>;

public:
  void operator()(ParameterStack* callStack) override {
    using guts::typelist::map_types_to_values;
    using guts::typelist::reverse_t;

    // TODO Instead push a full args tuple to the stack?
    // TODO Pop in reverse (using Metaprogramming::reverse_t, but then also reverse results again? Performance?)
    ArgumentsTuple arguments =
      map_types_to_values<ParameterBaseTypes>([callStack] (auto t) {
        using ParameterType = typename decltype(t)::type;
        return callStack->pop<ParameterType>();
      });

    // TODO Check if this correctly moves the arguments from the tuple into the op
    ReturnType result = details::call_dispatcher_with_args_tuple<OpSchemaDef, ReturnType, ArgumentsTuple>::call(std::move(arguments));
    // TODO ezyang to smessmer: This looks questionable, it seems more likely that if we get a tuple back from
    // the result, we will want to unpack it into the stack, because if we have an implementation of Any that
    // has the small result optimization, this will help us having to avoid doing dynamic allocations to have space
    // to store the result.
    callStack->push(std::move(result));
  }
};

}
