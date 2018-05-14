#pragma once

#include <c10/dispatch/OpSchema.h>
#include "CallStack.h"
#include <c10/dispatch/Dispatcher.h>
#include <c10/guts/C++17.h>

namespace c10 {

namespace details {
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




class StackBasedOperator {
public:
  virtual void operator()(CallStack* callStack) = 0;

  virtual ~StackBasedOperator() = default;
};

template<class OpSchemaDef>
class ConcreteStackBasedOperator final : public StackBasedOperator {
private:
  using Schema = OpSchema<OpSchemaDef>;
  using ReturnType = typename Schema::signature::return_type;

  using ParameterBaseTypes = guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename Schema::signature::parameter_types>>;

  template<class T> using is_not_reference = guts::conjunction<guts::negation<std::is_reference<T>>, guts::negation<std::is_const<T>>>;
  static_assert(guts::typelist::true_for_each_type<is_not_reference, ParameterBaseTypes>::value, "bla");
  using ArgumentsTuple = typename ParameterBaseTypes::tuple_type;

public:
  void operator()(CallStack* callStack) override {
    using guts::typelist::map_types_to_values;
    using guts::typelist::reverse_t;

    // TODO Instead push a full args tuple to the stack?
    // TODO Pop in reverse (using Metaprogramming::reverse_t, but then also reverse results again? Performance?)
    ArgumentsTuple arguments =
      map_types_to_values<ParameterBaseTypes>([callStack] (auto* t) {
        using ParameterType = std::remove_cv_t<std::remove_reference_t<decltype(*t)>>;
        return callStack->pop<ParameterType>();
      });

    // TODO Check if this correctly moves the arguments from the tuple into the op
    ReturnType result = details::call_dispatcher_with_args_tuple<OpSchemaDef, ReturnType, ArgumentsTuple>::call(std::move(arguments));
    callStack->push(std::move(result));
  }
};

}
