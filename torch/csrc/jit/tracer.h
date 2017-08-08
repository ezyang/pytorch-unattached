#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/init_pass.h"

#include <memory>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>


namespace torch { namespace jit { namespace tracer {

using torch::autograd::Variable;
using variable_list = std::vector<std::shared_ptr<Variable>>;

// TracingState tracks the necessary state when we are tracing the execution of
// autograd code; most importantly, it holds a reference to the actual IR
// graph which we are recording the trace to.
//
// TODO: arguably it shouldn't be necessary to shared_ptr at
// TracingState
//
// TODO: I feel... some synchronization may be necessary
//
struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState()
    : graph(new Graph()) {}

  std::unique_ptr<Graph> graph;
  // NB: This map does not key on Variable*, because variable saving and
  // restoring can cause the pointer for a Variable to change. The
  // VariableUnique is a dummy object which stays stable across this.
  std::unordered_map<Variable::VariableUnique*, Node*> unique_map;
};

// This is the tracing state associated with the execution of the Python
// interpreter; e.g., when a user requests tracing from Python.
//
// Can't be a unique_ptr because multiple threads may be tracing into
// the same state (in case of autograd with multiple GPU devices.)  NB:
// you need locks here!
//
// Autograd engine responsible for setting this.  Could pass around
// explicitly but need to change calling convention.
extern thread_local std::shared_ptr<TracingState> ThreadTracingState;

// For now, we assume that the graph knows what's going on re stage
// (advanceStage: once you finish a stage, it can't ever be modified).
// But in principle you could do a graph, run it backwards, and then
// decide to add more to the first tage.

namespace detail {

// NB: the Eval node needs to know what the tracing state!

template<typename Subclass>
struct TracerHook : public autograd::FunctionPreHook {
protected:
  // Returns a vector of hooks that were registered. Subclasses can then perform additional initialization.
  static std::shared_ptr<Subclass> registerHook(variable_list& inputs);

public:
  virtual void run(variable_list& inputs) = 0;

  // Handle both kinds of hooks. In case of post hooks we only care about outputs.
  virtual variable_list operator()(const variable_list& _vars) {
    variable_list vars(_vars);
    for (auto& var : _vars)
      JIT_ASSERT(var);
    using this_type = typename std::remove_reference<decltype(*this)>::type;
    std::call_once(flag, std::bind(&this_type::run, this, vars));
    JIT_ASSERT(vars.size() == _vars.size());
    for (auto& var : vars) {
      JIT_ASSERT(var);
    }
    return vars;
  }

private:
  std::once_flag flag;
};

////////////////////////////////////////////////////////////////////////////////
// Trace hooks
////////////////////////////////////////////////////////////////////////////////

struct TraceEnterHook : public TracerHook<TraceEnterHook> {
private:
  friend struct TracerHook<TraceEnterHook>;

  virtual void run(variable_list& inputs) override;

public:
  static void registerHook(variable_list& outputs);
};

struct TraceExitHook : public TracerHook<TraceExitHook> {
private:
  friend struct TracerHook<TraceExitHook>;

  virtual void run(variable_list& outputs) override;

public:
  static void registerHook(variable_list& inputs);
};

} // namespace detail

/*
// Should a function which takes 'vars' as inputs be traced?
// It sufficies for ONE variable to be tracing: any "untraced" variables
// are treated as constants.
inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars) {
    if (!var) continue;
    auto state = var->tracing_state.state.lock();
    if (state && state->active)
        return true;
  }
  return false;
}

// Retrieve the tracing state which a function applied with 'vars' should
// be recorded to.  Precondition: isTracing(vars) == true.  At the moment,
// we don't support mixing up variables from different traces; this code
// will need to be revisited if that ever becomes supported.
inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  std::shared_ptr<TracingState> state;
  for (auto& var : vars) {
    if (!var) continue;
    auto var_state = var->tracing_state.state.lock();
    if (var_state) {
      if (!state) {
        state = var_state;
      }
      JIT_ASSERT(state == var_state);
    }
  }
  JIT_ASSERT(state);
  return state;
}
*/

// Having finished adding a new 'node' to the graph IR owned by TracingState 'state',
// 'setValueTrace' associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
inline void setValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, Node *node) {
  state->unique_map[var->unique.get()] = node;
}

// Given a variable 'var', return the 'node' which represents the instruction
// which computes the value of this variable in the IR.  When 'mustExist' is
// false, we interpret untraced variables as constants that are just embedded
// in the graph.  This is useful to handle code which does things like this
// (from torch.autograd.variable):
//
//    def mm(self, matrix):
//      output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
//      return Addmm.apply(output, self, matrix, 0, 1, True)
//
// Here, mm fakes up a dummy variable with uninitialized data to do an inplace
// update on, but subsequently ignores it because the alpha scaling factor is zero.
// This is one of the cases where a Variable can be created inside of a trace, and
// if we treat it as a constant, everything will work out.
inline Node* getValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, bool mustExist = false) {
  JIT_ASSERTM(var, "Not supported. NULL Variables will need to be removed from autograd");
  auto node_it = state->unique_map.find(var->unique.get());
  if (node_it == state->unique_map.end()) {
    if (mustExist) {
      std::cerr << *state->graph;
      throw std::runtime_error("untraced variable");
    }
    Node *constant = state->graph->appendNewNode<Constant>(var->data);
    setValueTrace(state, var, constant);
    return constant;
  } else {
    return node_it->second;
  }
}

// This is the USER level entry to start tracing
//
// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
// XXX: this changes variables in inputs!
inline void forward_enter(variable_list& inputs) {
  JIT_ASSERT(ThreadTracingState == nullptr);
  ThreadTracingState = std::make_shared<TracingState>();
  for (auto& input : inputs) {
    Node * node = ThreadTracingState->graph->addInput();
    setValueTrace(ThreadTracingState, input, node);
    node->inferTypeFrom(input->data);
  }
  detail::TraceExitHook::registerHook(inputs);
}

// gonna need this but not sure what the context is

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline std::shared_ptr<TracingState> forward_exit(variable_list& outputs) {
  JIT_ASSERT(ThreadTracingState != nullptr);
  // TODO: Shouldn't similar logic to this be invoked when we exit
  // backwards?  But AFAICT this is Python only logic...
  auto state = std::move(ThreadTracingState);
  for (auto& output : outputs) {
    state->graph->registerOutput(getValueTrace(state, output, true));
  }
  detail::TraceEnterHook::registerHook(outputs); // NB: modifies!
  // TODO: I hate this, get rid of it
  state->graph->advanceStage();
  return state;
}

////////////////////////////////////////////////////////////////////////////////
// Eval hooks
////////////////////////////////////////////////////////////////////////////////

struct EvalCommonState {
  // Filled in by EvalEnterHook when ran
  Node* eval_node;
  std::shared_ptr<EvalCommonState> next_common_state;
};

struct EvalEnterHook : public detail::TracerHook<EvalEnterHook> {
private:
  friend detail::TracerHook<EvalEnterHook>;

  std::shared_ptr<EvalCommonState> common_state;

  virtual void run(variable_list& vars) override;

public:
  static void registerHook(variable_list& outputs, std::shared_ptr<EvalCommonState> common_state);
};

struct EvalExitHook : public detail::TracerHook<EvalExitHook> {
private:
  friend detail::TracerHook<EvalExitHook>;

  std::shared_ptr<EvalCommonState> common_state;

  virtual void run(variable_list& vars) override;

public:
  static std::shared_ptr<EvalCommonState> registerHook(variable_list& inputs);
};

}}} // namespace torch::jit::tracer
