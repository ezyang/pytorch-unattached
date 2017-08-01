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

namespace detail {

struct OneTimeHook : public autograd::FunctionPreHook {

  variable_list operator()(const variable_list& vars) {
    std::call_once(flag, std::bind(&OneTimeHook::run, this, vars));
    return vars;
  }

  virtual void run(const variable_list& vars) = 0;

private:
  std::once_flag flag;
};

struct TraceEnterHook : public OneTimeHook {
private:
  struct SharedEnterState {
    SharedEnterState(std::shared_ptr<Graph> graph, int num_new_inputs, int num_forward_inputs)
      : mutex()
      , graph(std::move(graph))
      , inputs(num_new_inputs)
      , num_new_inputs(num_new_inputs)
      , num_hooks_called(0)
      , num_forward_inputs(num_forward_inputs) {}

    std::mutex mutex;
    std::shared_ptr<Graph> graph;
    variable_list inputs;
    int num_new_inputs;
    int num_hooks_called;
    int num_forward_inputs;
  };

  TraceEnterHook(std::shared_ptr<SharedEnterState> state, int creator_output_nr,
                 int backward_input_nr)
    : state(std::move(state))
    , creator_output_nr(creator_output_nr)
    , backward_input_nr(backward_input_nr) {}

  std::shared_ptr<SharedEnterState> state;
  int creator_output_nr;
  int backward_input_nr;

public:
  virtual void run(const variable_list& inputs);

  static void registerHooks(const std::shared_ptr<Graph>& graph, const variable_list& outputs);
};

struct TraceExitHook : public OneTimeHook {
private:
  struct SharedExitState {
    SharedExitState(std::shared_ptr<Graph> graph, int num_new_outputs)
      : mutex()
      , graph(std::move(graph))
      , outputs(num_new_outputs)
      , num_new_outputs(num_new_outputs)
      , num_hooks_called(0) {}

    std::mutex mutex;
    std::shared_ptr<Graph> graph;
    variable_list outputs;
    int num_new_outputs;
    int num_hooks_called;
  };

  TraceExitHook(std::shared_ptr<SharedExitState> state, int creator_output_nr, int backward_output_nr)
    : state(std::move(state))
    , creator_output_nr(creator_output_nr)
    , backward_output_nr(backward_output_nr) {}

  std::shared_ptr<SharedExitState> state;
  int creator_output_nr;
  int backward_output_nr;

public:
  virtual void run(const variable_list& outputs);

  static void registerHooks(const std::shared_ptr<Graph>& graph, const variable_list& inputs);
};

} // namespace detail

inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars) {
    if (!var) continue;
    auto& state = var->tracing_state;
    // NOTE: the fact that it's non-NULL doesn't mean it's a valid ptr
    if (state.trace) {
      auto graph = state.graph.lock();
      if (graph && graph->tracing)
        return true;
    }
  }
  return false;
}

inline std::shared_ptr<Graph> getGraph(const variable_list& vars) {
  std::shared_ptr<Graph> graph;
  for (auto& var : vars) {
    if (!var) continue;
    auto var_graph = var->tracing_state.graph.lock();
    if (var_graph) {
      if (!graph) {
        graph = var_graph;
      }
      JIT_ASSERT(graph == var_graph);
    }
  }
  JIT_ASSERT(graph);
  return graph;
}

// TODO: what if an output is used in an in-place op? it might appear in the trace again,
// but it really points to a different place in the graph than its trace
inline void setValueTrace(const std::shared_ptr<Variable>& var, Node *node) {
  auto var_graph = var->tracing_state.graph.lock();
  if (!var_graph) {
    var->tracing_state.graph = var_graph = node->owningGraph()->shared_from_this();
  } else {
    JIT_ASSERT(var_graph.get() == node->owningGraph());
  }
  var->tracing_state.trace = node;
}

inline Node* getValueTrace(const std::shared_ptr<Graph>& graph, const std::shared_ptr<Variable>& var, bool mustExist = false) {
  auto var_graph = var->tracing_state.graph.lock();
  if (var_graph) {
    JIT_ASSERT(var->tracing_state.graph.lock() == graph);
    return var->tracing_state.trace;
  }

  if (mustExist) throw std::runtime_error("untraced variable");

  Node *constant = graph->appendNewNode<Constant>(var->data);
  setValueTrace(var, constant);
  return constant;
}

inline std::shared_ptr<Graph> enter(const variable_list& inputs) {
  auto graph = std::make_shared<Graph>();
  for (auto& input : inputs) {
    JIT_ASSERT(input->tracing_state.graph.expired());
    input->tracing_state.graph = graph;
    input->tracing_state.trace = graph->addInput();
  }
  detail::TraceExitHook::registerHooks(graph, inputs);
  graph->tracing = true;
  return graph;
}

inline void exit(const std::shared_ptr<Graph>& graph, const variable_list& outputs) {
  for (auto& output : outputs) {
    JIT_ASSERT(output->tracing_state.graph.lock() == graph);
    graph->registerOutput(getValueTrace(graph, output, true));
  }
  graph->tracing = false;
  detail::TraceEnterHook::registerHooks(graph, outputs);
}

}}} // namespace torch::jit::tracer
