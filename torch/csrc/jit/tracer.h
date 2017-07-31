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

inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars)
    if (!var->tracing_state.graph.expired())
      return true;
  return false;
}

inline std::shared_ptr<Graph> getGraph(const variable_list& vars) {
  std::shared_ptr<Graph> graph;
  for (auto& var : vars) {
    auto var_graph = var->tracing_state.graph.lock();
    if (var_graph) {
      if (!graph) {
        graph = var_graph;
      } else if (graph != var_graph) {
        throw std::runtime_error("Mixing up traces");
      }
    }
  }
  JIT_ASSERT(graph);
  return graph;
}

inline void setValueTrace(const std::shared_ptr<Variable>& var, Node *node) {
  var->tracing_state.graph = node->owningGraph()->shared_from_this();
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
  // TODO: register exit hooks!
  return graph;
}

inline void exit(const std::shared_ptr<Graph>& graph, const variable_list& outputs) {
  for (auto& output : outputs) {
    JIT_ASSERT(output->tracing_state.graph.lock() == graph);
    graph->registerOutput(getValueTrace(graph, output, true));
    output->tracing_state.graph.reset();
    output->tracing_state.trace = nullptr;
  }
  // TODO: register enter hooks!
}

}}} // namespace torch::jit::tracer
