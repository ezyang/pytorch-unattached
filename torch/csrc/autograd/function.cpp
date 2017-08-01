#include "function.h"

#include <string>
#include <THPP/THPP.h>

#include "variable.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace autograd {

auto Function::flags(const variable_list& inputs) -> FunctionFlags {
  int num_inputs = inputs.size();
  FunctionFlags f;
  f.is_executable = false;
  f.is_volatile = false;
  f.next_functions.resize(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    auto& var = inputs[i];
    if (var) {
      f.is_executable |= var->requires_grad;
      f.is_volatile |= var->is_volatile;
      if (var->grad_fn) {
        f.next_functions[i] = std::make_pair<>(var->grad_fn, var->output_nr);
      } else {
        f.next_functions[i] = std::make_pair<>(var->get_grad_accumulator(), 0);
      }
    }
  }
  f.is_executable &= !f.is_volatile;
  return f;
}

auto Function::name() -> std::string {
  return std::string(typeid(*this).name());
}

namespace {

void removeNodesBetween(const std::shared_ptr<jit::Graph>& graph, const variable_list& inputs, const variable_list& outputs) {
  std::unordered_set<jit::Node*> input_nodes;
  std::vector<jit::Node*> queue;
  for (auto& input : inputs) {
    if (!input->tracing_state.trace) continue;
    input_nodes.emplace(jit::tracer::getValueTrace(graph, input, true));
  }
  for (auto& output : outputs) {
    if (!output->tracing_state.trace) continue;
    queue.emplace_back(jit::tracer::getValueTrace(graph, output, true));
  }
  while (!queue.empty()) {
    auto node = queue.back(); queue.pop_back();
    if (input_nodes.count(node) > 0) continue;
    JIT_ASSERT(node->uses().size() == 0);
    auto inputs = node->inputs();
    node->destroy();
    for (auto input : inputs) {
      if (input->uses().size() == 0)
        queue.emplace_back(input);
    }
  }
}

}

void Function::createTrace(const variable_list& inputs, const variable_list& outputs) {
  using namespace torch::jit;
  // TODO: actually descend for simple ops
  bool descend_trace = false;
  if (!descend_trace) {
    // Tear down the trace created by functions call within apply.
    auto graph = tracer::getGraph(inputs);
    removeNodesBetween(graph, inputs, outputs);
    auto* this_node = graph->appendNewNode<AutogradOp>(getSharedPtr());
    for (auto& input: inputs) {
      this_node->addInput(tracer::getValueTrace(graph, input));
    }
    int num_outputs = outputs.size();
    for (int i = 0; i < num_outputs; ++i) {
      auto& output = outputs[i];
      Node* sel = graph->appendNewNode<Select>(this_node, i);
      sel->inferTypeFrom(output->data);
      tracer::setValueTrace(output, sel);
    }
  }
}

}} // namespace torch::autograd
