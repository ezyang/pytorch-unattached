#include "function.h"

#include <string>

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

// This function is analogous to make_trace which operates on PythonOp, but this
// function instead works for C++ implemented autograd Functions, which don't
// actually have any backing Python class. We still need to trace them!
void Function::createTrace(const variable_list& inputs, const variable_list& outputs) {
  using namespace torch::jit;
  auto state = tracer::getTracingState(inputs);
  auto& graph = state->graph;
  // See Note [getValueTrace can allocate nodes]
  std::vector<Node*> value_traces;
  value_traces.reserve(inputs.size());
  for (auto& input: inputs) {
    value_traces.emplace_back(tracer::getValueTrace(state, input));
  }
  auto* this_node = graph->appendNewNode<CppOp>(getSharedPtr());
  for (auto value: value_traces) {
    this_node->addInput(value);
  }
  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    Node* sel = graph->appendNewNode<Select>(this_node, i);
    sel->inferTypeFrom(output->data);
    tracer::setValueTrace(state, output, sel);
  }
}

}} // namespace torch::autograd
