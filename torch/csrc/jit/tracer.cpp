#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"

namespace torch { namespace jit { namespace tracer {


namespace detail {

void TraceEnterHook::run(const variable_list& inputs) {
  auto& input = inputs[creator_output_nr];
  int num_hooks_called;
  {
    std::lock_guard<std::mutex> guard(state->mutex);
    state->inputs[backward_input_nr] = input;
    num_hooks_called = ++state->num_hooks_called;
    // Enter tracing
    if (num_hooks_called == 1) {
      state->graph->advanceStage();
      state->graph->tracing = true;
      for (int i = 0; i < state->num_new_inputs; ++i)
        state->graph->addInput();
    }
  }

  int graph_input_nr = state->num_forward_inputs + backward_input_nr;
  // TODO: support this. The problem is that we've likely already reserved an input for this
  // Variable, so we need to destroy it, but Node::destroy() works only with regular nodes.
  JIT_ASSERTM(input->tracing_state.graph.expired(), "having previous stage "
      "outputs as inputs to the next stage is not suported yet");
  setValueTrace(input, state->graph->inputs()[graph_input_nr]);
  if (num_hooks_called == state->num_new_inputs) {
    TraceExitHook::registerHooks(state->graph, state->inputs);
  }
}

void TraceEnterHook::registerHooks(const std::shared_ptr<Graph>& graph, const variable_list& outputs) {
  JIT_ASSERT(outputs.size() > 0);

  // Either no (e.g. after last backward) or all outputs should have a grad fn.
  bool has_grad_fn = static_cast<bool>(outputs[0]->grad_fn);
  for (auto& output : outputs) {
    JIT_ASSERT(static_cast<bool>(output->grad_fn) == has_grad_fn);
  }
  if (!has_grad_fn) return;

  int num_outputs = outputs.size();
  int num_forward_inputs = graph->inputs().size();
  auto state = std::make_shared<SharedEnterState>(graph, num_outputs, num_forward_inputs);
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    auto *hook = new TraceEnterHook(state, output->output_nr, i);
    output->grad_fn->pre_hooks.emplace_back(static_cast<FunctionPreHook*>(hook));
  }
}

void TraceExitHook::run(const variable_list& outputs) {
  auto& output = outputs[creator_output_nr];
  int num_hooks_called;
  {
    std::lock_guard<std::mutex> guard(state->mutex);
    num_hooks_called = ++state->num_hooks_called;
    state->outputs[backward_output_nr] = output;
  }
  if (num_hooks_called == state->num_new_outputs) {
    exit(state->graph, state->outputs);
  }
}

void TraceExitHook::registerHooks(const std::shared_ptr<Graph>& graph, const variable_list& inputs) {
  int num_inputs = inputs.size();
  int num_requires_grad = 0;
  for (auto& input: inputs) {
    if (input->requires_grad)
      num_requires_grad++;
  }
  auto state = std::make_shared<SharedExitState>(graph, num_requires_grad);
  for (int i = 0; i < num_inputs; ++i) {
    auto& input = inputs[i];
    if (!input->requires_grad) continue;
    auto* hook = new TraceExitHook(state, input->output_nr, i);
    auto& hook_container = input->grad_fn ? input->grad_fn->pre_hooks : input->hooks;
    hook_container.emplace_back(static_cast<FunctionPreHook*>(hook));
  }
}

} // namespace detail

}}} // namespace torch::jit::tracer
