#include "torch/csrc/jit/tracer.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/tensor.h"

namespace torch { namespace jit { namespace tracer {

thread_local std::shared_ptr<TracingState> ThreadTracingState;

namespace detail {

static std::shared_ptr<autograd::Function> insertIdentity(variable_list& vars) {
  int num_vars = vars.size();
  variable_list vars_with_grad;
  for (auto& var : vars) {
    if (var && var->requires_grad)
      vars_with_grad.emplace_back(var);
  }
  auto bw_hook_fn = std::make_shared<autograd::Identity>(autograd::Function::flags(vars_with_grad));
  bw_hook_fn->num_inputs = vars_with_grad.size();
  int output_nr = 0;
  for (int i = 0; i < num_vars; ++i) {
    if (!vars[i]) continue;
    auto& var = vars[i];
    if (!var->requires_grad) continue;
    auto var_clone = var->save(bw_hook_fn.get()).unpack(bw_hook_fn);
    var_clone->grad_fn = bw_hook_fn;
    var_clone->output_nr = output_nr++;
    vars[i] = var_clone;
  }
  return bw_hook_fn;
}

template<typename Subclass>
auto TracerHook<Subclass>::registerHook(
        variable_list& vars) -> std::shared_ptr<Subclass> {
  JIT_ASSERT(ThreadTracingState);
  auto id_fn = insertIdentity(vars);
  // We can't use make_shared, because make_shared is not a friend of Subclass,
  // so it can't use its private constructor...
  auto hook = std::shared_ptr<Subclass>(new Subclass());
  id_fn->pre_hooks.emplace_back(hook);
  return hook;
}

////////////////////////////////////////////////////////////////////////////////
// TraceEnterHook
////////////////////////////////////////////////////////////////////////////////

void TraceEnterHook::run(variable_list& vars) {
  JIT_ASSERT(ThreadTracingState);
  auto& graph = ThreadTracingState->graph;

  int num_vars = vars.size();
  for (int i = 0; i < num_vars; ++i) {
    setValueTrace(ThreadTracingState, vars[i], graph->addInput());
  }
  TraceExitHook::registerHook(vars);
}

void TraceEnterHook::registerHook(variable_list& outputs) {
  JIT_ASSERT(ThreadTracingState);
  JIT_ASSERT(outputs.size() > 0);

  // Either no (e.g. after last backward) or all outputs should have a grad fn.
  bool has_grad_fn = static_cast<bool>(outputs[0]->grad_fn);
  for (auto& output : outputs) {
    JIT_ASSERT(static_cast<bool>(output->grad_fn) == has_grad_fn);
  }
  if (!has_grad_fn) return;

  auto hook = TracerHook<TraceEnterHook>::registerHook(outputs);
}

////////////////////////////////////////////////////////////////////////////////
// TraceExitHook
////////////////////////////////////////////////////////////////////////////////

void TraceExitHook::run(variable_list& outputs) {
  // stripped down version of forward_exit
  JIT_ASSERT(ThreadTracingState);
  for (auto& output : outputs) {
    ThreadTracingState->graph->registerOutput(getValueTrace(ThreadTracingState, output, true));
  }
  detail::TraceEnterHook::registerHook(outputs);
  ThreadTracingState->graph->advanceStage(); // ugh
}

void TraceExitHook::registerHook(variable_list& inputs) {
  TracerHook<TraceExitHook>::registerHook(inputs);
}

} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// EvalEnterHook
////////////////////////////////////////////////////////////////////////////////

void EvalEnterHook::run(variable_list& vars) {
  JIT_ASSERT(ThreadTracingState);
  auto& graph = ThreadTracingState->graph;
  Node *eval_node = common_state->eval_node = graph->appendNewNode<Eval>();
  for (auto& input : vars)  {
    eval_node->addInput(tracer::getValueTrace(ThreadTracingState, input, true));
  }
  common_state->next_common_state = EvalExitHook::registerHook(vars);
}

void EvalEnterHook::registerHook(variable_list& outputs, std::shared_ptr<EvalCommonState> common_state) {
  auto hook = TracerHook<EvalEnterHook>::registerHook(outputs);
  hook->common_state = common_state;
}

////////////////////////////////////////////////////////////////////////////////
// EvalExitHook
////////////////////////////////////////////////////////////////////////////////

// TODO: handle saved_variable edges. probably need to go through traces of outputs before overwriting
// and find places where they refer to earlier-stage IR
void EvalExitHook::run(variable_list& vars) {
  JIT_ASSERT(ThreadTracingState);
  auto& graph = ThreadTracingState->graph;
  int num_vars = vars.size();
  for (int i = 0; i < num_vars; ++i) {
    auto& var = vars[i];
    auto select = graph->appendNewNode<Select>(common_state->eval_node, i);
    tracer::setValueTrace(ThreadTracingState, var, select);
  }
  EvalEnterHook::registerHook(vars, common_state->next_common_state);
}

std::shared_ptr<EvalCommonState> EvalExitHook::registerHook(variable_list& inputs) {
  JIT_ASSERT(ThreadTracingState);
  auto hook = TracerHook<EvalExitHook>::registerHook(inputs);
  hook->common_state = std::make_shared<EvalCommonState>();
  return hook->common_state;
}

}}}
