#pragma once

#include <Python.h>
#include "torch/csrc/autograd/engine.h"

bool THPEngine_initModule(PyObject *module);

namespace torch { namespace autograd { namespace python {

struct PythonEngine : public Engine {
  virtual void thread_init(int device) override;
  virtual void thread_on_exception(FunctionTask& task, std::exception& e) override;
  virtual void execute(
      const function_list& roots,
      const variable_list& inputs,
      bool keep_graph,
      const pre_callback_map& pre_callbacks = pre_callback_map(),
      const post_callback_map& post_callbacks = post_callback_map()) override;

  static PythonEngine& getDefaultEngine();
};

}}} // namespace torch::autograd::python
