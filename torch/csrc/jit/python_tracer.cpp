#include <Python.h>
#include "torch/csrc/jit/python_tracer.h"

#include "torch/csrc/autograd/jit_closure.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/python_ir.h"
#include "torch/csrc/THP.h"

using namespace torch::autograd;
using namespace torch::jit;

PyObject * THPTracer_enter(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* input_objs;
  if (!PyArg_ParseTuple(args, "O", &input_objs)) {
    return NULL;
  }
  THPUtils_assert(PyTuple_Check(input_objs), "inputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(input_objs));
  Py_ssize_t num_inputs = PyTuple_GET_SIZE(input_objs);

  variable_list inputs;
  for (int i = 0; i < num_inputs; i++) {
    PyObject* input_obj = PyTuple_GET_ITEM(input_objs, i);
    THPUtils_assert(THPVariable_Check(input_obj), "element %d of input "
        "tuple is not a Variable", i);
    inputs.emplace_back(((THPVariable*)input_obj)->cdata);
  }

  return THPGraph_Wrap(tracer::enter(inputs));
  END_HANDLE_TH_ERRORS
}

PyObject * THPTracer_exit(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* graph_obj = NULL;
  PyObject* output_objs = NULL;
  if (!PyArg_ParseTuple(args, "OO", &graph_obj, &output_objs)) {
    return NULL;
  }

  THPUtils_assert(THPGraph_Check(graph_obj), "graph argument is not a graph");
  THPUtils_assert(PyTuple_Check(output_objs), "outputs argument is "
    "expected to be a tuple, but got %s", THPUtils_typename(output_objs));
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(output_objs);

  variable_list outputs;
  for (int i = 0; i < num_outputs; i++) {
    PyObject* output_obj = PyTuple_GET_ITEM(output_objs, i);
    THPUtils_assert(THPVariable_Check(output_obj), "element %d of outputs "
        "tuple is not a Variable", i);
    auto& var = ((THPVariable*)output_obj)->cdata;
    outputs.emplace_back(var);
  }

  THPGraph *py_graph = (THPGraph*)graph_obj;
  tracer::exit(py_graph->cdata, outputs);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPTracer_createAutogradClosure(PyObject *_unused, PyObject *pygraph) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPGraph_Check(pygraph), "getClosure expected a Graph, but got %s",
      THPUtils_typename(pygraph));
  auto& graph = ((THPGraph*)pygraph)->cdata;

  auto closure = createAutogradClosure(graph.get());

  return THPWrapper_New(closure.release(),
                        [](void *fn_list) { delete reinterpret_cast<AutogradClosure*>(fn_list); });
  END_HANDLE_TH_ERRORS
}
