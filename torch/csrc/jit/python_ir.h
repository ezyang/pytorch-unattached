#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/jit/ir.h"

struct THPGraph {
    PyObject_HEAD
    std::shared_ptr<torch::jit::Graph> cdata;
};

extern PyObject *THPGraphClass;

PyObject * THPGraph_Wrap(const std::shared_ptr<torch::jit::Graph> node);

inline bool THPGraph_Check(PyObject *obj)
{
  return THPGraphClass && PyObject_IsInstance(obj, THPGraphClass);
}

bool THPIR_initModule(PyObject *module);
