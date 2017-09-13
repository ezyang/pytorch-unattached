#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/THP.h"

namespace py = pybind11;

namespace pybind11 { namespace detail {

template<> struct type_caster<torch::jit::tracer::TraceInput> {
public:
  PYBIND11_TYPE_CASTER(torch::jit::tracer::TraceInput, _("torch::jit::tracer::TraceInput"));
  bool load(handle src, bool) {
    PyObject *source = src.ptr();
    if (THPVariable_Check(source)) {
      value = torch::jit::tracer::TraceInput(((THPVariable*)source)->cdata);
      return true;
    } else if (THPModule_isTensor(source)) {
      value = torch::jit::tracer::TraceInput(torch::createTensor(source));
      return true;
    } else {
      return false;
    }
  }
  static handle cast(torch::jit::tracer::TraceInput src, return_value_policy /* policy */, handle /* parent */) {
    if (src.variable.defined()) {
      return handle(THPVariable_Wrap(src.variable));
    } else {
      return handle(torch::createPyObject(src.buffer));
    }
  }
};

template<> struct type_caster<torch::autograd::Variable> {
public:
  PYBIND11_TYPE_CASTER(torch::autograd::Variable, _("torch::autograd::Variable"));
  bool load(handle src, bool) {
    PyObject *source = src.ptr();
    if (THPVariable_Check(source)) {
      value = ((THPVariable*)source)->cdata;
      return true;
    } else {
      return false;
    }
  }
  static handle cast(torch::autograd::Variable src, return_value_policy /* policy */, handle /* parent */) {
    return handle(THPVariable_Wrap(src));
  }
};

template <> struct type_caster<torch::jit::Symbol> {
public:
  PYBIND11_TYPE_CASTER(torch::jit::Symbol, _("Symbol"));

  bool load(handle src, bool) {
    try {
      value = torch::jit::stringToSymbol(py::cast<std::string>(src));
    } catch (std::exception& e) {
      return false;
    }
    return true;
  }

  static handle cast(torch::jit::Symbol src, return_value_policy /* policy */, handle /* parent */) {
    return py::cast(std::string(torch::jit::symbolToString(src)), return_value_policy::copy).release();
  }
};

template <> struct type_caster<torch::jit::AttributeKind> {
public:
  PYBIND11_TYPE_CASTER(torch::jit::AttributeKind, _("AttributeKind"));

  bool load(handle src, bool) {
    return false;
  }

  static handle cast(torch::jit::AttributeKind src, return_value_policy /* policy */, handle /* parent */) {
    return py::cast(std::string(torch::jit::toString(src)), return_value_policy::copy).release();
  }
};

}} // namespace pybind11::detail

