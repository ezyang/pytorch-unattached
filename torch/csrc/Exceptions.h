#ifndef THP_EXCEPTIONS_H
#define THP_EXCEPTIONS_H

#include <exception>
#include <stdexcept>
#include <string>
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/auto_gil.h"

#define HANDLE_TH_ERRORS                                                       \
  try {

#define END_HANDLE_TH_ERRORS_RET(retval)                                       \
  } catch (python_error &e) {                                                  \
    return retval;                                                             \
  } catch (std::exception &e) {                                                \
    PyErr_SetString(PyExc_RuntimeError, e.what());                             \
    return retval;                                                             \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(NULL)

extern PyObject *THPException_FatalError;

// Throwing this exception means that the python error flags have been already
// set and control should be immediately returned to the interpreter.
struct python_error : public std::exception {
  python_error() : type(nullptr), value(nullptr), traceback(nullptr) {}

  ~python_error() {
    if (type || value || traceback) {
      AutoGIL gil;
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    }
  }

  /** Saves the exception so that it can be re-thrown on a different thread */
  inline void persist() {
    // PyErr_Fetch overwrites the pointers
    AutoGIL gil;
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Fetch(&type, &value, &traceback);
  }

  /** Sets the current Python error from this exception */
  inline void restore() {
    if (!type) return;
    // PyErr_Restore steals references
    AutoGIL gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
    PyErr_Restore(type, value, traceback);
  }

  PyObject* type;
  PyObject* value;
  PyObject* traceback;
};

#ifdef _THP_CORE

bool THPException_init(PyObject *module);
#endif

#endif
