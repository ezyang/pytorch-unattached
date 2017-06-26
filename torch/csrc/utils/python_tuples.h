#pragma once

#include <Python.h>

// Comparing iterators from different tuples is undefined behavior
class PyTuple {
private:
  PyObject* tuple;

public:
  PyTuple(PyObject* tuple) : tuple(tuple) {}

  class proxy {
    PyObject* tuple;
    Py_ssize_t index;
  public:
    proxy(PyObject *tuple, Py_ssize_t index) : tuple(tuple), index(index) {}
    operator PyObject*() const { return PyTuple_GET_ITEM(tuple, index); }
    void operator=(PyObject* o) {
      PyTuple_SET_ITEM(tuple, index, o);
    }
  };

  class iterator {
    PyObject* tuple;
    Py_ssize_t index;
  public:
    using difference_type = Py_ssize_t;
    using value_type = proxy;
    using pointer = proxy*;
    using reference = proxy&;
    using iterator_category = std::forward_iterator_tag;

    iterator(PyObject *tuple, Py_ssize_t index) : tuple(tuple), index(index) {}

    proxy operator*() const { return proxy(tuple, index); }
    iterator& operator++() { index++; return *this; }
    iterator operator++(int) { auto r = *this; ++(*this); return r; }
    bool operator==(iterator other) const { return index == other.index; }
    bool operator!=(iterator other) const { return !(*this == other); }
    friend difference_type operator-(iterator it1, iterator it2) { return it1.index - it2.index; }


  };

  iterator begin() {
    return iterator(tuple, 0);
  }

  iterator end() {
    return iterator(tuple, size());
  }

  Py_ssize_t size() {
    return PyTuple_GET_SIZE(tuple);
  }
};
