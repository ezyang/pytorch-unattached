#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <sstream>

namespace torch { namespace jit {

std::string getPythonName(const PyObject* obj, bool is_legacy) {
  AutoGIL gil;
  if (is_legacy) {
    return std::string(obj->ob_type->tp_name);
  } else {
    // NB: hypothetically __name__ could mutate the Python
    // object in a externally visible way. Please don't!
    auto wobj = const_cast<PyObject*>(obj);
    THPObjectPtr name{PyObject_GetAttrString(wobj, "__name__")};
    return THPUtils_unpackString(name.get());
  }
}

std::ostream& operator<<(std::ostream & out, const node_list & nodes) {
  size_t i = 0;
  for(auto n : nodes) {
    if(i++ > 0)
      out << ", ";
    if(auto s = n->cast<Select>())
      out << "%" << s->base()->unique() << "." << s->offset();
    else
      out << "%" << n->unique();
  }
  return out;
}

static std::ostream& operator<<(std::ostream & out, THPObjectPtr& obj) {
   THPObjectPtr repr { PyObject_Repr(obj.get()) };
   return out << THPUtils_unpackString(repr.get());
}

std::ostream& operator<<(std::ostream & out, Graph & g) {
  out << "graph(" << g.inputs() << ") {\n";
  for(auto n : g.nodes()) {
    out << "  %" << n->unique() << " = ";
    IR_IF(n,PythonOp)
      out << getPythonName(value->pyobj.get(),value->is_legacy);
      for (auto& scalar : value->scalar_args) {
        out << " " << scalar;
      }
      out << "(" << value->inputs() << ");\n";
    IR_ELSEIF(Select)
    IR_ELSE()
      out << toString(n->kind()) << "(" << n->inputs() << ");\n";
    IR_END()
    }
  out << "  return (" << g.outputs() << ");\n}\n";
  return out;
}

// NB: Prefer using at() for bounds checking.
void Graph::lint() const {
  // Basic structural properties
  std::unordered_set<Node*> known_nodes;
  for (auto& n : all_nodes) {
    JIT_ASSERT(!known_nodes.count(n));
    known_nodes.insert(n);
  }
  // TODO: nodes_ is a permutation of known_nodes
  for (auto& n : nodes_) {
    JIT_ASSERT(known_nodes.count(n));
  }
  std::unordered_set<Node*> known_inputs;
  for (auto& n : inputs_) {
    JIT_ASSERT(known_nodes.count(n));
    JIT_ASSERT(!known_inputs.count(n));
    JIT_ASSERT(n->kind_ == NodeKind::Param);
    known_inputs.insert(n);
  }
  JIT_ASSERT(known_nodes.count(output_));
  JIT_ASSERT(output_->kind_ == NodeKind::Return);
  for (auto& n : all_nodes) {
    JIT_ASSERT(all_nodes.at(n->unique_) == n);
    for (auto& u : n->uses_) {
      JIT_ASSERT(known_nodes.count(u.user));
      JIT_ASSERT(u.user->inputs_.at(u.offset) == n);
    }
    // TODO: The select invariant (there is a unique select node for each
    // output of an op, if the node is a multi-return op).
    // See https://github.com/ezyang/pytorch/issues/8
    IR_IF(n, Return)
      JIT_ASSERT(n->uses_.size() == 0);
    IR_ELSEIF(Param)
      JIT_ASSERT(n->inputs_.size() == 0);
    IR_ELSEIF(Select)
      JIT_ASSERT(n->inputs_.size() == 1);
    IR_ELSEIF(PythonOp)
      int n_scalars = 0, n_tensors = 0;
      for (auto c : value->cconv) {
        if (c == 's') {
          n_scalars++;
        } else if (c == 't') {
          n_tensors++;
        } else {
          JIT_ASSERT(0);
        }
        JIT_ASSERT(value->pyobj != nullptr);
      }
      JIT_ASSERT(n_scalars == value->scalar_args.size());
      JIT_ASSERT(n_tensors == n->inputs_.size());
    IR_ELSE()
      // no-op
    IR_END()
  }

}

}}
