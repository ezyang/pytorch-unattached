#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <iostream>
#include <unordered_map>
#include <set>
#include <stack>
#include <sstream>
#include <algorithm>

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

using node_set = std::set<Node*>;
#define ALL_OF(container) container.begin(), container.end()

// NB: Prefer using at() for bounds checking.
void Graph::lint() const {
  node_set all_nodes_set(ALL_OF(all_nodes));
  node_set nodes_set(ALL_OF(nodes_));
  node_set inputs_set(ALL_OF(inputs_));
  node_set output_set {output_};

  // no duplicates
  JIT_ASSERT(all_nodes_set.size() == all_nodes.size());
  JIT_ASSERT(nodes_set.size() == nodes_.size());
  JIT_ASSERT(inputs_set.size() == inputs_.size());

  // all_nodes must contain all of our other nodes
  JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(nodes_set)));
  JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(inputs_set)));
  JIT_ASSERT(std::includes(ALL_OF(all_nodes_set), ALL_OF(output_set)));

  // the only possible types of nodes are inputs, outputs and computation nodes
  node_set sum_set;
  sum_set.insert(ALL_OF(nodes_set));
  sum_set.insert(ALL_OF(inputs_set));
  sum_set.insert(ALL_OF(output_set));
  JIT_ASSERT(std::includes(ALL_OF(sum_set), ALL_OF(all_nodes_set)));

  // inputs, outputs and computation nodes are disjoint
  JIT_ASSERT(std::find_first_of(ALL_OF(nodes_set), ALL_OF(inputs_set)) == nodes_set.end());
  JIT_ASSERT(std::find_first_of(ALL_OF(nodes_set), ALL_OF(output_set)) == nodes_set.end());
  JIT_ASSERT(std::find_first_of(ALL_OF(inputs_set), ALL_OF(output_set)) == inputs_set.end());

  // all inputs should be Params, output should be Return
  JIT_ASSERT(std::all_of(ALL_OF(inputs_set), [](Node* n) { return n->kind_ == NodeKind::Param; }));
  JIT_ASSERT(std::all_of(ALL_OF(output_set), [](Node* n) { return n->kind_ == NodeKind::Return; }));
  JIT_ASSERT(std::all_of(ALL_OF(nodes_set),  [](Node* n) {
    return n->kind_ != NodeKind::Return && n->kind_ != NodeKind::Param;
  }));

  // per node properties
  for (auto& n : all_nodes) {

    // unique is the index into all_nodes
    JIT_ASSERT(all_nodes.at(n->unique_) == n);

    // uses information is consistent
    for (auto& u : n->uses_) {
      JIT_ASSERT(all_nodes_set.count(u.user));
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
