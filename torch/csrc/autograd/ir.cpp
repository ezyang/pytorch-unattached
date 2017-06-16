#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <sstream>

namespace torch { namespace autograd {

std::string InputNode::name() const {
  return std::string("Variable ") + debug_name;
}

std::string PyNode::name() const {
  AutoGIL gil;
  // NB: hypothetically __name__ could mutate the Python
  // object in a externally visible way. Please don't!
  auto wobj = const_cast<PyObject*>(pyobj.get());
  THPObjectPtr name{PyObject_GetAttrString(wobj, "__name__")};
  // TODO: missing error check
  return std::string(PyString_AsString(name));
}

void printFreeVariables(const Node* root) {
  std::unordered_set<const Node*> seen;
  std::stack<const Node*> todo;
  todo.push(root);
  std::cout << "-- BEGIN free variables --" << std::endl;
  while (!todo.empty()) {
    auto n = todo.top();
    todo.pop();
    if (seen.count(n)) continue;
    for (auto c : n->inputs) {
      todo.push(c.node.get());
    }
    seen.insert(n);
    if (auto input = dynamic_cast<const InputNode*>(n)) {
      std::stringstream ss;
      ss << n;
      std::cout << input->name() << " " << ss.str() << std::endl;
    }
  }
  std::cout << "-- END free variables --" << std::endl;
}

void printGraph(const Node* root) {
  std::cout << "-- BEGIN graph --" << std::endl;
  std::unordered_map<const Node*, int> ids;
  std::vector<const Node*> nodes;
  int id = 0;
  std::stack<const Node*> todo;
  todo.push(root);
  while (!todo.empty()) {
    auto n = todo.top();
    todo.pop();
    if (ids.find(n) != ids.end()) continue;
    for (auto c : n->inputs) {
      todo.push(c.node.get());
    }
    ids[n] = id;
    nodes.emplace_back(n);
    id++;
  }

  for (auto n : nodes) {
    std::cout << ids.at(n) << ": " << n->name() << "(";
    for (auto c : n->inputs) {
      std::cout << ids.at(c.node.get()) << "[" << c.output_nr << "], ";
    }
    std::cout << ")" << std::endl;
  }
  std::cout << "-- END graph --" << std::endl;
}

}}
