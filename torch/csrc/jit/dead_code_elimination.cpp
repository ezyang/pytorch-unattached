#include "torch/csrc/jit/dead_code_elimination.h"

namespace torch { namespace jit {

void EliminateDeadCode(std::unique_ptr<Graph>& graph) {
  auto& nodes = graph->nodes();
  auto go = [&](Node *node) {
    if (node->uses().size() == 0) {
      node->destroy();
    }
  };
  if (nodes.end() != nodes.begin()) {
    auto it = std::prev(nodes.end());
    // nodes.begin() handling hoisted out of loop to avoid
    // UB from nodes.begin()-1
    while (it != nodes.begin()) {
      Node *node = *it;
      it--; // avoid iterator invalidation
      go(node);
    }
    go(*nodes.begin());
  }
}

}}
