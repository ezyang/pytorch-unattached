#pragma once

#include <Python.h>
#include <memory>
#include <string>
#include <mutex>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/engine.h"

namespace torch { namespace autograd {

struct EvalOutput : Function {
  EvalOutput(const edge_type& next_edge)
    : next_edge(next_edge) {
    num_inputs = 1;
    is_executable = next_edge.first->is_executable;
  }

  virtual variable_list apply(const variable_list& inputs) override {
    throw std::logic_error("EvalOutput::apply() called");
  }

  edge_type next_edge;
};

struct Eval : Function {
  struct edge_hasher {
    std::size_t operator()(const edge_type& edge) const {
#define HASH_IDX(idx) std::hash<std::tuple_element<idx, edge_type>::type>()(std::get<idx>(edge))
      // TODO: that's probably a bad hash function, but whatever
      return HASH_IDX(0) ^ HASH_IDX(1);
    }
  };
  using edge_set = std::unordered_set<edge_type, edge_hasher>;
  using placeholder_list = std::vector<std::shared_ptr<EvalOutput>>;

  // This struct has only one member, but it's useful to e.g. add a set of all
  // nodes when debugging this stuff, so I'm leaving it as is.
  struct Subgraph {
    struct Boundary {
      // All nodes from within the subgraph that connect to the outside.
      // These are the places that will need to be patched to point to placeholders.
      // Contains pairs of (fn, offset into next_functions).
      edge_set begins;
      // All nodes that are not in the subgraph, but are in the union of
      // next_functions of all nodes from the subgraph. These are the places that
      // will be modeled by placeholders.
      // Contains pairs of (fn, input_nr) and is equivalent to next_functions
      // of an Eval node that will replace the subgraph.
      edge_set ends;
    };

    Boundary boundary;
  };

  virtual inline bool is_traceable() final { return traceable; }

  virtual variable_list apply(const variable_list& inputs) override;

  void replaceSubgraph(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders = placeholder_list());

  static variable_list filterRelevantOutputs(const variable_list& inputs, const variable_list& outputs);

  function_list roots;
  placeholder_list placeholders;
  bool traceable = false;

private:
  Engine::callback_map getCallbacks(variable_list& outputs, std::mutex& outputs_mutex);

  Subgraph getSubgraph(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders);
};

}} // namespace torch::autograd
