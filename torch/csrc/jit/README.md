# jit

The jit directory contains infrastructure for a just-in-time compiler for
PyTorch.

TODO: Describe the general philosophy of the JIT.

## Well-known functions

Ordinarily, when defining a compiler you want the set of functions to be user
extensible; e.g., a user can add to the set of defined functions by defining an
appropriate autograd Function.  However, there are some functions where we want
to make assumptions about their semantics, because we are going to write
optimizations over them or insert them into the program.  Such functions are
"well-known" functions, because the JIT compiler knows about them, and a user
implementation must abide by the contract (sometimes implicitly) specified by
the compiler.

A well-known function is usually implemented in several parts:

* First, we pre-intern the string (`interned_strings.h`) that identifies
  the node.  This allows us to more conveniently refer to these operators
  without having to first do a lookup through the intern table.

* If we generate this operator during optimizations, we will often have
  a helper function in `Graph` (`ir.h`) for creating the operator.  This is
  the easiest way to find out, in code, what attributes we assume for an
  operator.

* There is a runtime interpretation of the operator in
  `torch/csrc/autograd/functions/jit_closure.cpp`, which specifies how we
  actually interpret programs that contain such an operator.

So, whence the specifications!  For the most part, we are following
the [ONNX operator specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md)
to determine the semantics of our operators.  However, there are a few
other well-known functions which are specific to PyTorch.

* **FusionGroup**

  A fusion group takes some number of input tensors, applies a graph `Subgraph`
  to them, producing the returned tensors of the subgraph.  Operationally,
  operators inside a FusionGroup are fused into a single kernel, so that their
  intermediate results are never materialized.  Not all operators support
  fusion:

  * **attribute**:
    <dl>
      <dt>Subgraph</dt>
      <dd>The graph of fused operators.  Its inputs and outputs should match
      the number of inputs and outputs to the FusionGroup operator.</dd>
    </dl>
  * **input**: 1 - ∞ (same as inputs of Subgraph)
  * **output**: 1 - ∞ (same as outputs of Subgraph)

* **Eval** (renders as `CppOp[N5torch8autograd4EvalE]`)

  An Eval node takes some inputs, and an autograd closure `Handle`.  It applies
  those inputs to the autograd closure, and returns the results of having
  executed the closure.  An Eval node is primarily used to implement backwards
  operations for black box forward operations: because the backwards computation
  of a black box forwards is not known until we actually execute the forward
  operation, we have to run the forward computation, giving us an autograd
  closure to compute backwards, and then run it later when we actually
  execute backwards.

  * **input**:
    <dl>
      <dt>Input1, Input2, ...</dt>
      <dd>Any number of inputs, which will be passed as inputs to the
      autograd closure</dd>
      <dt>Handle</dt>
      <dd>An autograd closure (opaquely represented with type `Handle` in our
      IR) which specifies how to execute the operation.)</dd>
    </dl>
  * **output**: 1 - ∞ (same as outputs of autograd closure)

