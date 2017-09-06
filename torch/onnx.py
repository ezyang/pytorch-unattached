"""
The torch.onnx module contains functions to export models into the ONNX
IR format.  These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

from __future__ import absolute_import
import torch
import torch.jit
import torch.autograd
import torch.serialization
import contextlib
import re
import io
import collections
from ._utils import _range


def export(model, args, f, export_params=True, verbose=False, training=False):
    """
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported; at the
    moment, it does not support dynamic models (e.g., RNNs.)

    See also: :ref:`onnx-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Variable arguments will
            be hard-coded into the exported model; any Variable arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Variable, this is equivalent
            to having called it with a 1-ary tuple of that Variable.
            (Note: passing keyword arguments to the model is not currently
            supported.  Give us a shout if you need it.)
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, the ordering as specified by ``model.state_dict().values()``.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        training (bool, default False): export the model in training mode.  At
            the moment, ONNX is oriented towards exporting models for inference
            only, so you will generally not need to set this to True.
    """
    _export(model, args, f, export_params, verbose)


@contextlib.contextmanager
def set_training(model, mode):
    """
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.
    """
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def _export(model, args, f, export_params=True, verbose=False, training=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )
    # It's important to run the model in inference mode when exporting;
    # otherwise internal buffers may get updated, dropout gets applied, etc.
    with set_training(model, training):
        # TODO: Pass keyword arguments to model.  Note that record_trace
        # doesn't actually pass on kwargs, so torch.jit API would need
        # to be adjusted.
        trace, torch_out = torch.jit.record_trace(model, *args)
        # TODO: Don't allocate a in-memory string for the protobuf
        if export_params:
            proto = trace.export(list(model.state_dict().values()), verbose)
        else:
            proto = trace.export(verbose)
        torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
        return torch_out


def verify(model, args, backend, verbose=False, training=False, decimal=3, test_args=2):
    """
    Export a model into ONNX, import it into a specified ONNX backend, and then
    on a few random inputs verify that PyTorch and the backend produced the same
    results.  Requires onnx to be installed.

    This function may spuriously fail: some operators are implemented with
    different numerical precision in an ONNX backend, in which case an unstable
    network (e.g., Inception) may blow up these numerical instabilities.  This
    situation is less likely to happen if your model has been trained.  However,
    if this is not the case, you may have found a bug!  Please report it to the
    PyTorch developers.  You can also debug the issue yourself by removing
    suffixes of operators from your model until verification passes.

    For reproduceability, we recommend explicitly setting PyTorch's seed before
    invoking this function.

    Arguments:
        model (torch.nn.Module): the model to be exported and verified
        args (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Variable arguments will
            be hard-coded into the exported model; any Variable arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Variable, this is equivalent
            to having called it with a 1-ary tuple of that Variable.
            (Note: passing keyword arguments to the model is not currently
            supported.  Give us a shout if you need it.)
        backend (onnx.backend module): ONNX backend to verify with
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        training (bool, default False): export the model in training mode.  At
            the moment, ONNX is oriented towards exporting models for inference
            only, so you will generally not need to set this to True.
        decimal (int, default 3): how many decimal places to test precision
        test_args (int or iterable of args, default 2):
            either an integer specifying the number
            of random arguments to generate, or an iterable producing arguments
            to test under.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("To use torch.onnx.verify, you must install the 'onnx' library.")
    import numpy as np
    import difflib

    # TODO: Figure out how to excise the onnx/numpy/precision captures, then
    # move this top-level
    class Errors(object):
        """
        An error-collecting object which supports error recovery.
        """
        def __init__(self, msg):
            self.msg = msg
            self.errors = []
            self.context = []

            class ShortCircuit(Exception):
                pass
            self.exc_class = ShortCircuit

        def requireAlmostEqual(self, x, y, msg=None):
            self.almostEqualAndThen(x, y, msg, self.failWith)

        def checkAlmostEqual(self, x, y, msg=None):
            self.almostEqualAndThen(x, y, msg, self.addErr)

        def almostEqualAndThen(self, x, y, msg, k):
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                try:
                    # decimal is captured
                    np.testing.assert_almost_equal(x, y, decimal=decimal)
                except AssertionError as e:
                    if msg:
                        k("{}: {}".format(msg, str(e).lstrip()))
                    else:
                        k(str(e).lstrip())
            else:
                raise RuntimeError("Unsupported almost equal test")

        def requireEqual(self, x, y, msg=None):
            self.equalAndThen(x, y, msg, self.failWith)

        def checkEqual(self, x, y, msg=None):
            self.equalAndThen(x, y, msg, self.addErr)

        # Bit-for-bit accuracy test
        def equalAndThen(self, x, y, msg, k):
            if isinstance(x, onnx.TensorProto) and isinstance(y, onnx.TensorProto):
                self.equalAndThen(x.name, y.name, msg, k)
                # Use numpy for the comparison
                t1 = onnx.numpy_helper.to_array(x)
                t2 = onnx.numpy_helper.to_array(y)
                if msg:
                    new_msg = "{}, in embedded parameter '{}'".format(msg, x.name)
                else:
                    new_msg = "In embedded parameter '{}'".format(x.name)
                self.equalAndThen(t1, t2, new_msg, k)
            elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                try:
                    np.testing.assert_equal(t1, t2)
                except AssertionError as e:
                    if msg is not None:
                        msg += "\n"
                    else:
                        msg = ""
                    k("{}: {}".format(msg, x.name, str(e).lstrip()))
            else:
                if x != y:
                    # TODO: Better algorithm for lists
                    if msg:
                        k(msg)
                    else:
                        sx = str(x)
                        sy = str(y)
                        if len(sx) > 40 or len(sy) > 40 or '\n' in sx or '\n' in sy:
                            # long form
                            l = "=" * 50
                            k("\nThe value\n{}\n{}\n{}\n\ndoes not equal\n\n{}\n{}\n{}"
                                .format(l, sx, l, l, sy, l))
                        else:
                            k("{} != {}".format(sx, sy))

        def requireMultiLineEqual(self, x, y, msg=None):
            self.multiLineEqualAndThen(x, y, msg, self.failWith)

        def multiLineEqualAndThen(self, x, y, msg, k):
            if msg is None:
                msg = "Strings are not equal:"
            if x != y:
                diff = difflib.ndiff(x.splitlines(True), y.splitlines(True))
                k("{}\n\n{}".format(msg, "".join(diff)))

        def addErr(self, msg):
            # TODO: instead of putting in strings, delay context
            msg_w_ctx = msg
            for c in reversed(self.context):
                msg += "\n\n  * " + "\n    ".join(c.splitlines())
            self.errors.append(msg)

        def fail(self):
            raise self.exc_class()

        def failWith(self, msg):
            self.addErr(msg)
            self.fail()

        def failIfErrs(self):
            if self.errors:
                self.fail()

        def recover(parent_self):
            class Recover(object):
                def __enter__(self):
                    pass

                def __exit__(self, exc_type, exc_value, traceback):
                    if exc_type == parent_self.exc_class:
                        return True
            return Recover()

        def addErrCtxt(parent_self, msg):
            class AddContext(object):
                def __enter__(self):
                    parent_self.context.append(msg)

                def __exit__(self, exc_type, exc_value, traceback):
                    parent_self.context.pop()
            return AddContext()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.errors:
                errors_msg = "\n\n".join(map(lambda x: "ERROR: " + x, self.errors))
                final_msg = "{}\n{}\n{}".format(self.msg, '-' * 70, errors_msg)
                raise AssertionError(final_msg)
            if exc_type == self.exc_class:
                raise RuntimeError("ShortCircuit was raised, but no errors were recorded")

    def is_variable(o):
        return isinstance(o, torch.autograd.Variable)

    def randomize_arg(arg):
        new_data = arg.data.clone()
        # For now, don't try randomizing non-float tensors; these
        # are likely to be things like indices, where just randomly
        # spattering some longs is unlikely to work.  One way we could
        # make this work is to apply a random permutation or something.
        if hasattr(new_data, 'uniform_'):
            new_data.uniform_()
        return torch.autograd.Variable(new_data, volatile=arg.volatile, requires_grad=arg.requires_grad)

    def randomize_args(args):
        return torch.autograd.function._nested_map(is_variable, randomize_arg)(args)

    def backend_args(args):
        # TODO: onnx should accept iterables
        return tuple(v.data.cpu().numpy() for v in torch.autograd.function._iter_variables(args))

    def load_bytes(b):
        b.seek(0)
        return onnx.load(b)

    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    with set_training(model, training):
        proto_bytes = io.BytesIO()
        torch_out = _export(model, args, proto_bytes, verbose=verbose, training=training)
        proto = load_bytes(proto_bytes)
        prepared = backend.prepare(proto)

        def run(args):
            alt_proto_bytes = io.BytesIO()
            torch_out = _export(model, args, alt_proto_bytes, verbose=verbose, training=training)
            if proto_bytes.getvalue() != alt_proto_bytes.getvalue():
                # OK, let's try to figure out what happened.
                msg = "When I exported your model with different inputs, the result was different."
                if not verbose:
                    msg += "\n(To get more information, run torch.onnx.verify(..., verbose=True))"
                with Errors(msg) as errs:
                    alt_proto = load_bytes(alt_proto_bytes)

                    # First, check if the parameters have the same order.
                    # If they don't, something has *really* gone wrong.
                    initializer_hint = ("A difference in embedded parameters usually means that\n"
                                        "your model is updating parameters/buffers even in inference\n"
                                        "mode.  Look for a buggy nn.Module which isn't respecting train().\n")
                    with errs.recover(), errs.addErrCtxt(initializer_hint):
                        # If the initializer order is jumbled up, we have bigger
                        # problems.
                        errs.requireEqual(list(map(lambda x: x.name, proto.initializer)),
                                          list(map(lambda x: x.name, alt_proto.initializer)))
                        for x, y in zip(proto.initializer, alt_proto.initializer):
                            errs.checkEqual(x, y)

                    # Next, check if the model structure lines up.
                    structure_hint = ("A difference in model structure usually means that\n"
                                      "your model has dynamic control flow.  These models are not\n"
                                      "currently supported by the exporter.")
                    with errs.recover(), errs.addErrCtxt(structure_hint):
                        # TODO: This currently relies on the graph exporter not
                        # actually printing the initializers
                        errs.requireMultiLineEqual(onnx.helper.printable_graph(proto),
                                                   onnx.helper.printable_graph(alt_proto))
                        # Not very user friendly. Last resort!
                        # NB: Delete initializers since we already tested them
                        stripped_proto = onnx.GraphProto()
                        stripped_proto.CopyFrom(proto)
                        stripped_alt_proto = onnx.GraphProto()
                        stripped_alt_proto.CopyFrom(alt_proto)
                        del stripped_proto.initializer[:]
                        del stripped_alt_proto.initializer[:]
                        errs.requireMultiLineEqual(str(stripped_proto), str(stripped_alt_proto))
                        errs.requireEqual(stripped_proto, stripped_alt_proto)

                    errs.failIfErrs()

                    # At this point, we should have figured out why the binary
                    # protobufs differed, and short-circuited out of this code
                    # with a helpful error message.  But what if we didn't?
                    # We better still try to give a good error message in this
                    # case.  We EXPECT these requires to fail.  If they don't,
                    # that is a bug in verify
                    errs.requireEqual(proto, alt_proto)
                    errs.requireEqual(proto_bytes.getvalue(), alt_proto_bytes.getvalue())
                    assert False

            # TODO: test that the traced model also returns the same thing...
            run_helper(torch_out, args)

        # Factored out so we can avoid one run of the model
        def run_helper(torch_out, args):
            backend_out = prepared.run(backend_args(args))
            if isinstance(torch_out, torch.autograd.Variable):
                torch_out = (torch_out,)
            # NB: onnx backend NEVER returns bare numpy array
            msg = "ONNX backend returned different results from PyTorch"
            result_hint = ("If you are not using trained parameters, a difference in results\n"
                           "could mean that your network is numerically unstable.  Otherwise\n"
                           "it indicates a bug in PyTorch/ONNX; please file a bug report.")
            with Errors(msg) as errs, errs.addErrCtxt(result_hint):
                for i, (x, y) in enumerate(zip(torch_out, backend_out)):
                    errs.checkAlmostEqual(x.data.cpu().numpy(), y, "In output {}".format(i))

        run_helper(torch_out, args)

        if isinstance(test_args, int):
            for i in range(test_args):
                run(randomize_args(args))
        else:
            for test_arg in test_args:
                run(test_arg)


attr_pattern = re.compile("^(.+)_([ifstg])$")


def _add_attribute(node, key, value):
    """ initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if isinstance(value, collections.Iterable):
        kind += "s"
    return getattr(node, kind + '_')(name, value)


def _newNode(self, opname, *args, **kwargs):
    n = self.create(opname, args)
    for k, v in sorted(kwargs.items()):
        _add_attribute(n, k, v)
    return n


def _op(self, opname, *args, **kwargs):
    outputs = kwargs.pop('outputs', 1)
    n = self.appendNode(_newNode(self, opname, *args, **kwargs))
    if outputs == 1:
        return n
    return tuple(self.appendNode(self.createSelect(n, i)) for i in _range(outputs))

torch._C.Graph.op = _op
