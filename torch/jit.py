import torch.autograd.function as F
import torch._C
from torch.autograd import Variable
import types
import contextlib

# Example how to use:
#
# import torch.jit
# model = model.RNNModel(args.model, ...)
# model = torch.jit.wrap_model(model)


def flatten(x):
    return tuple(F._iter_variables(x))


def record_trace(f, inputs):
    inputs = torch._C._tracer_enter(inputs)
    out = f()
    # TODO: unflatten
    trace = torch._C._tracer_exit(flatten(out))
    torch._C._jit_pass_lint(trace)
    return (trace, out)


@contextlib.contextmanager
def fork_rng():
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.  This is important
    if we are evaluating a trace twice, and it incorporates
    randomness: if we don't reset the seed, we might get totally
    different results!

    TODO: Randomness in models is a big problem for reproduceability,
    because it means if we start executing models out of order,
    they may behave differently.  Interesting question is whether
    or not backwards pass ever has randomness.  I hope not.
    """
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = None
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state()

    yield

    torch.set_rng_state(cpu_rng_state)
    if gpu_rng_state is not None:
        torch.cuda.set_rng_state(gpu_rng_state)


# LIMITATIONS:
# - This assumes that the model will run exactly identically the
#   next time you call forward; if the model looks at some global
#   variables, or has data dependent computation, we will silently
#   give the wrong result.  Adding sanity checking is a TODO.
def wrap_model(model):
    """
    Trace a model the first time you run it, and then use that trace
    to execute the model on subsequent runs.
    """
    real_forward = model.forward

    def forward(self, *args):
        if not hasattr(self, "saved_trace"):
            # TODO: saved_out LEAKS those tensors
            self.saved_trace, self.saved_out = \
                record_trace(lambda: real_forward(*args),
                             tuple(self.parameters()) + flatten(args))
            return self.saved_out
        else:
            flat_out = Variable._execution_engine.run_forward(
                self.saved_trace, tuple(self.parameters()) + flatten(args))
            return F._unflatten(flat_out, self.saved_out)
    model.forward = types.MethodType(forward, model)
    return model


def verify_model(model):
    """
    Test if a model can be JITed, by tracing it, and then running the
    real model and the trace side-by-side.  This will throw an error
    if they give different results.  Once you have verified they behave
    identically, you can use wrap_model.
    """
    real_forward = model.forward

    def forward(self, *args):
        if not hasattr(self, "saved_trace"):
            self.saved_trace, real_out = \
                record_trace(lambda: real_forward(*args),
                             tuple(self.parameters()) + flatten(args))
            return real_out
        else:
            # clone the input tensors and run the tracing engine
            cloned_inputs = []
            for inp in tuple(self.parameters()) + flatten(args):
                # It doesn't matter that we throw away flags, because
                # we're not going to try to do backwards on the trace output.
                cloned_inputs.append(Variable(inp.data.clone()))

            with fork_rng():
                flat_trace_out = Variable._execution_engine.run_forward(self.saved_trace, tuple(cloned_inputs))

            # run the real model on the actual variables
            real_out = real_forward(*args)
            flat_real_out = flatten(real_out)

            # test for equality
            for x, y in zip(flat_trace_out, flat_real_out):
                if isinstance(x, Variable) and isinstance(y, Variable):
                    # TODO: Could there ever be numeric instability?
                    if not x.data.equal(y.data):
                        print(x)
                        print(y)
                        raise "JIT and real computation mismatch"
                else:
                    print(x)
                    print(y)
                    raise "Output is not variables"

            return real_out
    model.forward = types.MethodType(forward, model)
    return model


def print_trace(model):
    """
    Trace and print the trace for a model, do not try to execute trace.
    """
    real_forward = model.forward

    def forward(self, *args):
        if not hasattr(self, "saved_trace"):
            self.saved_trace, real_out = \
                record_trace(lambda: real_forward(*args),
                             tuple(self.parameters()) + flatten(args))
            print(self.saved_trace)
            return real_out
        else:
            real_out = real_forward(*args)
            return real_out

    model.forward = types.MethodType(forward, model)
    return model


def trace_model(model):
    """
    Trace a function, but also return the trace.  If original
    function returns output, this function returns (trace, output).

    Unlike verify_model/wrap_model, this does NOT cache the trace
    and attempt to rerun it.
    """
    real_forward = model.forward

    def forward(self, *args):
        return record_trace(lambda: real_forward(*args),
                            tuple(self.parameters()) + flatten(args))
    model.forward = types.MethodType(forward, model)
    return model


def trace_fn(f):
    """
    Trace a function the first time you run it, but also
    return the trace.  This function returns (trace, output)
    """
    def go(*args):
        return record_trace(lambda: f(*args), flatten(args))
    return go

if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")
