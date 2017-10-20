import torch
from torch.autograd._functions.utils import check_onnx_broadcast  # TODO: move me
from torch.nn.modules.utils import _pair

# EDITING THIS FILE? READ THIS FIRST!
#
# - Parameter ordering does NOT necessarily match what is in VariableType.cpp;
#   tensors are always first, then non-tensor arguments.
# - Parameter names must *exactly* match the names in VariableType.cpp, because
#   dispatch is done with keyword arguments.
# - Looking for inplace ops?  They're detected by the trailing underscore, and
#   transparently dispatched to their non inplace versions in
#   'run_symbolic_function'.   See Note [Export inplace]

# Helper functions:


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


# Symbolics:


def add(g, self, other, alpha):
    if _scalar(alpha) != 1:
        raise NotImplementedError("add: alpha != 1")
    return g.op("Add", self, other)


def sub(g, self, other, alpha):
    if _scalar(alpha) != 1:
        raise NotImplementedError("sub: alpha != 1")
    return g.op("Sub", self, other)


def mul(g, self, other):
    return g.op("Mul", self, other)


def div(g, self, other):
    return g.op("Div", self, other)


# TODO: untested
def addmm(g, self, mat1, mat2, beta, alpha):
    return g.op("Gemm", mat1, mat2, self, beta_f=_scalar(beta), alpha_f=_scalar(alpha))


def tanh(g, self):
    return g.op("Tanh", self)


def sigmoid(g, self):
    return g.op("Sigmoid", self)


def mean(g, self, dim=None, keepdim=None):
    kwargs = {}
    # NB: ONNX's default is different from PyTorch's
    if keepdim is None:
        keepdim = 0
    return g.op("ReduceMean", self, axes_i=dim, keepdims_i=keepdim)


def t(g, self):
    return g.op("Transpose", self, perm_i=(1, 0))


def transpose(g, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    axes = list(range(len(self.type().sizes())))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return g.op("Transpose", self, perm_i=axes)


def view(g, self, size):
    return g.op("Reshape", self, shape_i=size)


def squeeze(g, self, dim=None):
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [dim]
    return g.op("Squeeze", self, axes_i=dims)


def prelu(g, input, weight):
    if all(s == 1 for s in weight.type().sizes()):
        raise RuntimeError("single weight shared among input channels not supported")
    return g.op("PRelu", input, weight)


def threshold(g, input, threshold, value, inplace):
    # See Note [Export inplace]
    if _scalar(threshold) != 0:
        raise RuntimeError("threshold: Non-zero threshold in Threshold not supported")
    if _scalar(value) != 0:
        raise RuntimeError("threshold: Non-zero value in Threshold not supported")
    return g.op("Relu", input)


def leaky_relu(g, input, negative_slope, inplace):
    # See Note [Export inplace]
    # TODO: Talk to ONNX about unconditional cast of scalar to float
    return g.op("LeakyRelu", input, alpha_f=_scalar(negative_slope))


def glu(g, input, dim):
    assert input.type().sizes()[dim] % 2 == 0

    first, second = g.op('Split', input, axis_i=dim, outputs=2)
    return g.op('Mul', first, g.op('Sigmoid', second))


def softmax(g, input, dim=None):
    return g.op('Softmax', input, axis_i=dim)


def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        raise RuntimeError("ceil_mode not supported in MaxPool2d")
    if not stride:
        stride = kernel_size
    r = g.op("MaxPool", input,
             kernel_shape_i=_pair(kernel_size),
             pads_i=_pair(padding),
             dilations_i=_pair(dilation),
             strides_i=_pair(stride))
    return r, None


def avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad):
    if ceil_mode:
        raise RuntimeError("ceil_mode not supported in AvgPool2d")
    if not stride:
        stride = kernel_size
    # TODO: What about count_include_pad?!
    return g.op("AveragePool", input,
                kernel_shape_i=_pair(kernel_size),
                strides_i=_pair(stride),
                pads_i=_pair(padding))


def log_softmax(g, input, dim=None):
    return g.op("Log", g.op('Softmax', input, axis_i=dim).setTypeAs(input))
