import torch.cuda
import torch.backends.cudnn as cudnn
from torch.backends.cudnn import check_error
import ctypes
from torch.autograd import Variable


def get_cudnn_mode(mode):
    if mode == 'RNN_RELU':
        return cudnn.CUDNN_RNN_RELU
    elif mode == 'RNN_TANH':
        return cudnn.CUDNN_RNN_TANH
    elif mode == 'LSTM':
        return cudnn.CUDNN_LSTM
    elif mode == 'GRU':
        return cudnn.CUDNN_GRU
    else:
        raise Exception("Unknown mode: {}".format(mode))


class Unserializable(object):

    def __init__(self, inner):
        self.inner = inner

    def get(self):
        return self.inner

    def __getstate__(self):
        # Note: can't return {}, because python2 won't call __setstate__
        # if the value evaluates to False
        return "<unserializable>"

    def __setstate__(self, state):
        self.inner = None


# Needs in fn: dropout, train, dropout_state, dropout_seed
def init_dropout_descriptor(handle, dropout, train, dropout_seed, dropout_state):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if (dropout_desc_name not in dropout_state) or (dropout_state[dropout_desc_name].get() is None):
        dropout_state[dropout_desc_name] = Unserializable(
            cudnn.DropoutDescriptor(handle, dropout_p, dropout_seed)
        )
    dropout_desc = dropout_state[dropout_desc_name].get()
    dropout_desc.set_dropout(dropout_p, dropout_seed)
    return dropout_desc


def get_dropout_state(fn, handle):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = fn.dropout if fn.train else 0
    dropout_desc = fn.dropout_state[dropout_desc_name].get()
    return dropout_desc.state


def forward(fn, input, hx, weight):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
        else:
            cx = None

        handle = cudnn.get_handle()
        dropout_desc = init_dropout_descriptor(handle, fn.dropout, fn.train, fn.dropout_seed, fn.dropout_state)

        # TODO: in an ideal world, we could pass list of list direct
        weight_arr = [Variable(w) for ws in weight for w in ws]
        weight_stride0 = len(weight[0])
        for ws in weight:
            assert len(ws) == weight_stride0

        output, hy, cy, reserve, new_weight_buf = torch._C._VariableFunctions._cudnn_rnn(
            Variable(input), weight_arr, weight_stride0,
            Variable(fn.weight_buf) if fn.weight_buf is not None else None,
            Variable(hx),
            Variable(cx) if cx is not None else None,
            fn.mode, fn.hidden_size, fn.num_layers,
            fn.batch_first, fn.dropout, fn.train, bool(fn.bidirectional),
            fn.batch_sizes if fn.batch_sizes else (),
            Variable(dropout_desc.state) if dropout_desc.state is not None else None)

        # For backwards
        fn.weight_buf = new_weight_buf.data
        fn.reserve = reserve.data

        if cx is not None:
            extra_outs = (hy.data, cy.data)
        else:
            extra_outs = hy.data

        return output.data, extra_outs


def backward_grad(fn, input, hx, weight, output, grad_output, grad_hy):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
            grad_hy, grad_cy = grad_hy
        else:
            cx, grad_cy = None, None

        handle = cudnn.get_handle()
        dropout_desc = init_dropout_descriptor(handle, fn.dropout, fn.train, fn.dropout_seed, fn.dropout_state)
        dx, dhx, dcx = torch._C._VariableFunctions._cudnn_rnn_backward_grad(
            Variable(input), Variable(fn.weight_buf), Variable(hx), Variable(cx) if cx is not None else None,
            Variable(output), Variable(grad_output), Variable(grad_hy), Variable(grad_cy) if grad_cy is not None else None,
            fn.mode, fn.hidden_size, fn.num_layers,
            fn.batch_first, fn.dropout, fn.train, bool(fn.bidirectional),
            fn.batch_sizes if fn.batch_sizes else (),
            Variable(dropout_desc.state) if dropout_desc.state is not None else None,
            Variable(fn.reserve))

        if cx is not None:
            return dx.data, (dhx.data, dcx.data)
        else:
            return dx.data, dhx.data


def backward_weight(fn, input, hx, output, weight):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
        else:
            cx = None

        handle = cudnn.get_handle()
        weight_arr = [Variable(w) for ws in weight for w in ws]
        weight_stride0 = len(weight[0])
        dropout_desc = init_dropout_descriptor(handle, fn.dropout, fn.train, fn.dropout_seed, fn.dropout_state)
        dw = torch._C._VariableFunctions._cudnn_rnn_backward_weight(
            Variable(input), weight_arr, weight_stride0, Variable(fn.weight_buf), Variable(hx), Variable(cx) if cx is not None else None,
            Variable(output),
            fn.mode, fn.hidden_size, fn.num_layers,
            fn.batch_first, fn.dropout, fn.train, bool(fn.bidirectional),
            fn.batch_sizes if fn.batch_sizes else (),
            Variable(dropout_desc.state) if dropout_desc.state is not None else None,
            Variable(fn.reserve))

        return [list(map(lambda x: x.data, dw[i:i + weight_stride0])) for i in range(0, len(dw), weight_stride0)]
