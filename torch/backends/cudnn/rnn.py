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
def init_dropout_descriptor(fn, handle):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = fn.dropout if fn.train else 0
    if (dropout_desc_name not in fn.dropout_state) or (fn.dropout_state[dropout_desc_name].get() is None):
        fn.dropout_state[dropout_desc_name] = Unserializable(
            cudnn.DropoutDescriptor(handle, dropout_p, fn.dropout_seed)
        )
    dropout_desc = fn.dropout_state[dropout_desc_name].get()
    dropout_desc.set_dropout(dropout_p, fn.dropout_seed)
    return dropout_desc


def get_dropout_state(fn, handle):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = fn.dropout if fn.train else 0
    dropout_desc = fn.dropout_state[dropout_desc_name].get()
    return dropout_desc.state


def init_rnn_descriptor(fn, handle):
    dropout_desc_name = 'desc_' + str(torch.cuda.current_device())
    dropout_p = fn.dropout if fn.train else 0
    if (dropout_desc_name not in fn.dropout_state) or (fn.dropout_state[dropout_desc_name].get() is None):
        fn.dropout_state[dropout_desc_name] = Unserializable(
            cudnn.DropoutDescriptor(handle, dropout_p, fn.dropout_seed)
        )
    dropout_desc = fn.dropout_state[dropout_desc_name].get()
    dropout_desc.set_dropout(dropout_p, fn.dropout_seed)
    return cudnn.RNNDescriptor(
        handle,
        fn.hidden_size,
        fn.num_layers,
        dropout_desc,
        fn.input_mode,
        fn.bidirectional,
        fn.mode,
        fn.datatype
    )


def init_weight_descriptor(fn, weight):
    w_desc = cudnn.FilterDescriptor()
    w_view = weight.view(-1, 1, 1)  # seems that filters require >=3 dimensions
    w_desc.set(w_view)
    return w_desc


def get_num_weights(handle, rnn_desc, x_desc, datatype):
    weight_size = ctypes.c_long()
    check_error(cudnn.lib.cudnnGetRNNParamsSize(
        handle,
        rnn_desc,
        x_desc,
        ctypes.byref(weight_size),
        datatype
    ))
    elem_size = cudnn._sizeofmap[datatype]
    assert weight_size.value % elem_size == 0
    return weight_size.value // elem_size


def get_parameters(fn, handle, weight_buf):
    """Returns weight and bias tensors for each layer of the RNN. These tensors
    are views on the underlying weight buffer allocated by CuDNN.

    Note: for LSTM and GRU, which have multiple parameters of each type (4 and 3, respectively),
          these parameters are concatenated along the first dimension.
          These parameters are returned in a consistent order by CuDNN:
              (reset, forget, cell, outut) for LSTM
              (reset, input, new) for GRU
    Args:
        fn: The RNN function object holding the RNN state
        handle: a CuDNN handle
        weight_buf: a 1D tensor containing the CuDNN-allocated weight (or grad_weight) buffer
    Returns:
        parameters: [(weight_ih, weight_hh, bias_ih, bias_hh)*], with length equal to the num_layers.
    """

    cudnn_methods = [
        cudnn.lib.cudnnGetRNNLinLayerMatrixParams,
        cudnn.lib.cudnnGetRNNLinLayerBiasParams
    ]

    params = []
    num_linear_layers = _num_linear_layers(fn)
    num_layers = fn.num_directions * fn.num_layers
    for layer in range(num_layers):
        layer_params = []
        for cudnn_method in cudnn_methods:
            for linear_id in range(num_linear_layers):
                lin_layer_mat_desc = cudnn.FilterDescriptor()
                matrix_pointer = ctypes.c_void_p()
                check_error(cudnn_method(
                    handle,
                    fn.rnn_desc,
                    layer,
                    fn.x_descs[0],
                    fn.w_desc,
                    ctypes.c_void_p(weight_buf.data_ptr()),
                    linear_id,
                    lin_layer_mat_desc,
                    ctypes.byref(matrix_pointer)))

                data_type = ctypes.c_int()
                format = ctypes.c_int()
                nb_dims = ctypes.c_int()
                min_dim = 3
                filter_dim_a = torch.IntTensor(min_dim)
                check_error(cudnn.lib.cudnnGetFilterNdDescriptor(
                    lin_layer_mat_desc,
                    min_dim,
                    ctypes.byref(data_type),
                    ctypes.byref(format),
                    ctypes.byref(nb_dims),
                    ctypes.c_void_p(filter_dim_a.data_ptr())))

                assert nb_dims.value <= min_dim
                filter_dim_a = filter_dim_a[:nb_dims.value]
                elem_size = cudnn._sizeofmap[fn.datatype]
                offset_bytes = (matrix_pointer.value - weight_buf.data_ptr())
                assert offset_bytes % elem_size == 0
                offset = offset_bytes // elem_size

                # for all the RNN types provided by CUDNN, all the ih weights
                # are the same size and are allocated in a contiguous chunk
                # (same for the hh weights, and the ih and hh biases).
                # Since we're storing all the weights in a single tensor anyway,
                # might as well merge the CUDNN ones into a single tensor as well
                if linear_id == 0 or linear_id == num_linear_layers / 2:
                    assert filter_dim_a.prod() == filter_dim_a[0]
                    size = (filter_dim_a[0] * num_linear_layers // 2, filter_dim_a[2])
                    param = fn.weight_buf.new().set_(
                        weight_buf.storage(), offset, size)
                    layer_params.append(param)
                else:
                    assert cur_offset == offset

                cur_offset = offset + filter_dim_a[0]

        params.append(layer_params)

    return params


def _copyParams(params_from, params_to):
    assert len(params_from) == len(params_to)
    for layer_params_from, layer_params_to in zip(params_from, params_to):
        # NOTE: these lists have all weights before all biases, so if the layer doesn't
        # use biases, zip will terminate once layer_params_from ends and ignore them.
        for param_from, param_to in zip(layer_params_from, layer_params_to):
            assert param_from.type() == param_to.type()
            param_to.copy_(param_from, broadcast=False)


def forward(fn, input, hx, weight):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
        else:
            cx = None

        handle = cudnn.get_handle()
        dropout_desc = init_dropout_descriptor(fn, handle)

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
        dropout_desc = init_dropout_descriptor(fn, handle)
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


def _num_linear_layers(fn):
    if fn.mode == cudnn.CUDNN_LSTM:
        return 8
    elif fn.mode == cudnn.CUDNN_GRU:
        return 6
    elif fn.mode == cudnn.CUDNN_RNN_RELU:
        return 2
    elif fn.mode == cudnn.CUDNN_RNN_TANH:
        return 2
    else:
        raise RuntimeError('Unknown mode: {}'.format(fn.mode))


def backward_weight(fn, input, hx, output, weight):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
        else:
            cx = None

        handle = cudnn.get_handle()
        weight_arr = [Variable(w) for ws in weight for w in ws]
        weight_stride0 = len(weight[0])
        dropout_desc = init_dropout_descriptor(fn, handle)
        dw = torch._C._VariableFunctions._cudnn_rnn_backward_weight(
            Variable(input), weight_arr, weight_stride0, Variable(fn.weight_buf), Variable(hx), Variable(cx) if cx is not None else None,
            Variable(output),
            fn.mode, fn.hidden_size, fn.num_layers,
            fn.batch_first, fn.dropout, fn.train, bool(fn.bidirectional),
            fn.batch_sizes if fn.batch_sizes else (),
            Variable(dropout_desc.state) if dropout_desc.state is not None else None,
            Variable(fn.reserve))

        return [list(map(lambda x: x.data, dw[i:i + weight_stride0])) for i in range(0, len(dw), weight_stride0)]
