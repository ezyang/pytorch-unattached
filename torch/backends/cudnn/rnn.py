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


def _input_size(fn, input):
    if fn.batch_sizes is not None:
        return (input.size(0), fn.input_size)
    else:
        return (fn.seq_length, fn.mini_batch, fn.input_size)


def _hidden_size(fn):
    return (fn.num_layers * fn.num_directions, fn.mini_batch, fn.hidden_size)


def _output_size(fn, input):
    if fn.batch_sizes is not None:
        return (input.size(0), fn.hidden_size * fn.num_directions)
    else:
        return (fn.seq_length, fn.mini_batch, fn.hidden_size * fn.num_directions)


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


def forward(fn, input, hx, weight, out_output, out_hy):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
            out_hy, out_cy = out_hy
        else:
            cx, out_cy = None, None

        handle = cudnn.get_handle()

        # blah blah backwards
        lib = cudnn.lib
        fn.datatype = cudnn._typemap[input.type()]
        orig_input = input
        is_input_packed = fn.batch_sizes is not None
        if fn.batch_first and not is_input_packed:
            input = input.transpose(0, 1)
        if is_input_packed:
            fn.seq_length = len(fn.batch_sizes)
            fn.mini_batch = fn.batch_sizes[0]
            fn.input_size = input.size(-1)
        else:
            fn.seq_length, fn.mini_batch, fn.input_size = input.size()

        hidden_size = _hidden_size(fn)
        output_size = _output_size(fn, input)

        x = input.contiguous()
        out_output.resize_(*output_size)
        out_hy.resize_(*hidden_size)
        if out_cy is not None:
            out_cy.resize_(*hidden_size)
        y = out_output

        fn.rnn_desc = init_rnn_descriptor(fn, handle)
        if is_input_packed:
            fn.x_descs = cudnn.descriptor_sequence(x, fn.batch_sizes)
        else:
            fn.x_descs = cudnn.descriptor(x[0], fn.seq_length)

        # create the weight buffer and copy the weights into it
        if fn.weight_buf is None:
            num_weights = get_num_weights(
                handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
            fn.weight_buf = x.new(num_weights)
            fn.w_desc = init_weight_descriptor(fn, fn.weight_buf)
            # this zero might not seem necessary, but it is in the case
            # where biases are disabled; then they won't be copied and must be zero'd.
            # Alternatively, _copyParams could be written more carefully.
            fn.weight_buf.zero_()
            params = get_parameters(fn, handle, fn.weight_buf)
            _copyParams(weight, params)
        else:
            fn.w_desc = init_weight_descriptor(fn, fn.weight_buf)

        workspace_size = ctypes.c_long()
        check_error(lib.cudnnGetRNNWorkspaceSize(
            handle,
            fn.rnn_desc,
            fn.seq_length,
            fn.x_descs,
            ctypes.byref(workspace_size)
        ))
        fn.workspace_size = workspace_size.value

        # OK actual stuff
        #dropout_desc = init_dropout_descriptor(fn, handle)
        dropout_state = get_dropout_state(fn, handle)
        # Variable massaging
        output, hy, cy, reserve = torch._C._VariableFunctions._cudnn_rnn(
            Variable(orig_input), Variable(fn.weight_buf), Variable(hx), Variable(cx) if cx is not None else None, fn.mode, fn.hidden_size, fn.num_layers,
            fn.batch_first, fn.dropout, fn.train, bool(fn.bidirectional),
            fn.batch_sizes if fn.batch_sizes else (),
            Variable(dropout_state) if dropout_state is not None else None)

        # WOAAAAAH DUUUUDE
        if fn.batch_first and not is_input_packed:
            out_output.transpose_(0, 1)
        out_output.resize_as_(output.data)
        out_output.copy_(output.data)
        out_hy.resize_as_(hy.data)
        out_hy.copy_(hy.data)
        if out_cy is not None:
            out_cy.resize_as_(cy.data)
            out_cy.copy_(cy.data)
        fn.reserve = reserve.data


def backward_grad(fn, input, hx, weight, output, grad_output, grad_hy, grad_input, grad_hx):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
            grad_hx, grad_cx = grad_hx
            grad_hy, grad_cy = grad_hy
        else:
            cx, grad_cx, grad_cy = None, None, None

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

        grad_input.resize_as_(dx.data)
        grad_input.copy_(dx.data)
        grad_hx.resize_as_(dhx.data)
        grad_hx.copy_(dhx.data)
        if grad_cx is not None:
            grad_cx.resize_as_(dcx.data)
            grad_cx.copy_(dcx.data)


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


def backward_weight(fn, input, hx, output, weight, grad_weight):
    with torch.cuda.device_of(input):
        if fn.mode == cudnn.CUDNN_LSTM:
            hx, cx = hx
        else:
            cx = None

        handle = cudnn.get_handle()
        dropout_desc = init_dropout_descriptor(fn, handle)
        dw = torch._C._VariableFunctions._cudnn_rnn_backward_weight(
            Variable(input), Variable(fn.weight_buf), Variable(hx), Variable(cx) if cx is not None else None,
            Variable(output),
            fn.mode, fn.hidden_size, fn.num_layers,
            fn.batch_first, fn.dropout, fn.train, bool(fn.bidirectional),
            fn.batch_sizes if fn.batch_sizes else (),
            Variable(dropout_desc.state) if dropout_desc.state is not None else None,
            Variable(fn.reserve))

        # copy the weights from the weight_buf into grad_weight
        grad_params = get_parameters(fn, handle, dw)
        _copyParams(grad_params, grad_weight)
        return grad_weight
