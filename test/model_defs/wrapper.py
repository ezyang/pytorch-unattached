from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import itertools

import google.protobuf.text_format

import torch.jit
from torch.autograd import Variable

import toffee
from toffee.backend import Caffe2Backend as c2

import timeit

try:
    import caffe2
except ImportError:
    print('Cannot import caffe2, hence caffe2-torch test will not run.')
    sys.exit(0)

torch.set_default_tensor_type('torch.FloatTensor')
try:
    import torch
except ImportError:
    print('Cannot import torch, hence caffe2-torch test will not run.')
    sys.exit(0)

if torch.cuda.is_available():
    def toC(x):
        return x.cuda()
else:
    def toC(x):
        return x

proto_init = False


def torch_export(model, x):

    # Enable tracing on the model
    ts1 = timeit.default_timer()
    trace, torch_out = torch.jit.record_trace(toC(model), toC(x))
    ts2 = timeit.default_timer()
    print('\n[time] {} spent {:.2f} seconds.'.format('pytorch_execution', ts2 - ts1))
    if proto_init is False:
        proto = torch._C._jit_pass_export(trace)
    else:
        proto = torch._C._jit_pass_export(trace, model.state_dict().values())
    ts3 = timeit.default_timer()
    print('[time] {} spent {:.2f} seconds.'.format('export_proto', ts3 - ts2))
    print('[size] proto is {} bytes ({:.2f} MB), and speed is {:.2f} MB/s.'.format(
          len(proto), len(proto) / 1024.0 / 1024,
          len(proto) / 1024.0 / 1024 / (ts3 - ts2)))
    return proto, torch_out


def caffe2_load(proto, model, x, state_dict=None, use_gpu=False):

    ts4 = timeit.default_timer()
    graph_def = toffee.GraphProto.FromString(proto)
    # TODO: This is a hack; PyTorch should set it
    graph_def.version = toffee.GraphProto().version

    ts5 = timeit.default_timer()
    print('[time] {} spent {:.2f} seconds.'.format('load_proto', ts5 - ts4))

    toffee.checker.check_graph(graph_def)

    ts6 = timeit.default_timer()
    print('[time] {} spent {:.2f} seconds.'.format('check_graph', ts6 - ts5))

    # Translate the parameters into Caffe2 form
    W = {}
    if proto_init is False:
        if state_dict:
            parameters = []
            # Passed in state_dict may have a different order.  Make
            # sure our order is consistent with the model's order.
            # TODO: Even better: keyword arguments!
            for k in model.state_dict():
                parameters.append(state_dict[k])
        else:
            parameters = model.state_dict().values()
        for k, v in zip(graph_def.input, itertools.chain(parameters, [x])):
            # On C2 side, we don't run on CUDA yet so convert to CPU memory
            if isinstance(v, Variable):
                W[k] = v.data.cpu().numpy()
            else:
                W[k] = v.cpu().numpy()
    else:
        W[graph_def.input[-1]] = x.data.cpu().numpy()

    ts7 = timeit.default_timer()
    print('[time] {} spent {:.2f} seconds.'.format('translate_to_caffe2', ts7 - ts6))

    caffe2_out_workspace = c2.run_graph(
        init_graph=None,
        predict_graph=graph_def,
        inputs=W,
        use_gpu=use_gpu)
    caffe2_out = list(caffe2_out_workspace.values())[0]

    ts8 = timeit.default_timer()
    print('[time] {} spent {:.2f} seconds.'.format('caffe2_execution', ts8 - ts7))

    return caffe2_out
