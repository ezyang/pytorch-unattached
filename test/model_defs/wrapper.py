from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import itertools

import google.protobuf.text_format

import torch.jit

import toffee
from toffee.backend import Caffe2Backend as c2


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


def torch_export(model, x):
    # Enable tracing on the model
    trace, torch_out = torch.jit.record_trace(toC(model), toC(x))
    proto = torch._C._jit_pass_export(trace)
    return proto, torch_out


def caffe2_load(proto, model, x, state_dict=None):

    graph_def = toffee.GraphProto.FromString(proto)
    # TODO: This is a hack; PyTorch should set it
    graph_def.version = toffee.GraphProto().version

    toffee.checker.check_graph(graph_def)

    # Translate the parameters into Caffe2 form
    W = {}
    state_dict_running_vals = []
    bn_running_values = [
        s for s in graph_def.input if "saved_" in s]
    if state_dict is not None:
        # if we have the pre-trained model, use the running mean/var values
        # from it otherwise use the dummy values
        state_dict_running_vals = [
            s for s in state_dict.keys() if "running_" in s]
    else:
        for v in bn_running_values:
            size = int(v.split('_')[-3])
            if "mean" in v:
                W[v] = torch.zeros(size).numpy()
            else:
                W[v] = torch.ones(size).numpy()
    real_inputs = [s for s in graph_def.input if "saved_" not in s]
    for (v1, v2) in zip(bn_running_values, state_dict_running_vals):
        W[v1] = state_dict[v2].cpu().numpy()
    for k, v in zip(real_inputs, itertools.chain(model.parameters(), [x])):
        # On C2 side, we don't run on CUDA yet so convert to CPU memory
        W[k] = v.data.cpu().numpy()

    caffe2_out_workspace = c2.run_graph(
        init_graph=None,
        predict_graph=graph_def,
        inputs=W)
    caffe2_out = list(caffe2_out_workspace.values())[0]
    return caffe2_out
