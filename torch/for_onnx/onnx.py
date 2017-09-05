# We are going to import standar onnx package and this module is also named
# onnx, so we do an absolute_import to avoid conflicts that might arise
from __future__ import absolute_import

import toffee as onnx
from toffee.backend import Caffe2Backend as c2


def import_model(proto, input, workspace=None, use_gpu=True):
    graph_def = onnx.GraphProto.FromString(proto)
    onnx.checker.check_graph(graph_def)

    if workspace is None:
        workspace = {}
    workspace[graph_def.input[-1]] = input

    caffe2_out_workspace = c2.run_graph(
        init_graph=None,
        predict_graph=graph_def,
        inputs=workspace,
        use_gpu=use_gpu)
    caffe2_out = caffe2_out_workspace[0]
    return caffe2_out
