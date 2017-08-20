from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import itertools

try:
    import caffe2
except ImportError:
    print('Cannot import caffe2, hence caffe2-torch test will not run.')
    sys.exit(0)

try:
    import torch
except ImportError:
    print('Cannot import torch, hence caffe2-torch test will not run.')
    sys.exit(0)

import torch.nn as nn
import torch.jit
from torch.autograd import Variable, Function

import toffee
from toffee.backend import Caffe2Backend as c2

import google.protobuf.text_format

import unittest


class TestCaffe2Backend(unittest.TestCase):
    def test_alexnet(self):
        # PyTorch AlexNet definition
        class AlexNet(nn.Module):

            def __init__(self, num_classes=1000):
                super(AlexNet, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.classifier = nn.Sequential(
                    # nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=False),
                    # nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=False),
                    nn.Linear(4096, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), 256 * 6 * 6)
                x = self.classifier(x)
                return x

        # Randomly but deterministically initialize the parameters (I hope!)
        torch.manual_seed(0)
        underlying_model = AlexNet()
        # underlying_model.train(False)

        # Random (deterministic) input
        x = Variable(torch.randn(10, 3, 224, 224), requires_grad=True)

        # Enable tracing on the model
        trace, torch_out = torch.jit.record_trace(underlying_model, x)
        proto = torch._C._jit_pass_export(trace)

        graph_def = toffee.GraphProto()
        google.protobuf.text_format.Merge(proto, graph_def)

        # TODO: This is a hack; PyTorch should set it
        graph_def.version = toffee.GraphProto().version

        toffee.checker.check_graph(graph_def)

        # Translate the parameters into Caffe2 form
        W = {}
        batch_norm_running_values = [
            s for s in graph_def.input if "saved" in s]
        real_inputs = [s for s in graph_def.input if "saved" not in s]
        for v in batch_norm_running_values:
            # print(v)
            size = int(v.split('_')[-2])
            if "mean" in v:
                W[v] = torch.zeros(size).numpy()
            else:
                W[v] = torch.ones(size).numpy()
        for k, v in zip(real_inputs, itertools.chain(underlying_model.parameters(), [x])):
            W[k] = v.data.numpy()

        caffe2_out_workspace = c2.run_graph(
            init_graph=None,
            predict_graph=graph_def,
            inputs=W)
        caffe2_out = list(caffe2_out_workspace.values())[0]
        np.testing.assert_almost_equal(
            torch_out.data.numpy(), caffe2_out, decimal=3)


if __name__ == '__main__':
    unittest.main()
