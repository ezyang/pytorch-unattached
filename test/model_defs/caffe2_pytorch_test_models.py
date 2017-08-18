from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import itertools
from vgg import *
from alexnet import *
from resnet import *
from mnist import *
from squeezenet import *
from inception import *

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

# import torch.nn as nn
import torch.jit
from torch.autograd import Variable

import toffee
from toffee.backend import Caffe2Backend as c2

import google.protobuf.text_format

import unittest


BATCH_SIZE = 10

# BN -> train mode
# Dropout -> test mode

# AlexNet - done
# VGG16 - done [test mode due to Dropout]
# VGG16-BN - done [train mode but remove Dropout()]
# VGG19 - done [test mode due to Dropout]
# VGG19-BN - done [train mode but remove Dropout()]
# SqueezeNet - done [test mode due to Dropout]
# ResNet50 - done [train mode]
# Inception3
# DCGAN
#   - netG
#   - netD
# DenseNet121


class TestCaffe2Backend(unittest.TestCase):
    def test_models(self):
        # TODO: we run BN in train mode only right now [hard-coded is_test=0]
        # in primspec. VGG model has Dropout() in train mode so it is recommended
        # to run it in test mode. VGG-BN model should run in train model only
        # due to BN but this contradicts Dropout()

        # models = [make_vgg16_bn(), make_vgg19_bn()]
        # models = [make_vgg16(), make_vgg19(), AlexNet()]
        # models = [resnet50]
        # resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], inplace=False)
        sqnet_v1_1 = SqueezeNet(version=1.1, inplace=False)
        models = [sqnet_v1_1]
        for underlying_model in models:
            torch.manual_seed(0)
            underlying_model.train(False)

            # Random (deterministic) input
            # x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28), requires_grad=True)
            x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224), requires_grad=True)

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
            batch_norm_running_values = [s for s in graph_def.input if "saved" in s]
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
            np.testing.assert_almost_equal(torch_out.data.numpy(), caffe2_out, decimal=3)


if __name__ == '__main__':
    unittest.main()
