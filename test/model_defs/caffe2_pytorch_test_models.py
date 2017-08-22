from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import itertools

from vgg import *
from dcgan import _netD, _netG, weights_init, bsz, imgsz, nz, ngf, ndf, nc
from alexnet import AlexNet
from resnet import Bottleneck, ResNet
from inception import Inception3
from squeezenet import SqueezeNet
from densenet import DenseNet


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


BATCH_SIZE = 2


class TestCaffe2Backend(unittest.TestCase):
    def run_model_test(self, model, train, batch_size, x=None):
        torch.manual_seed(0)
        model.train(train)

        # Random (deterministic) input
        if x is None:
            x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)

        # Enable tracing on the model
        trace, torch_out = torch.jit.record_trace(model, x)
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
            size = int(v.split('_')[-2])
            if "mean" in v:
                W[v] = torch.zeros(size).numpy()
            else:
                W[v] = torch.ones(size).numpy()
        for k, v in zip(real_inputs, itertools.chain(model.parameters(), [x])):
            W[k] = v.data.numpy()

        caffe2_out_workspace = c2.run_graph(
            init_graph=None,
            predict_graph=graph_def,
            inputs=W)
        caffe2_out = list(caffe2_out_workspace.values())[0]
        np.testing.assert_almost_equal(torch_out.data.numpy(), caffe2_out,
                                       decimal=3)
        print('Finished testing model.')

    def test_alexnet(self):
        alexnet = AlexNet()
        self.run_model_test(alexnet, train=False,
                            batch_size=BATCH_SIZE)

    def test_vgg(self):
        models = [make_vgg16(), make_vgg19()]
        for underlying_model in models:
            self.run_model_test(underlying_model, train=False,
                                batch_size=BATCH_SIZE)

    def test_vgg_bn(self):
        models = [make_vgg16_bn(), make_vgg19_bn()]
        for underlying_model in models:
            self.run_model_test(underlying_model, train=False,
                                batch_size=BATCH_SIZE)

    def test_resnet(self):
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], inplace=False)
        self.run_model_test(resnet50, train=False, batch_size=BATCH_SIZE)

    def test_squeezenet(self):
        sqnet_v1_1 = SqueezeNet(version=1.1, inplace=False)
        self.run_model_test(sqnet_v1_1, train=False, batch_size=BATCH_SIZE)

    def test_densenet(self):
        densenet121 = DenseNet(num_init_features=64, growth_rate=32,
                               block_config=(6, 12, 24, 16), inplace=False)
        self.run_model_test(densenet121, train=False, batch_size=BATCH_SIZE)

    def test_inception(self):
        inception = Inception3(aux_logits=False, inplace=False)
        self.run_model_test(inception, train=False, batch_size=BATCH_SIZE)

    def test_dcgan(self):
        netD = _netD(1)
        netD.apply(weights_init)
        input = Variable(torch.Tensor(bsz, 3, imgsz, imgsz))
        self.run_model_test(netD, False, BATCH_SIZE, input)

        netG = _netG(1)
        netG.apply(weights_init)
        noise = Variable(torch.Tensor(bsz, nz, 1, 1).normal_(0, 1))
        self.run_model_test(netG, False, BATCH_SIZE, noise)

if __name__ == '__main__':
    unittest.main()
