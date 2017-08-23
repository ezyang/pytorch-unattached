from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import itertools

from vgg import *
from dcgan import *
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

import torch.jit
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import toffee
from toffee.backend import Caffe2Backend as c2

import google.protobuf.text_format

import unittest


BATCH_SIZE = 2

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class TestCaffe2Backend(unittest.TestCase):
    def run_model_test(self, model, train, batch_size, state_dict=None):
        torch.manual_seed(0)
        model.train(train)

        # Random (deterministic) input
        x = Variable(torch.randn(batch_size, 3, 224, 224), requires_grad=True)

        # Enable tracing on the model
        trace, torch_out = torch.jit.record_trace(model, x)
        proto = torch._C._jit_pass_export(trace)
        # print(proto)

        graph_def = toffee.GraphProto()
        google.protobuf.text_format.Merge(proto, graph_def)

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
                size = int(v.split('_')[-2])
                if "mean" in v:
                    W[v] = torch.zeros(size).numpy()
                else:
                    W[v] = torch.ones(size).numpy()
        real_inputs = [s for s in graph_def.input if "saved_" not in s]
        for (v1, v2) in zip(bn_running_values, state_dict_running_vals):
            print('v1: {} v2: {}'.format(v1, v2))
            W[v1] = state_dict[v2].cpu().numpy()
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
        print('testing AlexNet model')
        alexnet = AlexNet()
        alexnet.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        self.run_model_test(alexnet, train=False,
                            batch_size=BATCH_SIZE)

    def test_vgg(self):
        print('testing VGG-16/19 models without BN')
        vgg16 = make_vgg16()
        vgg16.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        vgg19 = make_vgg19()
        vgg19.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
        models = [vgg16, vgg19]

        for underlying_model in models:
            self.run_model_test(underlying_model, train=False,
                                batch_size=BATCH_SIZE)

    def test_vgg_bn(self):
        print('testing VGG-16/19 models with BN')
        models = [make_vgg16_bn(), make_vgg19_bn()]
        for underlying_model in models:
            self.run_model_test(underlying_model, train=False,
                                batch_size=BATCH_SIZE)

    def test_resnet(self):
        print('testing ResNet50 model')
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], inplace=False)
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        resnet50.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        self.run_model_test(resnet50, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_squeezenet(self):
        print('testing SqueezeNet1.1 model')
        sqnet_v1_1 = SqueezeNet(version=1.1, inplace=False)
        state_dict = model_zoo.load_url(model_urls['squeezenet1_1'])
        self.run_model_test(sqnet_v1_1, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_densenet(self):
        print('testing DenseNet121 model')
        densenet121 = DenseNet(num_init_features=64, growth_rate=32,
                               block_config=(6, 12, 24, 16), inplace=False)
        # TODO: debug densenet for pretrained weights
        # state_dict = model_zoo.load_url(model_urls['densenet121'])
        self.run_model_test(densenet121, train=False, batch_size=BATCH_SIZE,
                            state_dict=None)

    def test_inception(self):
        print('testing Inception3 model')
        inception = Inception3(aux_logits=False, inplace=False)
        state_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        self.run_model_test(inception, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)


if __name__ == '__main__':
    unittest.main()
