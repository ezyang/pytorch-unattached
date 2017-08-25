from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import unittest

import torch.jit
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

from vgg import make_vgg16, make_vgg19, make_vgg16_bn, make_vgg19_bn
from alexnet import AlexNet
from resnet import Bottleneck, ResNet
from inception import Inception3
from squeezenet import SqueezeNet
from densenet import DenseNet
import dcgan
from wrapper import torch_export, caffe2_load

skip = unittest.skip

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
    embed_params = False
    def run_model_test(self, model, train, batch_size, state_dict=None,
                       input=None):
        torch.manual_seed(0)
        model.train(train)

        if state_dict is not None:
            model.load_state_dict(state_dict)

        # Either user specified input or random (deterministic) input
        if input is None:
            input = Variable(torch.randn(batch_size, 3, 224, 224),
                             requires_grad=True)
        toffeeir, torch_out = torch_export(model, input, self.embed_params)
        caffe2_out = caffe2_load(toffeeir, model, input, state_dict, self.embed_params)
        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(),
                                       caffe2_out, decimal=3)

    def test_alexnet(self):
        alexnet = AlexNet()
        state_dict = model_zoo.load_url(model_urls['alexnet'])
        self.run_model_test(alexnet, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_dcgan(self):
        netD = dcgan._netD(1)
        netD.apply(dcgan.weights_init)
        input = Variable(torch.Tensor(BATCH_SIZE, 3, dcgan.imgsz, dcgan.imgsz))
        self.run_model_test(netD, train=False, batch_size=BATCH_SIZE,
                            input=input)

        netG = dcgan._netG(1)
        netG.apply(dcgan.weights_init)
        noise = Variable(torch.Tensor(BATCH_SIZE, dcgan.nz, 1, 1).normal_(0, 1))
        self.run_model_test(netG, train=False, batch_size=BATCH_SIZE,
                            input=noise, state_dict=None)

    @unittest.skipIf(not torch.cuda.is_available(),
                     "model on net has cuda in it, awaiting fix")
    def test_densenet(self):
        densenet121 = DenseNet(num_init_features=64, growth_rate=32,
                               block_config=(6, 12, 24, 16), inplace=False)
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        self.run_model_test(densenet121, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    # @skip("doesn't match exactly...")
    def test_inception(self):
        torch.manual_seed(0)
        inception = Inception3(aux_logits=True, transform_input=False)
        # state_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        state_dict = None
        input = Variable(torch.randn(BATCH_SIZE, 3, 299, 299),
                         requires_grad=True)
        self.run_model_test(inception, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict, input=input)

    def test_resnet(self):
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], inplace=False)
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        self.run_model_test(resnet50, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_squeezenet(self):
        sqnet_v1_1 = SqueezeNet(version=1.1, inplace=False)
        state_dict = model_zoo.load_url(model_urls['squeezenet1_1'])
        self.run_model_test(sqnet_v1_1, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_vgg16(self):
        vgg16 = make_vgg16()
        state_dict = model_zoo.load_url(model_urls['vgg16'])
        self.run_model_test(vgg16, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    @skip("disable to run tests faster...")
    def test_vgg16_bn(self):
        underlying_model = make_vgg16_bn()
        self.run_model_test(underlying_model, train=False,
                            batch_size=BATCH_SIZE)

    @skip("disable to run tests faster...")
    def test_vgg19(self):
        vgg19 = make_vgg19()
        state_dict = model_zoo.load_url(model_urls['vgg19'])
        self.run_model_test(vgg19, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    @skip("disable to run tests faster...")
    def test_vgg19_bn(self):
        underlying_model = make_vgg19_bn()
        self.run_model_test(underlying_model, train=False,
                            batch_size=BATCH_SIZE)

# add the same test suite as above, but switch embed_params=False
# to embed_params=True
TestCaffe2BackendEmbed = type(str("TestCaffe2BackendEmbed"),
                              (unittest.TestCase,),
                              dict(TestCaffe2Backend.__dict__, embed_params=True))

if __name__ == '__main__':
    unittest.main()
