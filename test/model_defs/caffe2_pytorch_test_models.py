from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import wraps
import numpy as np
import sys
import unittest

import torch.toffee
from torch import nn
from torch.for_toffee.toffee import import_model
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from debug_embed_params import test_embed_params
import io

# Import various models for testing
from vgg import make_vgg16, make_vgg19, make_vgg16_bn, make_vgg19_bn
from alexnet import AlexNet
from resnet import Bottleneck, ResNet
from inception import Inception3
from squeezenet import SqueezeNet
from densenet import DenseNet
from super_resolution import SuperResolutionNet
import dcgan

skip = unittest.skip


def do_export(model, inputs, *args, **kwargs):
    f = io.BytesIO()
    out = torch.toffee.export(model, inputs, f, *args, **kwargs)
    return f.getvalue(), out


def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            if 'Lapack library not found' in e.args[0]:
                raise unittest.SkipTest('Compiled without Lapack')
            raise
    return wrapper


torch.set_default_tensor_type('torch.FloatTensor')
try:
    import torch
except ImportError:
    print('Cannot import torch, hence caffe2-torch test will not run.')
    sys.exit(0)


BATCH_SIZE = 2

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'dcgan_b': 'https://s3.amazonaws.com/pytorch/test_data/export/netG_bedroom_epoch_1-0649e76b.pth',
    'dcgan_f': 'https://s3.amazonaws.com/pytorch/test_data/export/netG_faces_epoch_49-d86035a6.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-d66d3027.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'super_resolution': 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class TestCaffe2Backend(unittest.TestCase):
    embed_params = False

    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)

    def convert_cuda(self, model, input):
        cuda_model = model.cuda()
        cuda_input = input.cuda()
        return cuda_model, cuda_input

    def run_debug_test(self, model, train, batch_size, state_dict=None,
                       input=None, use_gpu=True):
        """
        # TODO: remove this from the final release version
        This test is for our debugging only for the case where
        embed_params=False
        """
        model.train(train)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        # Either user specified input or random (deterministic) input
        if input is None:
            input = Variable(torch.randn(batch_size, 3, 224, 224),
                             requires_grad=True)
        if use_gpu:
            model, input = self.convert_cuda(model, input)

        toffeeir, torch_out = do_export(model, input, export_params=self.embed_params)
        caffe2_out = test_embed_params(toffeeir, model, input, state_dict,
                                       use_gpu=use_gpu)
        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(),
                                       caffe2_out, decimal=3)

    def run_actual_test(self, model, train, batch_size, state_dict=None,
                        input=None, use_gpu=True):
        """
        This is what the user facing version will look like
        """
        # set the training/test mode for the model
        model.train(train)
        # use the pre-trained model params if available
        if state_dict is not None:
            model.load_state_dict(state_dict)

        # Either user specified input or random (deterministic) input
        if input is None:
            input = Variable(torch.randn(batch_size, 3, 224, 224),
                             requires_grad=True)
        # Convert the model to ToffeeIR and run model in pytorch
        if use_gpu:
            model, input = self.convert_cuda(model, input)

        toffeeir, torch_out = do_export(model, input, export_params=self.embed_params)

        input = input.data.cpu().numpy()
        # Pass the ToffeeIR and input to load and run in caffe2
        caffe2_out = import_model(toffeeir, input, use_gpu=use_gpu)

        # Verify Pytorch and Caffe2 produce almost same outputs upto
        # certain decimal places
        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(),
                                       caffe2_out, decimal=3)

    def run_model_test(self, model, train, batch_size, state_dict=None,
                       input=None, use_gpu=True):
        use_gpu_ = torch.cuda.is_available() and use_gpu
        if self.embed_params:
            self.run_actual_test(model, train, batch_size, state_dict, input,
                                 use_gpu=use_gpu_)
        else:
            self.run_debug_test(model, train, batch_size, state_dict, input,
                                use_gpu=use_gpu_)

    def test_linear(self):
        model = nn.Linear(1, 1)
        input = Variable(torch.randn(1, 1), requires_grad=True)
        toffeeir, torch_out = do_export(model, input, export_params=False)
        caffe2_out = test_embed_params(toffeeir, model, input)
        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(),
                                       caffe2_out, decimal=3)

    def test_alexnet(self):
        alexnet = AlexNet()
        state_dict = model_zoo.load_url(model_urls['alexnet'])
        self.run_model_test(alexnet, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_dcgan(self):
        # dcgan is flaky on some seeds, see:
        # https://github.com/ProjectToffee/ToffeeIR/pull/70
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)

        netD = dcgan._netD(1)
        netD.apply(dcgan.weights_init)
        input = Variable(torch.Tensor(BATCH_SIZE, 3, dcgan.imgsz, dcgan.imgsz))
        self.run_model_test(netD, train=False, batch_size=BATCH_SIZE,
                            input=input)

        netG = dcgan._netG(1)
        netG.apply(dcgan.weights_init)
        state_dict = model_zoo.load_url(model_urls['dcgan_b'])
        # state_dict = model_zoo.load_url(model_urls['dcgan_f'])
        noise = Variable(
            torch.Tensor(BATCH_SIZE, dcgan.nz, 1, 1).normal_(0, 1))
        self.run_model_test(netG, train=False, batch_size=BATCH_SIZE,
                            input=noise, state_dict=state_dict)

    @unittest.skipIf(not torch.cuda.is_available(),
                     "model on net has cuda in it, awaiting fix")
    def test_densenet(self):
        densenet121 = DenseNet(num_init_features=64, growth_rate=32,
                               block_config=(6, 12, 24, 16), inplace=False)
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        self.run_model_test(densenet121, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    @skip("doesn't match exactly...")
    # TODO: figure out the numerical instabilities
    def test_inception(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 299, 299), requires_grad=True)
        inception = Inception3(aux_logits=True)
        # state_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        state_dict = None
        self.run_model_test(inception, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict, input=x)

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

    # TODO: CUDA side on C2 supports maximum 5 dim
    @skipIfNoLapack
    def test_super_resolution(self):
        super_resolution_net = SuperResolutionNet(upscale_factor=3)
        state_dict = model_zoo.load_url(model_urls['super_resolution'])
        x = Variable(
            torch.randn(BATCH_SIZE, 1, 224, 224), requires_grad=True)
        self.run_model_test(super_resolution_net, train=False,
                            batch_size=BATCH_SIZE, state_dict=state_dict,
                            input=x, use_gpu=False)

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

    def test_constant(self):
        c = Variable(torch.randn(BATCH_SIZE, 3, 224, 224))

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                return input + c.type_as(input)

        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_consumed_bn(self):
        underlying = nn.BatchNorm2d(3)
        self.run_model_test(underlying, train=True, batch_size=BATCH_SIZE)


# add the same test suite as above, but switch embed_params=False
# to embed_params=True
TestCaffe2BackendEmbed = type(str("TestCaffe2BackendEmbed"),
                              (unittest.TestCase,),
                              dict(TestCaffe2Backend.__dict__, embed_params=True))

if __name__ == '__main__':
    unittest.main()
