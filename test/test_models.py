import torch
import torch.jit
from torch.autograd import Variable
from common import TestCase, run_tests
from model_defs.alexnet import AlexNet
from model_defs.vgg import make_layers, VGG, cfg
from model_defs.resnet import Bottleneck, ResNet
from model_defs.inception import Inception3
from model_defs.squeezenet import SqueezeNet
from model_defs.densenet import DenseNet
from model_defs.dcgan import _netD, _netG, weights_init, bsz, imgsz, nz, ngf, ndf, nc
from model_defs.op_test import DummyNet


class TestModels(TestCase):
    maxDiff = None

    def test_leaky_relu(self):

        inplace = False
        x = Variable(
            torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True
        )
        trace, _ = torch.jit.record_trace(DummyNet(inplace=inplace), x)
        print(str(trace))
        self.assertExpected(str(trace))
        self.assertExpected(torch._C._jit_pass_export(trace), "pbtxt")

    def test_alexnet(self):

        inplace = False
        x = Variable(
            torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True
        )
        trace, _ = torch.jit.record_trace(AlexNet(inplace=inplace), x)
        self.assertExpected(str(trace))
        self.assertExpected(torch._C._jit_pass_export(trace), "pbtxt")

    def test_vgg(self):

        inplace = False
        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        vgg16 = VGG(make_layers(cfg['D']), inplace=inplace)
        trace, _ = torch.jit.record_trace(vgg16, x)
        self.assertExpected(str(trace), "16")
        self.assertExpected(torch._C._jit_pass_export(trace), "16-pbtxt")

        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        vgg16_bn = VGG(make_layers(cfg['D'], inplace=inplace, batch_norm=True))
        trace, _ = torch.jit.record_trace(vgg16_bn, x)
        self.assertExpected(str(trace), "16_bn")
        # self.assertExpected(torch._C._jit_pass_export(trace), "16_bn-pbtxt")

        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        vgg19 = VGG(make_layers(cfg['E']), inplace=inplace)
        trace, _ = torch.jit.record_trace(vgg19, x)
        self.assertExpected(str(trace), "19")
        self.assertExpected(torch._C._jit_pass_export(trace), "19-pbtxt")

        # VGG 19-layer model (configuration 'E') with batch normalization
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        vgg19_bn = VGG(make_layers(cfg['E'], inplace=inplace, batch_norm=True))
        trace, _ = torch.jit.record_trace(vgg19_bn, x)
        self.assertExpected(str(trace), "19_bn")
        # self.assertExpected(torch._C._jit_pass_export(trace), "19_bn-pbtxt")

    def test_resnet(self):

        inplace = False
        # ResNet50 model
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], inplace=inplace)
        trace, _ = torch.jit.record_trace(resnet50, x)
        self.assertExpected(str(trace), "50")
        # self.assertExpected(torch._C._jit_pass_export(trace), "50-pbtxt")

    def test_inception(self):

        inplace = False
        x = Variable(
            torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(Inception3(inplace=inplace), x)
        self.assertExpected(str(trace), "3")
        # self.assertExpected(torch._C._jit_pass_export(trace), "3-pbtxt")

    def test_squeezenet(self):

        inplace = False
        # SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
        # <0.5MB model size
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        sqnet_v1_0 = SqueezeNet(version=1.1, inplace=inplace)
        trace, _ = torch.jit.record_trace(sqnet_v1_0, x)
        self.assertExpected(str(trace), "1_0")
        # self.assertExpected(torch._C._jit_pass_export(trace), "1_0-pbtxt")

        # SqueezeNet 1.1 has 2.4x less computation and slightly fewer params
        # than SqueezeNet 1.0, without sacrificing accuracy.
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        sqnet_v1_1 = SqueezeNet(version=1.1, inplace=inplace)
        trace, _ = torch.jit.record_trace(sqnet_v1_1, x)
        self.assertExpected(str(trace), "1_1")
        # self.assertExpected(torch._C._jit_pass_export(trace), "1_1-pbtxt")

    def test_densenet(self):

        inplace = False
        # Densenet-121 model
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0),
                     requires_grad=True)
        dense121 = DenseNet(num_init_features=64, growth_rate=32,
                            block_config=(6, 12, 24, 16), inplace=inplace)
        trace, _ = torch.jit.record_trace(dense121, x)
        self.assertExpected(str(trace), "121")
        # self.assertExpected(torch._C._jit_pass_export(trace), "121-pbtxt")

    def test_dcgan(self):
        # note, could have more than 1 gpu
        netG = _netG(1)
        netG.apply(weights_init)
        netD = _netD(1)
        netD.apply(weights_init)

        input = torch.Tensor(bsz, 3, imgsz, imgsz)
        noise = torch.Tensor(bsz, nz, 1, 1)
        fixed_noise = torch.Tensor(bsz, nz, 1, 1).normal_(0, 1)

        fixed_noise = Variable(fixed_noise)

        netD.zero_grad()
        inputv = Variable(input)
        trace, _ = torch.jit.record_trace(netD, inputv)
        self.assertExpected(str(trace), "dcgan-netD")
        # self.assertExpected(torch._C._jit_pass_export(trace), "dcgan-netD-pbtxt")

        noise.resize_(bsz, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        trace, _ = torch.jit.record_trace(netG, noisev)
        self.assertExpected(str(trace), "dcgan-netG")
        # self.assertExpected(torch._C._jit_pass_export(trace), "dcgan-netG-pbtxt")


if __name__ == '__main__':
    run_tests()
