import torch
import torch.nn as nn
from torch.autograd import Variable


class DummyNet(nn.Module):

    def __init__(self, inplace=False, num_classes=1000):
        super(DummyNet, self).__init__()
        self.features = nn.Sequential(
            nn.LeakyReLU(0.02, inplace=inplace),
            nn.BatchNorm2d(3),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )

    def forward(self, x):
        output = self.features(x)
        output.view(-1, 1).squeeze(1)
        return x


class ConcatNet(nn.Module):

    def __init__(self):
        super(ConcatNet, self).__init__()

    def forward(self, inputs):
        return torch.cat(inputs, 1)


class PermuteNet(nn.Module):

    def __init__(self):
        super(PermuteNet, self).__init__()

    def forward(self, input):
        return input.permute(2, 3, 0, 1)

class IndexNet(nn.Module):

    def __init__(self):
        super(IndexNet, self).__init__()

    def forward(self, input):
        input.data = input.data.index(0)
        return input
