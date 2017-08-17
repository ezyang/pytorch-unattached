import torch.nn as nn


class DummyNet(nn.Module):

    def __init__(self, inplace=False, num_classes=1000):
        super(DummyNet, self).__init__()
        self.features = nn.Sequential(
            nn.LeakyReLU(0.02, inplace=inplace)
        )

    def forward(self, x):
        x = self.features(x)
        return x
