from Projects.Project3.Encoder import Encoder
from Projects.Project3.Trainer import Trainer

from torch import nn


class Autoencoder(nn.Module):

    def __init__(self,
                 encoder,
                 decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """
        Performs the forward pass of the autoencoder
        :param x: Input image, shape: [batch_size, image channels, width, height]
        :return:
        """