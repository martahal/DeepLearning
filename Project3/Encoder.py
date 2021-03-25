from torch import nn


class Encoder(nn.Module):

    def __init__(self,
                 input_size,
                 num_filters,
                 latent_vector_size):
        """
        Constructs the encoder used in the SSN and SCN
        :param input_size: tuple, (number of color channels of the image, image width, image height)
        :param latent_vector_size: int, length of latent vector
        """
        super().__init__()

        self.input_channels = input_size[0]
        self.num_filters = num_filters
        self.latent_vector_size = latent_vector_size
        self.last_layer_features = self.num_filters # = num_filters * some constant depending on convolution process
        self.model = nn.Sequential(
            #  [in: 3, out ___ , input spatial size: __x__, output spatial size: __x__ (same spatial output as input) ]
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=self.num_filters, out_features=self.latent_vector_size))


