from torch import nn


class Encoder(nn.Module):

    def __init__(self,
                 input_shape,
                 num_filters,
                 latent_vector_size):
        """
        Constructs the encoder used in the SSN and SCN
        :param input_shape: tuple, (number of color channels of the image, image width, image height)
        :param latent_vector_size: int, length of latent vector
        """
        super().__init__()

        self.input_channels = input_shape[0]
        self.num_filters = num_filters
        self.latent_vector_size = latent_vector_size
        self.last_layer_dim = (self.num_filters, input_shape[1], input_shape[2]) # = num_filters * width * height * some constant depending on convolution process
        self.model = nn.Sequential(
            #  [in: 1, out ___ , input spatial size: __x__, output spatial size: __x__ (same spatial output as input) ]
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=self.last_layer_dim[0] * self.last_layer_dim[1] * self.last_layer_dim[2],
                out_features=self.latent_vector_size))

    def forward(self, x):
       """
       Performs the forward pass of the encoder
       :param x:  tensor, image input shape [batch size, image_width, image_height]
       :return output: the latent vector of the encoder, shape [batch size, latent vector size]
       """
       x = self.model(x)
       output = x

       self._test_correct_output(output)
       return output

    def _test_correct_output(self, output):
        batch_size = output.shape[0]
        expected_shape = (batch_size, self.latent_vector_size)
        assert output.shape == (batch_size, self.latent_vector_size), \
            f"Expected output of forward pass to be: {expected_shape}, but got: {output.shape}"
