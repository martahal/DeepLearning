from torch import nn

class Decoder(nn.Module):

    def __init__(self,
                 input_size,
                 encoder_last_layer_dim,
                 hidden_filters,
                 output_size):
        super().__init__()
        self.input_size = input_size
        self.encoder_last_layer_dim = encoder_last_layer_dim

        self.channels,\
        self.height, \
        self.width = self.encoder_last_layer_dim

        self.hidden_filters = hidden_filters
        self.output_size = output_size

        self.reconstructed_channels, \
        self.reconstructed_height, \
        self.reconstructed_width = self.output_size

        self.model = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=self.encoder_last_layer_dim[0] * self.encoder_last_layer_dim[1] * self.encoder_last_layer_dim[2]),
            nn.Unflatten(
                dim= 1,
                unflattened_size=self.encoder_last_layer_dim),
            nn.ConvTranspose2d(
                in_channels=self.encoder_last_layer_dim[0],
                out_channels=self.hidden_filters,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)
            ),
            nn.ConvTranspose2d(
                in_channels=self.encoder_last_layer_dim[0],
                out_channels=self.hidden_filters,
                kernel_size=(3, 3),
                stride=(3, 3),
                padding=(1, 1)
            ),
            nn.ConvTranspose2d(
                in_channels=self.hidden_filters,
                out_channels=self.reconstructed_channels,
                kernel_size=(3, 3),
                stride=(3, 3),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.Sigmoid() # scale reconstruction between 0 and 1
        )

    def forward(self, latent_vector):
        """
        Performs the forward pass of the decoder
        :param latent_vector: tensor, The latent vector of the encoder, shape [batch size, latent vector size]
        :return:
        """
        image = self.model(latent_vector)
        #self._test_correct_output(image)

        return image

    def _test_correct_output(self, output):
        batch_size = output.shape[0]
        expected_shape = (batch_size, self.output_size)
        assert output.shape == (batch_size, self.output_size), \
            f"Expected output of forward pass to be: {expected_shape}, but got: {output.shape}"