from torch import nn, distributions
import torch
class Encoder(nn.Module):

    def __init__(self,
                 input_shape,
                 num_filters,
                 last_conv_layer_dim,
                 output_vector_size,
                 latent_vector_size):
        """
        Constructs the encoder used in the SSN and SCN
        :param input_shape: tuple, (number of color channels of the image, image width, image height)
        :param latent_vector_size: int, length of latent vector
        """
        super().__init__()

        self.input_channels = input_shape[0]
        self.num_filters = num_filters
        self.output_vector_size =output_vector_size
        self.latent_vector_size = latent_vector_size
        self.last_conv_layer_dim = last_conv_layer_dim
        #self.last_layer_dim = (self.num_filters, input_shape[1], input_shape[2]) # = num_filters * width * height * some constant depending on convolution process
        self.body = nn.Sequential(
            #  [in: 1, out ___ , input spatial size: __x__, output spatial size: __x__ (same spatial output as input) ]
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(3,3),
                stride=(3,3),
                padding=(1,1)
            ),
            nn.BatchNorm2d(self.num_filters),
            #nn.Conv2d(
            #    in_channels=self.num_filters,
            #    out_channels=self.num_filters //2,
            #    kernel_size=(3, 3),
            #    stride=(3,3),
            #    padding=(1, 1)
            #),
            #nn.BatchNorm2d(self.num_filters//2),
            #nn.Conv2d(
            #    in_channels=self.num_filters//2,
            #    out_channels=self.num_filters//4,
            #    kernel_size=(3, 3),
            #    stride=(3,3),
            #    padding=(1,1)
            #),
            #nn.BatchNorm2d(self.num_filters//4),
            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=self.last_conv_layer_dim[0],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(self.last_conv_layer_dim[0]),
            nn.ReLU(),
            nn.Flatten(),
            #nn.Linear(
            #    in_features=self.last_conv_layer_dim[0] * self.last_conv_layer_dim[1] * self.last_conv_layer_dim[2],
            #    out_features=self.output_vector_size),
        )
        self.mean_layer = nn.Linear(
            in_features=self.last_conv_layer_dim[0] * self.last_conv_layer_dim[1] * self.last_conv_layer_dim[2],
            out_features=self.latent_vector_size)
        self.log_std_layer = nn.Linear(
            in_features=self.last_conv_layer_dim[0] * self.last_conv_layer_dim[1] * self.last_conv_layer_dim[2],
            out_features=self.latent_vector_size)

    def forward(self, x):
       """
       Performs the forward pass of the encoder
       :param x:  tensor, image input shape [batch size, image_width, image_height]
       :return mean and log std values
       """
       x = self.body(x)

       #self._test_correct_output(x)
       return self.mean_layer(x), self.log_std_layer(x)

    def sample_encoded_x(self, x):
        # Learn a parametrization q(z|x), make the distribution and sample from it
        mean, log_var = self.forward(x)
        std = torch.exp(log_var/2)
        #Debugging
        #ok = distributions.Normal.arg_constraints["loc"].check(std)
        #bad_elements = std[~ok]
        #print(bad_elements)
        #Debugging
        q = distributions.Normal(mean, std)
        z = q.rsample()
        return z, mean, std

    def _test_correct_output(self, output):
        batch_size = output.shape[0]
        expected_shape = (batch_size, self.output_vector_size)
        assert output.shape == (batch_size, self.output_vector_size), \
            f"Expected output of forward pass to be: {expected_shape}, but got: {output.shape}"
