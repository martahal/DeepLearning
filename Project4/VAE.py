import pytorch_lightning as pl
from torch import nn
import torch
import pyro
import pyro.distributions as dist
from Project4.Encoder import Encoder
from Project4.Decoder import Decoder
from Project4.stacked_mnist import StackedMNISTData
from Project4.verification_net import VerificationNet

class VAE(nn.Module):
    def __init__(self, encoder_output_size, latent_vector_size, encoder, decoder):
        super().__init__()
        # Defining encoder and decoder:
        self.encoder = encoder  # The encoder is no longer outputting to the latent vector but to the mean and variance layers
        self.decoder = decoder

        self.mu_layer = nn.Linear(encoder_output_size, latent_vector_size)
        self.variance_layer = nn.Linear(encoder_output_size, latent_vector_size)
        self.log_scale = nn.Parameter(torch.Tensor([0.0])) #TODO What is this?


        self.latent_vector_size = latent_vector_size
        self.encoder_output_size = encoder_output_size

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x): # since we use convolution x is not reshaped to flat image, hence just x and not x.shape[0]
            # parameters for prior p(z)
            zeros = x.new_zeros(torch.Size((self.encoder_output_size, self.latent_vector_size)))
            ones = x.new_ones(torch.Size((self.encoder_output_size, self.latent_vector_size)))
            p = dist.Normal(zeros, ones)
            #sample z from p to calculate p(z)
            z = pyro.sample("latent_vectors", p.to_event(1)) # TODO Resolve this doesn't make sense. Should sample from q?
            # decoding the latent vectors
            x_hat = self.decoder(z)

            #calculate reconstruction_loss log_p(x|z) from gaussian likelihood
            scale = torch.exp(self.log_scale)
            mean = x_hat
            p_xz = dist.Normal(mean,scale)
            pyro.sample("obs", p_xz.to_event(1), obs=x)



    # Define q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x): # since we use convolution x is not reshaped to flat image, hence just x and not x.shape[0]
            # Parametrize Q(z|x)
            encoded_x = self.encoder(x)
            mu, log_variance = self.mu_layer(encoded_x), self.variance_layer(
                encoded_x)  # TODO Why is this called log_var?
            sigma = torch.exp(log_variance / 2)
            q = dist.Normal(mu,sigma)
            pyro.sample("latent_vectors", q.to_event(1)) # TODO i have noo clue what this does.

    def reconstruct_images(self,x):
        encoded_x = self.encoder(x)
        mu, log_variance = self.mu_layer(encoded_x), self.variance_layer(encoded_x)
        #sample from the latent_variable
        z = dist.Normal(mu, log_variance).sample()
        reconstruction = self.decoder(z)
        return reconstruction

def main():
    batch_size = 16
    data_object = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=batch_size)
    net = VerificationNet(force_learn=False)
    net.train(generator=data_object,
              epochs=5)  # gen=data_object, makes sure we test on the same type of data as the model was trained on
    verification_tolerance = 0.8 if data_object.channels == 1 else 0.5


    encoder_output_size = 32 * 4 * 4 # Just last layer feature map dimensions flattened
    latent_vector_size = 64

    vae = VAE(encoder_output_size, latent_vector_size)



if __name__ == '__main__':
    main()

