from torch import nn

from Project4.stacked_mnist import StackedMNISTData
from Project4.verification_net import VerificationNet

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        # Defining encoder and decoder:
        self.encoder = encoder  # The encoder is no longer outputting to the latent vector but to the mean and variance layers
        self.decoder = decoder

    def forward(self, x):
        encoded_x, mean, log_std = self.encoder.sample_encoded_x(x)

        x_hat = self.decoder(encoded_x)
        return x_hat, mean, log_std, encoded_x



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

