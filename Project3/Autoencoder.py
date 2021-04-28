from Projects.Project3.Encoder import Encoder
from Projects.Project3.Decoder import Decoder
from Projects.Project3.Trainer import Trainer
from Projects.Project3.Dataloader import load_fashion_mnist

from torch import nn


class Autoencoder(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 reconstructed_image_shape):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.reconstructed_image_shape = reconstructed_image_shape
    def forward(self, x):
        """
        Performs the forward pass of the autoencoder
        :param x: Input image, shape: [batch_size, image channels, width, height]
        :return: reconstructed images, shape [batch_size, image channels, width, height]
                latent vectors, shape [batch size, latent vector size]
        """
        latent_vector = self.encoder(x)
        reconstructed_image = self.decoder(latent_vector)

        #self._test_correct_output(reconstructed_image)

        return reconstructed_image, latent_vector


    def _test_correct_output(self, output):
        batch_size = output.shape[0]
        expected_shape = (batch_size, self.reconstructed_image_shape)
        assert output.shape == (batch_size, self.reconstructed_image_shape), \
            f"Expected output of forward pass to be: {expected_shape}, but got: {output.shape}"

def main():
    epochs = 10
    batch_size = 16
    learning_rate = 0.0002
    loss_function = 'binary_cross_entropy'
    optimizer = 'SGD'

    # Loading FashionMNIST
    num_classes = 10
    dataloaders = load_fashion_mnist(batch_size, D1_fraction=0.8, validation_fraction=0.1, test_fraction=0.1)
    input_size = (1,28,28) # TODO this should be detected on the fly when we determine which dataset to run
    latent_vector_size = 256

    encoder = Encoder(
        input_shape=input_size,
        num_filters=4,
        latent_vector_size=latent_vector_size)
    decoder = Decoder(
        input_size=latent_vector_size,
        encoder_last_layer_dim=encoder.last_layer_dim,
        output_size=input_size)
    Autoencoder_model = Autoencoder(encoder, decoder, input_size)

    SSN_trainer = Trainer(batch_size=batch_size,
                          lr=learning_rate,
                          epochs= epochs,
                          model= Autoencoder_model,
                          dataloaders=dataloaders,
                          loss_function=loss_function,
                          optimizer=optimizer)

    SSN_trainer.do_autoencoder_train()

if __name__ == '__main__':
    main()