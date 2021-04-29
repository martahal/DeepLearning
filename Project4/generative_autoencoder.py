from Projects.Project3.Autoencoder import Autoencoder
from Projects.Project3.Encoder import Encoder
from Projects.Project3.Decoder import Decoder
from Projects.Project4.Trainer import Trainer
from Projects.Project4.stacked_mnist import StackedMNISTData, DataMode

import matplotlib.pyplot as plt
from torch import nn

class Generative_autoencoder:

    def __init__(
            self,
            data,
            autoencoder_learning_rate: float,
            autoencoder_loss_function: str,
            autoencoder_optimizer: str,
            autoencoder_epochs: int,


            latent_vector_size: int,
            batch_size: int,
            training_data_size: int,
            num_samples: int,


    ):
        self.data = data
        self.image_dimensions = (data.test_images.shape[-1], data.test_images.shape[-2], data.test_images.shape[-3])
        self.num_samples = num_samples



        self.encoder = Encoder(
            input_shape=self.image_dimensions,
            num_filters=16,
            last_layer_dim=(32, 10, 10),
            latent_vector_size=latent_vector_size)

        self.decoder = Decoder(
            input_size=latent_vector_size,
            encoder_last_layer_dim=self.encoder.last_layer_dim,
            hidden_filters=self.encoder.num_filters,
            output_size=self.image_dimensions)

        self.autoencoder = Autoencoder(self.encoder, self.decoder, self.image_dimensions)

        self.autoencoder_trainer = Trainer(
            batch_size=batch_size,
            lr=autoencoder_learning_rate,
            training_data_size= training_data_size,
            epochs=autoencoder_epochs,
            model=self.autoencoder,
            data=self.data,
            loss_function=autoencoder_loss_function,
            optimizer=autoencoder_optimizer)

    def train_autoencoder(self):

        self.autoencoder_trainer.do_autoencoder_train()

        #z_sample = self.get_latent_vector_and_classes(self.autoencoder.encoder,self.num_samples, self.dataloaders)

    @staticmethod
    def get_latent_vector_and_classes(encoder,n_samples, data):
        """
        samples a random distribution of the latent vectors, Z, that is produced by the data examples
        :param encoder: The encoder that produces the latent vectors
        :param n_samples: number of samples from Z
        :param data: input data to the encoder
        :return: a random sample of Z from the standard normal distribution
        """
        pass


def main():
    batch_size = 256
    data_object = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=batch_size)
    training_data_size = 20000
    #Produce a subsample of the training set that we actually like to train with instead of the whole of mnist
    #TODO consider to move this elsewhere as it might be applicable to other stuff



    autoencoder_learning_rate = 0.2
    autoencoder_loss_function = 'binary_cross_entropy'  # AVAILABLE 'binary_cross_entropy'
    autoencoder_optimizer = 'SGD'  # AVAILABLE 'SGD' #'adam' #
    autoencoder_epochs = 3  # Optimal for MNIST: 3

    num_samples = 10
    latent_vector_size = 64  # recommended for MNIST between 16 and 64

    gen_autoencoder = Generative_autoencoder(
        data_object,
        autoencoder_learning_rate,
        autoencoder_loss_function,
        autoencoder_optimizer,
        autoencoder_epochs,

        latent_vector_size,
        batch_size,
        training_data_size,
        num_samples,
    )
    gen_autoencoder.train_autoencoder()
    print(gen_autoencoder.image_dimensions)


if __name__ == '__main__':
    main()