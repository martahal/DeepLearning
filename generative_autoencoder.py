from Project4.Autoencoder import Autoencoder
from Project4.Encoder import Encoder
from Project4.Decoder import Decoder
from Project3 import visualisations
from Trainer import Trainer
from Project4.stacked_mnist import StackedMNISTData, DataMode
import utils

import matplotlib.pyplot as plt
import numpy as np


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
            num_samples: int,


    ):
        self.data = utils.get_data_to_tensors(data, batch_size)
        self.image_dimensions = (data.test_images.shape[-1], data.test_images.shape[-2], data.test_images.shape[-3])
        self.num_samples = num_samples
        self.batch_size = batch_size


        self.encoder = Encoder(
            input_shape=self.image_dimensions,
            num_filters=16,
            last_layer_dim=(32, 4, 4),
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
            epochs=autoencoder_epochs,
            model=self.autoencoder,
            data=self.data,
            loss_function=autoencoder_loss_function,
            optimizer=autoencoder_optimizer,
            early_stop_count = 4
        )

    def train_autoencoder(self):
        self.autoencoder_trainer.load_best_model()
        self.autoencoder_trainer.do_autoencoder_train()
        self.plot_autoencoder_training(self.autoencoder_trainer)

        Z = self.get_latent_vector_and_classes(self.autoencoder.encoder, self.num_samples)#, self.dataloaders)
        utils.generate_images_from_Z(Z, self.autoencoder.decoder, self.image_dimensions)
        #selecting a fixed sample of the test data we like to visualize
        visualisation_data = self.data[1][:12]
        utils.make_reconstructions_figure(
            self.autoencoder,
            visualisation_data,
            num_images=12,
            batch_size=self.batch_size,
            image_dimensions=self.image_dimensions)




    @staticmethod
    def get_latent_vector_and_classes(encoder, n_samples): #, data):
        """
        samples a random distribution of the latent vectors, Z, that is produced by the data examples
        :param encoder: The encoder that produces the latent vectors
        :param n_samples: number of samples from Z
        :param data: input data to the encoder
        :return: a random sample of Z from the standard normal distribution
        """
        z = np.random.randn(n_samples, encoder.latent_vector_size)
        return z

    @staticmethod
    def plot_autoencoder_training(autoencoder_trainer):
        plt.figure(figsize=(10, 8))
        plt.title('Autoencoder loss')
        visualisations.plot_metric(autoencoder_trainer.train_history['loss'], label='Autoencoder training loss',
                                   averaged_plot=True)
        visualisations.plot_metric(autoencoder_trainer.validation_history['loss'], label='Autoencoder validation loss',
                                   averaged_plot=False)
        # plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.savefig(f'figures/autoencoder_{autoencoder_trainer.loss_function}_{autoencoder_trainer.epochs}_training.png')

def main():
    batch_size = 16
    data_object = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=batch_size)

    autoencoder_learning_rate = 0.02
    autoencoder_loss_function = 'MSE' #'binary_cross_entropy'  # AVAILABLE 'binary_cross_entropy'
    autoencoder_optimizer = 'adam'#'SGD'  # AVAILABLE 'SGD' # #
    autoencoder_epochs = 10  # Optimal for MNIST: 3

    num_samples = 12
    latent_vector_size = 64  # recommended for MNIST between 16 and 64

    gen_autoencoder = Generative_autoencoder(
        data_object,
        autoencoder_learning_rate,
        autoencoder_loss_function,
        autoencoder_optimizer,
        autoencoder_epochs,

        latent_vector_size,
        batch_size,
        num_samples,
    )
    gen_autoencoder.train_autoencoder()


if __name__ == '__main__':
    main()