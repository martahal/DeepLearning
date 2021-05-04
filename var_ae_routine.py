from Project4.VAE import VAE
from Project4.VAE_Encoder import Encoder
from Project4.VAE_Decoder import Decoder
from Project4.verification_net import VerificationNet
from Project3 import visualisations
from Trainer import Trainer
from Project4.stacked_mnist import StackedMNISTData, DataMode
import utils

import matplotlib.pyplot as plt
import numpy as np
import torch


class VAE_Routine():
    def __init__(
            self,
            data,
            learning_rate: float,
            loss_function: str,
            optimizer: str,
            epochs: int,

            latent_vector_size: int,
            batch_size: int,
            num_samples: int,
    ):
        self.data = utils.get_data_to_tensors(data, batch_size)
        self.image_dimensions = (data.test_images.shape[-1], data.test_images.shape[-2], data.test_images.shape[-3])
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.enc_last_layer_dim = (8, 2, 2)
        self.encoder = Encoder(
            input_shape=self.image_dimensions,
            num_filters=16,
            last_conv_layer_dim=self.enc_last_layer_dim,
            output_vector_size=latent_vector_size * 2, # trying this first
            latent_vector_size= latent_vector_size # TODO check this
        ) # The encoder is no longer outputting to the latent vector but to the mean and variance layers

        self.decoder = Decoder(
            input_size=latent_vector_size,
            encoder_last_layer_dim=self.encoder.last_conv_layer_dim,
            hidden_filters=self.encoder.num_filters,
            output_size=self.image_dimensions)

        self.vae = VAE(self.encoder, self.decoder, latent_vector_size)

        self.vae_trainer = Trainer(
            batch_size=batch_size,
            lr=learning_rate,
            epochs=epochs,
            model=self.vae,
            data=self.data,
            loss_function=loss_function,
            optimizer=optimizer,
            early_stop_count=4,
            is_vae=True
        )

    def train_vae(self):
        self.vae_trainer.do_VAE_train()
        self.plot_vae_training(self.vae_trainer)

        # selecting a fixed sample of the test data we like to visualize
        visualisation_data = self.data[1][:12]
        images, reconstructions, labels = utils.make_vae_reconstructions(
            self.vae,
            visualisation_data,
            num_images=12,
            batch_size=self.batch_size,
            image_dimensions=self.image_dimensions)
        # checking quality of reproduced images
        return images, reconstructions, labels

    def generate_samples(self):
        Z = self.get_latent_vectors(self.vae.encoder, self.num_samples )
        generated_images = utils.generate_images_from_Z(Z, self.vae.decoder, self.image_dimensions)
        return generated_images


    @staticmethod
    def plot_vae_training(vae_trainer):
        plt.figure(figsize=(10, 8))
        plt.title('ELBO loss')
        visualisations.plot_metric(vae_trainer.train_history['loss'], label='VAE training loss',
                                   averaged_plot=True)
        visualisations.plot_metric(vae_trainer.validation_history['loss'], label='VAE validation loss',
                                   averaged_plot=False)
        # plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.savefig(
            f'figures/VAE_{vae_trainer.loss_function}_{vae_trainer.epochs}_training.png')

    @staticmethod
    def get_latent_vectors(encoder, n_samples):
        """
        samples a random distribution of the latent vectors, Z, that is produced by the data examples
        :param encoder: The encoder that produces the latent vectors
        :param n_samples: number of samples from Z
        :return: a random sample of Z from the standard normal distribution
        """
        p = torch.distributions.Normal(torch.zeros(encoder.latent_vector_size), torch.ones(encoder.latent_vector_size))
        temp_tensor = torch.ones(n_samples)
        Z = p.sample(sample_shape=temp_tensor.shape) # Wow, so ugly, but my brain hurts now
        return Z
def main():
    #torch.manual_seed(0)

    batch_size = 16
    data_object = StackedMNISTData(
        mode=DataMode.MONO_FLOAT_COMPLETE,
        default_batch_size=batch_size)
    net = VerificationNet(force_learn=False)
    net.train(
        generator=data_object,
        epochs=5)  # gen=data_object, makes sure we test on the same type of data as the model was trained on
    verification_tolerance = 0.8 if data_object.channels == 1 else 0.5

    learning_rate = 1.0e-3
    loss_function = 'elbo'
    optimizer= 'adam'
    epochs = 1

    latent_vector_size = 64
    batch_size = 16
    num_samples = 200

    vae_routine = VAE_Routine(
        data_object,
        learning_rate,
        loss_function,
        optimizer,
        epochs,

        latent_vector_size,
        batch_size,
        num_samples,
    )
    vae_routine.train_vae()
    generated_images = vae_routine.generate_samples()

if __name__ == '__main__':
    main()