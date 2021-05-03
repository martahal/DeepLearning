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
        self.enc_last_layer_dim = (32, 4, 4)
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

def main():
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
    epochs = 2

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

if __name__ == '__main__':
    main()