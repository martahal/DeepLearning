from Project4.Autoencoder import Autoencoder
from Project4.Encoder import Encoder
from Project4.Decoder import Decoder
from Project4.verification_net import VerificationNet
from Project3 import visualisations
from Trainer import Trainer
from Project4.stacked_mnist import StackedMNISTData, DataMode
import utils

import torch
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
        self.latent_vector_size = latent_vector_size


        self.encoder = Encoder(
            input_shape=self.image_dimensions,
            num_filters=16,
            last_conv_layer_dim=(32, 4, 4),
            output_vector_size=latent_vector_size)

        self.decoder = Decoder(
            input_size=latent_vector_size,
            encoder_last_layer_dim=self.encoder.last_conv_layer_dim,
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
        #self.autoencoder_trainer.load_best_model()
        self.autoencoder_trainer.do_autoencoder_train()
        self.plot_autoencoder_training(self.autoencoder_trainer)

    def reconstruct_test_data(self):
        #selecting a fixed sample of the test data we like to visualize
        visualisation_data = self.data[1][:12]
        images, reconstructions, labels = utils.make_reconstructions(
            self.autoencoder,
            visualisation_data,
            num_images=12,
            batch_size=self.batch_size,
            image_dimensions=self.image_dimensions)
        # checking quality of reproduced images
        return images, reconstructions, labels

    def anomaly_detection(self, k):
        # Calculate reconstruction loss (MSE) for test data
        # plot the k most anomalous images
        images, reconstructions, losses = self.autoencoder_trainer.ae_detect_anomaly_by_loss()

        worst_indices = np.argsort(losses)[-1:-(k + 1):-1]
        print("Anomaly loss values:", [losses[index] for index in worst_indices])
        anomalies = np.array([reconstructions[index] for index in worst_indices])
        visualisations.show_images_and_reconstructions(anomalies, f'AE_Anomalies_latent_size:{self.latent_vector_size}_lr_{self.autoencoder_trainer.lr}_epochs:{self.autoencoder_trainer.epochs}')



    def generate_samples(self):
        Z = self.get_latent_vector_and_classes(self.autoencoder.encoder, self.num_samples)#, self.dataloaders)
        generated_images = utils.generate_images_from_Z(Z, self.autoencoder.decoder, self.image_dimensions, title= "Gen_AE_generated_images")
        return generated_images

    def check_autoencoder_performance(self, verification_net, tolerance, images, labels=None):
        coverage = verification_net.check_class_coverage(
            data=images,
            tolerance=tolerance
        )
        print(f"Coverage: {100 * coverage:.2f}%")
        if labels is not None:
            if coverage != 0.0:
                predictability, accuracy = verification_net.check_predictability(
                    data=images,
                    correct_labels=labels,
                    tolerance=tolerance
                )
                print(f"Predictability: {100 * predictability:.2f}%")
                print(f"Accuracy: {100 * accuracy:.2f}%")
        else:
            if coverage != 0.0:
                predictability = verification_net.check_predictability(
                    data=images,
                    tolerance=tolerance
                )
                print(f"Predictability: {100 * predictability}%")#:.2f}%")


        



    @staticmethod
    def get_latent_vector_and_classes(encoder, n_samples):
        """
        samples a random distribution of the latent vectors, Z
        :param encoder: The encoder that produces the latent vectors
        :param n_samples: number of samples from Z
        :return: a random sample of Z from the standard normal distribution
        """
        p = torch.distributions.Normal(torch.zeros(encoder.latent_vector_size), torch.ones(encoder.latent_vector_size))
        temp_tensor = torch.ones(n_samples)
        Z = p.sample(sample_shape=temp_tensor.shape)  # Wow, so ugly, but my brain hurts now
        return Z

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
    torch.manual_seed(0)
    """ GENERATIVE AUTOENCODER ROUTINE"""
    batch_size = 16
    data_object = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=batch_size)
    #instantiate verification network
    net = VerificationNet(force_learn=False)
    net.train(generator=data_object, epochs=5) # gen=data_object, makes sure we test on the same type of data as the model was trained on
    verification_tolerance = 0.8 if data_object.channels == 1 else 0.5

    autoencoder_learning_rate = 0.0002
    autoencoder_loss_function = 'MSE' #'binary_cross_entropy'  # AVAILABLE 'binary_cross_entropy'
    autoencoder_optimizer = 'adam'#'SGD'#  # AVAILABLE 'SGD' # #
    autoencoder_epochs = 10  # Optimal for MNIST: 3

    num_samples = 200
    latent_vector_size = 64  # recommended for MNIST between 16 and 64

    #gen_autoencoder = Generative_autoencoder(
    #    data_object,
    #    autoencoder_learning_rate,
    #    autoencoder_loss_function,
    #    autoencoder_optimizer,
    #    autoencoder_epochs,
    #    # TODO add path to model weights as argument
    #    latent_vector_size,
    #    batch_size,
    #    num_samples,
    #)
    #gen_autoencoder.train_autoencoder()
    #images, reconstructions, labels = gen_autoencoder.reconstruct_test_data():
    ##Check quality of reconstructions
    #gen_autoencoder.check_autoencoder_performance(net, verification_tolerance, reconstructions, labels)

    ##Generate samples
    #generated_images = gen_autoencoder.generate_samples()

    ##check quality of generated images
    #gen_autoencoder.check_autoencoder_performance(net, verification_tolerance, generated_images)

    """ ANOMALY DETECTOR AUTOENCODER ROUTINE"""
    data_object = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=batch_size)
    number_anom_images_to_show = 6
    anom_autoencoder = Generative_autoencoder(
        data_object,
        autoencoder_learning_rate,
        autoencoder_loss_function,
        autoencoder_optimizer,
        autoencoder_epochs,
        # TODO add path to model weights as argument
        latent_vector_size,
        batch_size,
        num_samples,
    )
    #images, reconstructions, labels = anom_autoencoder.train_autoencoder()

    anom_autoencoder.anomaly_detection(number_anom_images_to_show)
if __name__ == '__main__':
    main()