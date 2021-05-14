from GenerativeModelling.Autoencoder import Autoencoder
from GenerativeModelling.Encoder import Encoder
from GenerativeModelling.Decoder import Decoder
from GenerativeModelling.verification_net import VerificationNet
from SemiSupervisedLearning import visualisations
from GenerativeModelling.Trainer import Trainer
from GenerativeModelling.stacked_mnist import StackedMNISTData, DataMode
from GenerativeModelling import utils

import torch
import matplotlib.pyplot as plt
import numpy as np

import pathlib


class Generative_AE_Routine:

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
            save_path: str

    ):
        self.data = utils.get_data_to_tensors(data, batch_size)
        self.image_dimensions = (data.test_images.shape[-1], data.test_images.shape[-2], data.test_images.shape[-3])
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.latent_vector_size = latent_vector_size


        self.encoder = Encoder(
            input_shape=self.image_dimensions,
            num_filters=16,
            last_conv_layer_dim= (16,10, 10), #(32, 4, 4),
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
            early_stop_count = 4,
            model_save_path=save_path,
        )

    def train_autoencoder(self):
        #self.autoencoder_trainer.load_best_model()
        self.autoencoder_trainer.do_autoencoder_train()
        self.plot_autoencoder_training(self.autoencoder_trainer)

    def reconstruct_test_data(self, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.autoencoder.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')
        #selecting a fixed sample of the test data we like to visualize
        visualisation_data = self.data[1]
        images, reconstructions, labels = utils.make_reconstructions(
            self.autoencoder,
            visualisation_data,
            num_images=25,
            batch_size=self.batch_size,
            image_dimensions=self.image_dimensions,
            title=f'AE_z_size:{self.latent_vector_size}_lr_{self.autoencoder_trainer.lr}_epochs:{self.autoencoder_trainer.epochs}'
        )
        # checking quality of reproduced images
        return images, reconstructions, labels

    def anomaly_detection(self, k, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.autoencoder.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')
        # Calculate reconstruction loss (MSE) for test data
        # plot the k most anomalous images
        images, reconstructions, losses = self.autoencoder_trainer.ae_detect_anomaly_by_loss()

        worst_indices = np.argsort(losses)[-1:-(k + 1):-1]
        print("Anomaly loss values:", [losses[index] for index in worst_indices])
        anomalies = np.array([images[index] for index in worst_indices])
        visualisations.show_images_and_reconstructions(anomalies, f'AE_Anomalies_latent_size:{self.latent_vector_size}_lr_{self.autoencoder_trainer.lr}_epochs:{self.autoencoder_trainer.epochs}')



    def generate_samples(self, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.autoencoder.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')
        Z = self.get_latent_vector_and_classes(self.autoencoder.encoder, self.num_samples)#, self.dataloaders)
        generated_images = utils.generate_images_from_Z(Z, self.autoencoder.decoder, self.image_dimensions, title="Gen_AE_generated_images")
        return generated_images

    def check_autoencoder_performance(self, verification_net, tolerance, images, labels=None, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.autoencoder.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')
        coverage = verification_net.check_class_coverage(
            data=images,
            tolerance=tolerance
        )
        print(f"Coverage: {100 * coverage:.2f}%")
        if labels is not None:
            #if coverage != 0.0:
                predictability, accuracy = verification_net.check_predictability(
                    data=images,
                    correct_labels=labels,
                    tolerance=tolerance
                )
                print(f"Predictability: {100 * predictability:.2f}%")
                print(f"Accuracy: {100 * accuracy:.2f}%")
        else:
            if coverage != 0.0:
                predictability, accuracy = verification_net.check_predictability(
                    data=images,
                    tolerance=tolerance
                )
                print(f"Predictability: {100 * predictability:.2f}%")

    @staticmethod
    def get_latent_vector_and_classes(encoder, n_samples):
        """
        samples a random distribution of the latent vectors, Z
        :param encoder: The encoder that produces the latent vectors
        :param n_samples: number of samples from Z
        :return: a random sample of Z from the standard normal distribution
        """
        p = torch.distributions.Normal(torch.zeros(encoder.output_vector_size), torch.ones(encoder.output_vector_size))
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
    autoencoder_epochs = 1  # Optimal for MNIST: 3

    num_samples = 2000
    latent_vector_size = 64  # recommended for MNIST between 16 and 64
    gen_name = 'Test_gen_AE'
    gen_ae_save_path = f'checkpoints/gen_AE/{gen_name}'
    gen_autoencoder = Generative_AE_Routine(
        data_object,
        autoencoder_learning_rate,
        autoencoder_loss_function,
        autoencoder_optimizer,
        autoencoder_epochs,

        latent_vector_size,
        batch_size,
        num_samples,
        gen_ae_save_path
    )
    gen_autoencoder.train_autoencoder()
    images, reconstructions, labels = gen_autoencoder.reconstruct_test_data()
    #Check quality of reconstructions
    gen_autoencoder.check_autoencoder_performance(net, verification_tolerance, reconstructions, labels)
#
    ##Generate samples
    #generated_images = gen_autoencoder.generate_samples()
#
    ##check quality of generated images
    #gen_autoencoder.check_autoencoder_performance(net, verification_tolerance, generated_images)
#
    #""" ANOMALY DETECTOR AUTOENCODER ROUTINE"""
    #data_object = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=batch_size)
    #number_anom_images_to_show = 16
    #anom_name = 'Test_anom_AE'
    #anom_ae_save_path = f'checkpoints/anom_AE/{anom_name}/'
    #anom_autoencoder = Generative_AE_Routine(
    #    data_object,
    #    autoencoder_learning_rate,
    #    autoencoder_loss_function,
    #    autoencoder_optimizer,
    #    autoencoder_epochs,
    #
    #    latent_vector_size,
    #    batch_size,
    #    num_samples,
    #    anom_ae_save_path
    #)
    #anom_autoencoder.train_autoencoder()
#
    #anom_autoencoder.anomaly_detection(number_anom_images_to_show)
if __name__ == '__main__':
    main()