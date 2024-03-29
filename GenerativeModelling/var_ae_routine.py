from GenerativeModelling.VAE import VAE
from GenerativeModelling.VAE_Encoder import Encoder
from GenerativeModelling.VAE_Decoder import Decoder
from GenerativeModelling.verification_net import VerificationNet
from SemiSupervisedLearning import visualisations
from GenerativeModelling.Trainer import Trainer
from GenerativeModelling.stacked_mnist import StackedMNISTData, DataMode
from GenerativeModelling import utils

import matplotlib.pyplot as plt
import numpy as np
import torch

import pathlib


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
            save_path: str
    ):
        self.data = utils.get_data_to_tensors(data, batch_size)
        self.image_dimensions = (data.test_images.shape[-1], data.test_images.shape[-2], data.test_images.shape[-3])
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.enc_last_layer_dim = (8,10,10) #(32, 2, 2)#(8, 4, 4) #
        self.latent_vector_size = latent_vector_size



        self.encoder = Encoder(
            input_shape=self.image_dimensions,
            num_filters=32,
            last_conv_layer_dim=self.enc_last_layer_dim,
            output_vector_size=latent_vector_size * 2, # trying this first
            latent_vector_size= latent_vector_size # TODO check this
        ) # The encoder is no longer outputting to the latent vector but to the mean and variance layers

        self.decoder = Decoder(
            input_size=latent_vector_size,
            encoder_last_layer_dim=self.encoder.last_conv_layer_dim,
            hidden_filters=self.encoder.num_filters,
            output_size=self.image_dimensions)

        self.vae = VAE(self.encoder, self.decoder)

        self.vae_trainer = Trainer(
            batch_size=batch_size,
            lr=learning_rate,
            epochs=epochs,
            model=self.vae,
            data=self.data,
            loss_function=loss_function,
            optimizer=optimizer,
            early_stop_count=4,
            model_save_path=save_path,
            is_vae=True
        )

    def train_vae(self):

        self.vae_trainer.do_VAE_train()

        self.plot_vae_training(self.vae_trainer, self.enc_last_layer_dim)

    def reconstruct_test_data(self, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.vae.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')

        # selecting a fixed sample of the test data we like to visualize
        visualisation_data = self.data[1][:] #self.data[1][:12]
        images, reconstructions, labels = utils.make_vae_reconstructions(
            self.vae,
            visualisation_data,
            num_images=25,
            batch_size=self.batch_size,
            image_dimensions=self.image_dimensions,
            title=f'VAE_z_size:{self.latent_vector_size}_lr_{self.vae_trainer.lr}_epochs:{self.vae_trainer.epochs}'
        )
        # checking quality of reproduced images
        # Returned images are numpy arrays
        return images, reconstructions, labels

    def anomaly_detection(self, k, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.vae.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')

        # Calculate reconstruction loss (MSE) for test data
        # plot the k most anomalous images
        images, reconstructions, losses = self.vae_trainer.vae_detect_anomaly_by_loss()

        worst_indices = np.argsort(losses)[-1:-(k + 1):-1]
        print("Anomaly loss values:", [losses[index] for index in worst_indices])
        anomalies = np.array([images[index] for index in worst_indices])
        visualisations.show_images_and_reconstructions(anomalies, f'VAE_Anomalies_latent_size:{self.latent_vector_size}_lr_{self.vae_trainer.lr}_epochs:{self.vae_trainer.epochs}')



    def generate_samples(self, data_object, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.vae.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')
        Z = self.sample_Z(self.vae.encoder, data_object, self.num_samples)
        generated_images = utils.generate_images_from_Z(
            Z,
            self.vae.decoder,
            self.image_dimensions,
            title=f'VAE_z_size:{self.latent_vector_size}_lr_{self.vae_trainer.lr}_epochs:{self.vae_trainer.epochs}'
        )
        return generated_images

    def check_vae_performance(self, verification_net, tolerance, images, labels=None, load_model_path=None):
        if load_model_path is not None:
            # self.vae_trainer.load_best_model() Does not return the model but sets the self.model in trainer to be best model
            # see if we can do:
            self.vae.load_state_dict(torch.load(pathlib.Path(load_model_path).joinpath("best.ckpt")))
            print(f'Loaded model from {load_model_path}')

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
                predictability, accuracy = verification_net.check_predictability(
                    data=images,
                    tolerance=tolerance
                )
                print(f"Predictability: {100 * predictability:.2f}%")#")


    @staticmethod
    def plot_vae_training(vae_trainer, enc_last_layer_dim):
        plt.figure(figsize=(10, 8))
        plt.title('ELBO loss')
        visualisations.plot_metric(vae_trainer.train_history['loss'], label='VAE training loss',
                                   averaged_plot=True)
        visualisations.plot_metric(vae_trainer.validation_history['loss'], label='VAE validation loss',
                                   averaged_plot=False)
        # plt.ylim(bottom=0, top=1)
        plt.legend()
        plt.savefig(
            f'figures/VAE_ll_dim:{enc_last_layer_dim}_lr:{vae_trainer.lr}_epochs:{vae_trainer.epochs}_training.png')
        plt.show()

    @staticmethod
    def sample_Z(encoder, data, n_samples):
        """
        samples a random distribution of the latent vectors, Z, that is produced by the data examples
        :param encoder: The encoder that produces the latent vectors
        :param n_samples: number of samples from Z
        :return: a random sample of Z from the standard normal distribution
        """
        epsilon = torch.distributions.Normal(torch.zeros(encoder.latent_vector_size), torch.ones(encoder.latent_vector_size))
        # Ugly fix to get sample in the shape I want
        temp_tensor = torch.ones(n_samples)
        # Inefficient quick-fix to make data batch in the shape we want
        #(train_data, test_data) = utils.get_data_to_tensors(data, batch_size=n_samples)

        # Reparametrization trick
        #x_hat = test_data[0][0]
        #mu, sigma = encoder(x_hat)
        # get samples Z = mean * std * epsilon
        Z = epsilon.sample(sample_shape=temp_tensor.shape) #* mu * sigma
        return Z
def main():
    torch.manual_seed(1)
    """ GENERATIVE VAE ROUTINE"""
    batch_size = 256
    data_object = StackedMNISTData(
        mode=DataMode.MONO_FLOAT_COMPLETE,
        default_batch_size=batch_size)
    net = VerificationNet(force_learn=False)
    net.train(
        generator=data_object,
        epochs=5)  # gen=data_object, makes sure we test on the same type of data as the model was trained on
    verification_tolerance = 0.8 if data_object.channels == 1 else 0.5

    learning_rate = 1.0e-2
    loss_function = 'elbo'
    optimizer= 'adam'
    epochs = 1

    latent_vector_size = 128
    num_samples = 2000
    gen_name = 'Test_gen_VAE'
    gen_vae_save_path = f'checkpoints/gen_VAE/{gen_name}'
    vae_routine = VAE_Routine(
        data_object,
        learning_rate,
        loss_function,
        optimizer,
        epochs,
#
        latent_vector_size,
        batch_size,
        num_samples,
        gen_vae_save_path
    )
    #vae_routine.train_vae()
    # Note, returned images, reconstructions and gen images are np arrays

    images, reconstructions, labels = vae_routine.reconstruct_test_data()
    ## Check quality of reconstructions:
    #print('CHECKING RECONSTRUCTED IMAGES QUALITY')
    print(f'Number of reconstructions: {len(reconstructions)}')
    vae_routine.check_vae_performance(net, verification_tolerance, reconstructions, labels)
#
#
    ## Check quality of generated images
    #print('CHECKING GENERATED IMAGES QUALITY')
    generated_images = vae_routine.generate_samples(data_object)
    print(f'Number of generated images: {len(generated_images)}')
    vae_routine.check_vae_performance(net, verification_tolerance, generated_images)

    """ ANOMALY DETECTOR VAE ROUTINE"""
    #data_object = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=batch_size)
    #number_anom_images_to_show = 16
    #anom_name = 'Test_anom_VAE'
    #anom_vae_save_path = f'checkpoints/anom_VAE/{anom_name}'
    #anom_vae = VAE_Routine(
    #    data_object,
    #    learning_rate,
    #    loss_function,
    #    optimizer,
    #    epochs,
#
    #    latent_vector_size,
    #    batch_size,
    #    num_samples,
    #    anom_vae_save_path
#
    #)
    #anom_vae.train_vae()
    #anom_vae.anomaly_detection(number_anom_images_to_show)

if __name__ == '__main__':
    main()