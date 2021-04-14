from Projects.Project3.Autoencoder import Autoencoder
from Projects.Project3.Classifier import Classifier
from Projects.Project3.Encoder import Encoder
from Projects.Project3.Decoder import Decoder
from Projects.Project3.Trainer import Trainer
from Projects.Project3.ClassifierHead import ClassifierHead
from Projects.Project3 import visualisations

import matplotlib.pyplot as plt
from torch import nn


class SSN:

    def __init__(
            self,
            dataloaders,
            autoencoder_learning_rate: float,
            autoencoder_loss_function: str,
            autoencoder_optimizer: str,
            autoencoder_epochs: int,
            classifier_learning_rate: float,
            classifier_loss_function: str,
            classifier_optimizer: str,
            classifier_epochs: int,

            latent_vector_size: int,
            batch_size: int,
            num_classes: int,
            image_dimensions: tuple,

            freeze_encoder_weights: bool,
            n_reconstructions_to_display: int,
            plot_t_sne: bool
    ):
        self.dataloaders = dataloaders
        self.image_dimensions = image_dimensions
        self.freeze_encoder_weights = freeze_encoder_weights
        self.n_reconstructions_to_display = n_reconstructions_to_display
        self.plot_t_sne = plot_t_sne

        self.encoder = Encoder(
            input_shape=image_dimensions,
            num_filters=16,
            last_layer_dim=(32,10,10),
            latent_vector_size=latent_vector_size)

        self.decoder = Decoder(
            input_size=latent_vector_size,
            encoder_last_layer_dim=self.encoder.last_layer_dim,
            hidden_filters=self.encoder.num_filters,
            output_size=image_dimensions)

        self.autoencoder = Autoencoder(self.encoder, self.decoder, image_dimensions)

        self.classifier_head = ClassifierHead(
            latent_vector_size,
            num_classes)

        self.classifier = Classifier(
            self.autoencoder.encoder, self.classifier_head, num_classes
        )

        self.autoencoder_trainer = Trainer(
            batch_size=batch_size,
            lr=autoencoder_learning_rate,
            epochs=autoencoder_epochs,
            model=self.autoencoder,
            dataloaders=self.dataloaders,
            loss_function=autoencoder_loss_function,
            optimizer=autoencoder_optimizer)

        self.SSN_trainer = Trainer(
            batch_size=batch_size,
            lr=classifier_learning_rate,
            epochs=classifier_epochs,
            model=self.classifier,
            dataloaders=self.dataloaders,
            loss_function=classifier_loss_function,
            optimizer=classifier_optimizer)

    def run_SSN_training_regime(self):
        untrained_lv_and_c = self.get_latent_vector_and_classes(self.encoder, self.dataloaders[-1])

        self.autoencoder_trainer.do_autoencoder_train()

        autoencoder_lv_and_c = self.get_latent_vector_and_classes(self.autoencoder.encoder, self.dataloaders[-1])

        self.plot_autoencoder_training(self.autoencoder_trainer)

        # Show autoencoders reconstruction of images of training data
        self.make_reconstructions_figure(
            self.autoencoder,
            self.dataloaders[1],
            self.n_reconstructions_to_display,
            self.autoencoder_trainer.batch_size,
            self.image_dimensions)

        if self.freeze_encoder_weights:
            for parameter in self.classifier.encoder.parameters():
                parameter.requires_grad = False
            # TODO find best practice for freezing encoder weights
            # SSN_model.encoder.freeze_weights()
            # or something like this but using the proper way to do this according to pytorch API
            pass

        self.SSN_trainer.do_classifier_train()
        classifier_lv_and_c = self.get_latent_vector_and_classes(self.classifier.encoder, self.dataloaders[-1])

        if self.plot_t_sne:
            self.create_t_sne_figure(
                untrained_lv_and_c,
                autoencoder_lv_and_c,
                classifier_lv_and_c
            )

        #return self.SSN_trainer

    @staticmethod
    def get_latent_vector_and_classes(encoder: nn.Module, data):
        latent_vectors = []
        classes = []
        for X_batch, Y_batch in data:
            latent_vectors_batch = encoder(X_batch)
            latent_vectors.extend(latent_vectors_batch.detach().numpy())
            classes.extend(Y_batch.detach().numpy())
        #returns only the first 250 latent vectors
        return latent_vectors[:250], classes[:250]

    @staticmethod
    def make_reconstructions_figure(autoencoder, vis_data, num_images, batch_size, image_dimensions):
        # Extremely inefficient way of doing this
        # Forward all images, then selecting the ones i want to visualize
        images = []
        reconstructions = []
        labels = []
        for X_batch, Y_batch in vis_data:
            images.extend(X_batch)
            labels.extend(Y_batch)
            reconstruction_batch, aux = autoencoder(X_batch)
            reconstruction_batch = reconstruction_batch.view(batch_size, image_dimensions[0], image_dimensions[1],
                                                             image_dimensions[2])
            reconstruction_batch = reconstruction_batch.detach().numpy()
            reconstructions.extend(reconstruction_batch)
        images = images[:num_images]
        reconstructions = reconstructions[:num_images]
        labels = labels[:num_images]

        visualisations.show_images_and_reconstructions(images, reconstructions, labels)
        plt.show()

    @staticmethod
    def create_t_sne_figure(
            untrained_latent_vectors_and_classes,
            pretrained_latent_vectors_and_classes,
            classifier_latent_vectors_and_classes
    ):

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 3, 1)
        plt.title('Encoder prior to training')
        visualisations.plot_t_sne(untrained_latent_vectors_and_classes)

        plt.subplot(1, 3, 2)
        plt.title('Encoder after semi-supervised training')
        visualisations.plot_t_sne(pretrained_latent_vectors_and_classes)

        plt.subplot(1, 3, 3)
        plt.title('Encoder after classifier training')
        visualisations.plot_t_sne(classifier_latent_vectors_and_classes)

        plt.show()

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

        plt.show()
