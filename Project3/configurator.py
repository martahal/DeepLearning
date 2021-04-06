from torch import nn
import matplotlib.pyplot as plt

from Projects.Project3.Autoencoder import Autoencoder
from Projects.Project3.Classifier import Classifier
from Projects.Project3.Encoder import Encoder
from Projects.Project3.ClassifierHead import ClassifierHead

from Projects.Project3.Decoder import Decoder
from Projects.Project3.Trainer import Trainer
from Projects.Project3.Dataloader import load_fashion_mnist
from Projects.Project3 import visualisations


def get_and_split_dataset(
        dataset_name,
        D1_fraction: float,
        D2_train_val_test_fraction: tuple,
        batch_size
):
    if dataset_name == 'FashionMNIST':
        #Loading FashionMNIST
        num_classes = 10
        dataloaders = load_fashion_mnist(
            batch_size=batch_size,
            D1_fraction=D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[0]
        )
        image_dimensions = (1, 28, 28)

    else:
        # TODO implement the other datasets
        raise NotImplementedError('Dataset not implemented yet')

    return dataloaders, image_dimensions, num_classes


def run_SSN_training_regime(
        dataloaders,
        learning_rate: float,
        loss_function: str,
        optimizer: str,
        latent_vector_size: int,
        epochs: int,
        batch_size: int,
        num_classes: int,
        image_dimensions: tuple,
        freeze_encoder_weights: bool,
        n_reconstructions_to_display: int,
        plot_t_sne: bool
):
    encoder = Encoder(
        input_shape=image_dimensions,
        num_filters=4,
        latent_vector_size=latent_vector_size)
    decoder = Decoder(
        input_size=latent_vector_size,
        encoder_last_layer_dim=encoder.last_layer_dim,
        output_size=image_dimensions)
    Autoencoder_model = Autoencoder(encoder, decoder, image_dimensions)

    Autoencoder_trainer = Trainer(batch_size=batch_size,
                          lr=learning_rate,
                          epochs=epochs,
                          model=Autoencoder_model,
                          dataloaders=dataloaders,
                          loss_function=loss_function,
                          optimizer=optimizer)

    Autoencoder_trainer.do_autoencoder_train()

    classifier_head = ClassifierHead(latent_vector_size, num_classes)

    # TODO Find best way to get sub-modules of modules i.e. get encoder module from autoencoder
    SSN_model = Classifier(Autoencoder_model.encoder, classifier_head, num_classes)

    if freeze_encoder_weights:
        # TODO find best practice for freezing encoder weights
        # SSN_model.encoder.freeze_weights()
        # or something like this but using the proper way to do this according to pytorch API
        pass

    SSN_trainer = Trainer(batch_size=batch_size,
                          lr=learning_rate,
                          epochs=epochs,
                          model=SSN_model,
                          dataloaders=dataloaders,
                          loss_function=loss_function,
                          optimizer=optimizer)

    SSN_trainer.do_classifier_train()

    if plot_t_sne:
        # TODO Creating the figure works, but i suspect the encoders are all the same encoder getting updated
        create_t_sne_figure(encoder, Autoencoder_model.encoder, SSN_model.encoder, dataloaders[-1])

    return SSN_trainer


def run_SCN_training_regime(
        dataloaders,
        learning_rate: float,
        loss_function: str,
        optimizer: str,
        latent_vector_size: int,
        epochs: int,
        batch_size: int,
        image_dimensions: tuple,
        num_classes: int
):
    encoder = Encoder(input_shape=image_dimensions, num_filters=4, latent_vector_size=latent_vector_size)
    classifier_head = ClassifierHead(latent_vector_size, num_classes)
    SCN_model = Classifier(encoder, classifier_head, num_classes)

    SCN_trainer = Trainer(batch_size=batch_size,
                          lr=learning_rate,
                          epochs= epochs,
                          model= SCN_model,
                          dataloaders=dataloaders,
                          loss_function=loss_function,
                          optimizer=optimizer)
    ## create pre-training t-SNE plot
#
    ## Train network
    SCN_trainer.do_classifier_train()
    return SCN_trainer
#
    ## TODO restructure the way of making plots
    ## Create accuracy and loss plots
    ##loss_and_accuracy_plots(SCN_trainer)
#
    ## Create t-SNE plots
    ## TODO First hack a plot, then dehack to apply plot for untrained, semi-supervised trained and supervised trained
    #visualisation_data = SCN_trainer.d2_test_dataloader
    #visualisations.plot_t_sne(SCN_trainer.model.encoder, visualisation_data)
    #pass


def create_t_sne_figure(untrained_encoder, pretrained_encoder, classifier_encoder, data):
    # TODO Shuffle and evenly select ~ 250 data cases to use in t-SNE plot
    vis_data = data

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    plt.title('Encoder prior to training')
    visualisations.plot_t_sne(untrained_encoder, vis_data)

    plt.subplot(1, 3, 2)
    plt.title('Encoder after semi-supervised training')
    visualisations.plot_t_sne(pretrained_encoder, vis_data)

    plt.subplot(1, 3, 3)
    plt.title('Encoder after classifier training')
    visualisations.plot_t_sne(classifier_encoder, vis_data)

    plt.show()

def compare_SSN_and_SCN(SSN_trainer, SCN_trainer):

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title('SCN and SSN loss')
    visualisations.plot_metric(SSN_trainer.train_history['loss'], label='SSN Training loss', averaged_plot=True)
    visualisations.plot_metric(SSN_trainer.validation_history['loss'], label='SSN Validation loss', averaged_plot=False)
    visualisations.plot_metric(SCN_trainer.train_history['loss'], label='SCN Training loss', averaged_plot=True)
    visualisations.plot_metric(SCN_trainer.validation_history['loss'], label=' SCN Validation loss', averaged_plot=False)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('SCN and SSN accuracy')
    visualisations.plot_metric(SSN_trainer.train_history['accuracy'], label='Training accuracy', averaged_plot=True)
    visualisations.plot_metric(SSN_trainer.validation_history['accuracy'], label='Validation accuracy', averaged_plot=False)
    visualisations.plot_metric(SCN_trainer.train_history['accuracy'], label='Training accuracy', averaged_plot=True)
    visualisations.plot_metric(SCN_trainer.validation_history['accuracy'], label='Validation accuracy', averaged_plot=False)
    plt.legend()

    plt.show()

def main():
    # Global parameters for both SSN and SCN
    dataset_name = 'FashionMNIST'
    D1_fraction = 0.8
    D2_train_val_test_fraction = (0.1, 0.1)

    latent_vector_size = 256
    batch_size = 16

    # Parameters for SSN training
    SSN_learning_rate = 0.0002
    SSN_loss_function = 'binary_cross_entropy'
    SSN_optimizer = 'SGD'
    SSN_epochs = 3

    freeze_encoder_weights = True
    plot_t_sne = True
    n_reconstructors_to_display = 10

    # Parameters for SCN training
    SCN_learning_rate = 0.0002
    SCN_loss_function = 'cross_entropy'
    SCN_optimizer = 'SGD'
    SCN_epochs = 10

    dataloaders, image_dimensions, num_classes = get_and_split_dataset(dataset_name, D1_fraction, D2_train_val_test_fraction, batch_size)

    SSN_trainer = run_SSN_training_regime(dataloaders, SSN_learning_rate, SSN_loss_function, SSN_optimizer, latent_vector_size, SSN_epochs,
                           batch_size, num_classes, image_dimensions,
                           freeze_encoder_weights, n_reconstructors_to_display, plot_t_sne)

    SCN_trainer = run_SCN_training_regime(dataloaders, SCN_learning_rate, SCN_loss_function, SCN_optimizer, latent_vector_size,
                            SCN_epochs, batch_size, image_dimensions, num_classes)
    compare_SSN_and_SCN(SSN_trainer, SCN_trainer)

if __name__ == '__main__':
    main()