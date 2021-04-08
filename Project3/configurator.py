from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from Projects.Project3.Autoencoder import Autoencoder
from Projects.Project3.Classifier import Classifier
from Projects.Project3.Encoder import Encoder
from Projects.Project3.Decoder import Decoder
from Projects.Project3.ClassifierHead import ClassifierHead
from Projects.Project3.Trainer import Trainer
from Projects.Project3.Dataloader import load_fashion_mnist, load_mnist, load_cifar10, load_kmnist
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

    elif dataset_name == 'MNIST':
        # Loading MNIST
        num_classes = 10
        image_dimensions = (1, 28, 28)

        dataloaders = load_mnist(
            batch_size=batch_size,
            D1_fraction= D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[1]
        )
    elif dataset_name == 'CIFAR10':
        # Loading CIFAR10
        num_classes = 10
        image_dimensions = (3, 28, 28)

        dataloaders = load_cifar10(
            batch_size=batch_size,
            D1_fraction=D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[1]
        )
    elif dataset_name == 'KMNIST':
        num_classes = 10
        image_dimensions = (1, 28, 28)

        dataloaders = load_kmnist(
            batch_size=batch_size,
            D1_fraction=D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[1]
        )

    else:
        raise NotImplementedError('Dataset not implemented yet')

    return dataloaders, image_dimensions,  num_classes


def run_SSN_training_regime(
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
    encoder = Encoder(
        input_shape=image_dimensions,
        num_filters=16,
        last_layer_dim=(32,10,10),
        latent_vector_size=latent_vector_size)
    decoder = Decoder(
        input_size=latent_vector_size,
        encoder_last_layer_dim=encoder.last_layer_dim,
        hidden_filters=encoder.num_filters,
        output_size=image_dimensions)
    untrained_lv_and_c = get_latent_vector_and_classes(encoder, dataloaders[-1])

    Autoencoder_model = Autoencoder(encoder, decoder, image_dimensions)

    Autoencoder_trainer = Trainer(
        batch_size=batch_size,
        lr=autoencoder_learning_rate,
        epochs=autoencoder_epochs,
        model=Autoencoder_model,
        dataloaders=dataloaders,
        loss_function=autoencoder_loss_function,
        optimizer=autoencoder_optimizer)

    Autoencoder_trainer.do_autoencoder_train()
    autoencoder_lv_and_c = get_latent_vector_and_classes(Autoencoder_model.encoder, dataloaders[-1])

    plot_autoencoder_training(Autoencoder_trainer)

    # Show autoencoders reconstruction of images of training data
    make_reconstructions_figure(Autoencoder_model, dataloaders[1], n_reconstructions_to_display, batch_size, image_dimensions)

    classifier_head = ClassifierHead(latent_vector_size, num_classes)

    SSN_model = Classifier(Autoencoder_model.encoder, classifier_head, num_classes)

    if freeze_encoder_weights:
        # TODO find best practice for freezing encoder weights
        # SSN_model.encoder.freeze_weights()
        # or something like this but using the proper way to do this according to pytorch API
        pass

    SSN_trainer = Trainer(
        batch_size=batch_size,
        lr=classifier_learning_rate,
        epochs=classifier_epochs,
        model=SSN_model,
        dataloaders=dataloaders,
        loss_function=classifier_loss_function,
        optimizer=classifier_optimizer)

    SSN_trainer.do_classifier_train()

    classifier_lv_and_c = get_latent_vector_and_classes(SSN_model.encoder, dataloaders[-1])

    if plot_t_sne:
        create_t_sne_figure(untrained_lv_and_c, autoencoder_lv_and_c, classifier_lv_and_c)

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
    encoder = Encoder(
        input_shape=image_dimensions,
        num_filters=16,
        last_layer_dim=(32, 10, 10),
        latent_vector_size=latent_vector_size)
    classifier_head = ClassifierHead(latent_vector_size, num_classes)
    SCN_model = Classifier(encoder, classifier_head, num_classes)

    SCN_trainer = Trainer(
        batch_size=batch_size,
        lr=learning_rate,
        epochs= epochs,
        model= SCN_model,
        dataloaders=dataloaders,
        loss_function=loss_function,
        optimizer=optimizer)

#
    # Train network
    SCN_trainer.do_classifier_train()
    return SCN_trainer


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
        reconstruction_batch = reconstruction_batch.view(batch_size, image_dimensions[0], image_dimensions[1], image_dimensions[2])
        reconstruction_batch = reconstruction_batch.detach().numpy()
        reconstructions.extend(reconstruction_batch)
    images = images[:num_images]
    reconstructions = reconstructions[:num_images]
    labels = labels[:num_images]

    visualisations.show_images_and_reconstructions(images, reconstructions, labels)
    plt.show()




def get_latent_vector_and_classes(encoder: nn.Module, data):
    latent_vectors = []
    classes = []
    for X_batch, Y_batch in data:
        latent_vectors_batch = encoder(X_batch)
        latent_vectors.extend(latent_vectors_batch.detach().numpy())
        classes.extend(Y_batch.detach().numpy())

    return latent_vectors, classes


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

def plot_autoencoder_training(autoencoder_trainer):
    plt.figure(figsize=(10, 8))
    plt.title('Autoencoder loss')
    visualisations.plot_metric(autoencoder_trainer.train_history['loss'], label= 'Autoencoder training loss', averaged_plot=True)
    visualisations.plot_metric(autoencoder_trainer.validation_history['loss'], label='Autoencoder validation loss', averaged_plot=False)
    #plt.ylim(bottom=0, top=1)
    plt.legend()

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
    plt.ylim(bottom=0, top=2.5)

    plt.subplot(1, 2, 2)
    plt.title('SCN and SSN accuracy')
    visualisations.plot_metric(SSN_trainer.train_history['accuracy'], label='SSN Training accuracy', averaged_plot=True)
    visualisations.plot_metric(SSN_trainer.validation_history['accuracy'], label='SSN Validation accuracy', averaged_plot=False)
    visualisations.plot_metric(SCN_trainer.train_history['accuracy'], label='SCN Training accuracy', averaged_plot=True)
    visualisations.plot_metric(SCN_trainer.validation_history['accuracy'], label='SCN Validation accuracy', averaged_plot=False)
    plt.legend()

    plt.ylim(bottom=0, top=1)
    plt.show()

def main():
    # Global parameters for both SSN and SCN
    dataset_name = 'MNIST'  # AVAILABLE  #'CIFAR10' # 'KMNIST' #'FashionMNIST' #
    D1_fraction = 0.8
    D2_train_val_test_fraction = (0.1, 0.1)

    latent_vector_size = 64
    batch_size = 16

    # Parameters for Autoencoder training
    autoencoder_learning_rate = 0.2
    autoencoder_loss_function = 'binary_cross_entropy'  # AVAILABLE 'binary_cross_entropy'
    autoencoder_optimizer = 'SGD'  # AVAILABLE 'SGD' #'adam' #
    autoencoder_epochs = 3

    freeze_encoder_weights = True
    plot_t_sne = True
    n_reconstructions_to_display = 10

    # Parameters for classifier training
    classifier_learning_rate = 0.0002
    classifier_loss_function = 'cross_entropy'  # AVAILABLE 'cross_entropy'
    classifier_optimizer = 'SGD'  # AVAILABLE 'SGD' #'adam'
    classifier_epochs = 10

    dataloaders, \
        image_dimensions,\
        num_classes \
        = get_and_split_dataset(dataset_name, D1_fraction, D2_train_val_test_fraction, batch_size)

    SSN_trainer = run_SSN_training_regime(
        dataloaders,
        autoencoder_learning_rate,
        autoencoder_loss_function,
        autoencoder_optimizer,
        autoencoder_epochs,

        classifier_learning_rate,
        classifier_loss_function,
        classifier_optimizer,
        classifier_epochs,

        latent_vector_size,
        batch_size,
        num_classes,
        image_dimensions,

        freeze_encoder_weights,
        n_reconstructions_to_display,
        plot_t_sne
    )

    SCN_trainer = run_SCN_training_regime(
        dataloaders,
        classifier_learning_rate,
        classifier_loss_function,
        classifier_optimizer,
        latent_vector_size,
        classifier_epochs,
        batch_size,
        image_dimensions,
        num_classes
    )
    compare_SSN_and_SCN(SSN_trainer, SCN_trainer)


if __name__ == '__main__':
    main()